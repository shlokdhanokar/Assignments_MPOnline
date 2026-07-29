"""RAG Chatbot — Retrieval-Augmented Generation over a local knowledge base.

Pipeline: ingest -> chunk -> index (hybrid sparse + dense) -> retrieve -> rerank
(MMR) -> generate -> cite.

Runs with NO API key and NO model download: retrieval uses TF-IDF plus LSA
(truncated SVD) from scikit-learn, and answers are composed extractively from the
retrieved passages. If ANTHROPIC_API_KEY is set and the `anthropic` package is
installed, generation is upgraded to Claude, grounded on the same retrieved
context. Retrieval is identical either way, so the retrieval evaluation below is
reproducible on any machine.

Usage:
    python rag_chatbot.py                 # interactive chat
    python rag_chatbot.py --evaluate      # run the retrieval evaluation suite
    python rag_chatbot.py --ask "..."     # single question
"""
from __future__ import annotations

import argparse
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


KB_DIR = "knowledge_base"
CHUNK_WORDS = 110
CHUNK_OVERLAP = 35
TOP_K = 4
CANDIDATE_K = 12
SPARSE_WEIGHT = 0.6          # lexical vs semantic blend for hybrid retrieval
MMR_LAMBDA = 0.7             # 1.0 = pure relevance, 0.0 = pure diversity
MIN_SCORE = 0.05             # below this the corpus has nothing useful to say
SVD_COMPONENTS = 60
CLAUDE_MODEL = "claude-opus-5"

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about machine learning. "
    "Answer ONLY from the numbered context passages provided. Cite the passages "
    "you use as [1], [2] etc. If the context does not contain the answer, say so "
    "plainly instead of guessing — do not use knowledge from outside the context."
)


@dataclass
class Chunk:
    text: str
    source: str
    heading: str
    index: int


def load_documents(kb_path: Path) -> list[tuple[str, str]]:
    files = sorted(kb_path.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md documents found in {kb_path}")
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


def chunk_document(name: str, text: str) -> list[Chunk]:
    """Split on headings first, then into overlapping word windows.

    Splitting on headings keeps semantically related sentences together; the
    overlap stops a relevant passage being cut in half at a window boundary.
    """
    chunks: list[Chunk] = []
    sections = re.split(r"\n(?=#{1,6}\s)", text)

    for section in sections:
        lines = section.strip().split("\n")
        if not lines or not lines[0].strip():
            continue
        heading = lines[0].lstrip("#").strip() if lines[0].startswith("#") else "Introduction"
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        if not body:
            continue

        words = body.split()
        step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
        for start in range(0, len(words), step):
            window = words[start:start + CHUNK_WORDS]
            if len(window) < 25 and chunks and chunks[-1].source == name:
                break   # trailing fragment: already covered by the overlap
            chunks.append(Chunk(" ".join(window), name, heading, len(chunks)))
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks


class RagIndex:
    """Hybrid retriever: TF-IDF lexical matching + LSA semantic matching."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus = [f"{c.heading}. {c.text}" for c in chunks]

        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1
        )
        self.sparse = self.vectorizer.fit_transform(corpus)

        # LSA: truncated SVD over the TF-IDF matrix gives dense vectors that match
        # paraphrases sharing no vocabulary — no model download required.
        n_components = min(SVD_COMPONENTS, max(2, self.sparse.shape[1] - 1),
                           max(2, self.sparse.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.dense = normalize(self.svd.fit_transform(self.sparse))
        self.explained = float(self.svd.explained_variance_ratio_.sum())

    def _scores(self, query: str) -> np.ndarray:
        q_sparse = self.vectorizer.transform([query])
        lexical = cosine_similarity(q_sparse, self.sparse).ravel()
        q_dense = normalize(self.svd.transform(q_sparse))
        semantic = cosine_similarity(q_dense, self.dense).ravel()
        return SPARSE_WEIGHT * lexical + (1 - SPARSE_WEIGHT) * semantic

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[tuple[Chunk, float]]:
        scores = self._scores(query)
        candidates = list(np.argsort(-scores)[:CANDIDATE_K])

        # Maximal Marginal Relevance: pick the most relevant chunk, then each
        # subsequent one for relevance minus similarity to what is already chosen,
        # so the context window is not filled with near-duplicate passages.
        selected: list[int] = []
        while candidates and len(selected) < top_k:
            if not selected:
                best = max(candidates, key=lambda i: scores[i])
            else:
                chosen = self.dense[selected]
                best, best_value = None, -np.inf
                for i in candidates:
                    redundancy = float(np.max(self.dense[i] @ chosen.T))
                    value = MMR_LAMBDA * scores[i] - (1 - MMR_LAMBDA) * redundancy
                    if value > best_value:
                        best, best_value = i, value
            selected.append(best)
            candidates.remove(best)

        return [(self.chunks[i], float(scores[i])) for i in selected]


def build_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(
        f"[{n}] (source: {c.source} — {c.heading})\n{c.text}"
        for n, (c, _) in enumerate(hits, start=1)
    )


def extractive_answer(query: str, hits: list[tuple[Chunk, float]]) -> str:
    """Offline fallback: score each sentence of the retrieved passages against
    the query and return the best-supported ones, with citations."""
    query_terms = {w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 3}
    scored: list[tuple[float, str, int]] = []

    for n, (chunk, chunk_score) in enumerate(hits, start=1):
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
            words = {w for w in re.findall(r"[a-z]+", sentence.lower()) if len(w) > 3}
            if not words or len(sentence.split()) < 6:
                continue
            overlap = len(query_terms & words) / max(1, len(query_terms))
            scored.append((overlap + 0.25 * chunk_score, sentence.strip(), n))

    scored.sort(key=lambda t: -t[0])
    picked = [(s, n) for score, s, n in scored[:3] if score > 0]
    if not picked:
        return "The knowledge base does not contain enough information to answer that."

    seen, parts = set(), []
    for sentence, n in picked:
        if sentence in seen:
            continue
        seen.add(sentence)
        parts.append(f"{sentence} [{n}]")
    return " ".join(parts)


def claude_answer(query: str, context: str) -> str | None:
    """Upgrade generation to Claude when a key and the SDK are both available."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Context passages:\n\n{context}\n\nQuestion: {query}",
            }],
        )
        if response.stop_reason == "refusal":
            return None
        return "".join(b.text for b in response.content if b.type == "text").strip()
    except Exception as error:  # noqa: BLE001 — fall back rather than crash the chat
        print(f"  [Claude unavailable ({type(error).__name__}); using extractive answer]")
        return None


def answer_question(index: RagIndex, query: str, show_sources: bool = True) -> str:
    hits = index.retrieve(query)
    if not hits or hits[0][1] < MIN_SCORE:
        return ("I don't have information about that in my knowledge base. "
                "Try asking about supervised learning, neural networks, clustering, "
                "recommendation systems, reinforcement learning, or RAG.")

    context = build_context(hits)
    body = claude_answer(query, context) or extractive_answer(query, hits)

    if not show_sources:
        return body

    sources = "\n".join(
        f"  [{n}] {c.source} — {c.heading}  (relevance {score:.3f})"
        for n, (c, score) in enumerate(hits, start=1)
    )
    return f"{body}\n\nSources:\n{sources}"


# ------------------------------------------------------------------ evaluation
# Each case: question -> the document that genuinely contains the answer.
EVAL_SET = [
    ("What is the difference between precision and recall?", "supervised_learning.md"),
    ("Why is accuracy misleading on imbalanced datasets?", "supervised_learning.md"),
    ("How does a random forest reduce variance?", "supervised_learning.md"),
    ("What does the kernel trick do in an SVM?", "supervised_learning.md"),
    ("Why do neural networks need hidden layers?", "neural_networks.md"),
    ("What is dropout used for?", "neural_networks.md"),
    ("How does a CNN differ from a dense network on images?", "neural_networks.md"),
    ("What does it mean when validation loss rises but training loss falls?", "neural_networks.md"),
    ("Why normalise pixel values before training?", "neural_networks.md"),
    ("How do I choose the number of clusters in K-Means?", "unsupervised_and_rl.md"),
    ("What does PCA actually do?", "unsupervised_and_rl.md"),
    ("What is the cold start problem?", "unsupervised_and_rl.md"),
    ("Why does DQN need a replay buffer and a target network?", "unsupervised_and_rl.md"),
    ("Why is feature scaling needed before distance based methods?", "unsupervised_and_rl.md"),
    ("What problem does retrieval augmented generation solve?", "rag_and_llms.md"),
    ("How should documents be chunked for retrieval?", "rag_and_llms.md"),
    ("What is Maximal Marginal Relevance for?", "rag_and_llms.md"),
    ("How do you measure whether retrieval is working?", "rag_and_llms.md"),
    ("Why use cosine similarity for text?", "rag_and_llms.md"),
    ("What are the limitations of RAG?", "rag_and_llms.md"),
]


def evaluate(index: RagIndex) -> None:
    print("=" * 72)
    print("RETRIEVAL EVALUATION")
    print("=" * 72)
    print(f"{len(EVAL_SET)} questions, each labelled with the document that "
          "genuinely contains the answer.\n")

    hits_at_1 = hits_at_3 = 0
    reciprocal_ranks: list[float] = []
    failures: list[tuple[str, str, str]] = []

    for question, expected in EVAL_SET:
        results = index.retrieve(question, top_k=5)
        sources = [c.source for c, _ in results]

        rank = sources.index(expected) + 1 if expected in sources else 0
        if rank == 1:
            hits_at_1 += 1
        if 0 < rank <= 3:
            hits_at_3 += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        if rank != 1:
            failures.append((question, expected, sources[0]))

    n = len(EVAL_SET)
    print(f"Hit Rate @1 : {hits_at_1}/{n}  ({hits_at_1 / n:.1%})")
    print(f"Hit Rate @3 : {hits_at_3}/{n}  ({hits_at_3 / n:.1%})")
    print(f"MRR         : {np.mean(reciprocal_ranks):.4f}")
    print()

    if failures:
        print("Questions whose top result came from another document:")
        for question, expected, got in failures:
            print(f"  - {question}")
            print(f"      expected {expected}, top hit was {got}")
        print()
        print("Note: the documents overlap by design (scaling is discussed in two of")
        print("them), so a 'miss' here often still returns a passage that answers the")
        print("question — this metric is deliberately strict.")
    else:
        print("Every question retrieved its source document at rank 1.")
    print()


def print_banner(index: RagIndex, docs: int) -> None:
    generation = ("Claude (ANTHROPIC_API_KEY detected)"
                  if os.environ.get("ANTHROPIC_API_KEY") else "extractive (offline)")
    print("=" * 72)
    print("RAG CHATBOT — Capstone Project")
    print("=" * 72)
    print(f"Documents      : {docs}")
    print(f"Chunks indexed : {len(index.chunks)}")
    print(f"Retrieval      : hybrid — {SPARSE_WEIGHT:.0%} TF-IDF lexical + "
          f"{1 - SPARSE_WEIGHT:.0%} LSA semantic")
    print(f"LSA components : {index.svd.n_components} "
          f"({index.explained:.1%} of TF-IDF variance retained)")
    print(f"Reranking      : MMR (lambda={MMR_LAMBDA}), top {TOP_K} of "
          f"{CANDIDATE_K} candidates")
    print(f"Generation     : {generation}")
    print("=" * 72)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG chatbot over a local knowledge base.")
    parser.add_argument("--evaluate", action="store_true", help="run the retrieval evaluation")
    parser.add_argument("--ask", type=str, help="answer a single question and exit")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    documents = load_documents(project_dir / KB_DIR)

    chunks: list[Chunk] = []
    for name, text in documents:
        chunks.extend(chunk_document(name, text))
    index = RagIndex(chunks)

    if args.evaluate:
        print_banner(index, len(documents))
        evaluate(index)
        return

    if args.ask:
        print_banner(index, len(documents))
        print(f"Q: {args.ask}\n")
        print(answer_question(index, args.ask))
        return

    print_banner(index, len(documents))
    print("Ask a question, or type 'exit' to quit. Try:")
    print("  - Why do neural networks need hidden layers?")
    print("  - What problem does retrieval augmented generation solve?")
    print("  - How do I choose the number of clusters in K-Means?")
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit", "bye"}:
            print("Goodbye.")
            return
        print()
        print(textwrap.indent(answer_question(index, question), "Bot: ",
                              lambda line: line.startswith(("The ", "I "))))
        print()


if __name__ == "__main__":
    main()
