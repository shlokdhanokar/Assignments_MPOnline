# RAG Chatbot — Capstone Project

A Retrieval-Augmented Generation chatbot that answers machine-learning questions from a local knowledge base, **with citations**, and refuses to answer what the corpus does not cover.

**It runs with no API key and no model download.** Retrieval uses TF-IDF + LSA from scikit-learn; answers are composed extractively from retrieved passages. If `ANTHROPIC_API_KEY` is set, generation upgrades to Claude — grounded on the *same* retrieved context, so the retrieval metrics below reproduce on any machine.

## Dataset / Knowledge Base

The corpus is authored for this project and ships in [`knowledge_base/`](knowledge_base/) — 4 Markdown documents covering supervised learning, neural networks, unsupervised learning + RL, and RAG/LLMs. No external dataset download is required.

Reference material behind the corpus:
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — https://arxiv.org/abs/2005.11401
- scikit-learn feature extraction — https://scikit-learn.org/stable/modules/feature_extraction.html
- Anthropic API documentation — https://platform.claude.com/docs

## Libraries Used
numpy · scikit-learn · anthropic *(optional — only for the Claude generation path)*

## Architecture

```
knowledge_base/*.md
      │
      ▼  heading-aware chunking (110 words, 35-word overlap)
   28 chunks
      │
      ├─▶ TF-IDF (1-2 grams, sublinear tf)  ──┐  lexical  (60%)
      └─▶ Truncated SVD → LSA dense vectors ──┤  semantic (40%)
                                              ▼
                                    hybrid score
                                              │
                                              ▼  top 12 candidates
                                    MMR rerank (λ=0.7)
                                              │
                                              ▼  top 4 passages
                          ┌───────────────────┴───────────────────┐
                          ▼                                       ▼
              Claude (if API key set)                extractive (offline)
                          └───────────────────┬───────────────────┘
                                              ▼
                                  answer + [n] citations + sources
```

### Design decisions

**Heading-aware chunking with overlap.** Splitting on Markdown headings keeps semantically related sentences together; a 35-word overlap stops a relevant passage being cut in half at a window boundary.

**Hybrid retrieval, not just TF-IDF.** Lexical matching fails on paraphrases that share no vocabulary with the source. LSA (truncated SVD over the TF-IDF matrix) gives dense vectors that capture term co-occurrence — semantic matching **without downloading a transformer**, which keeps the project runnable offline. The blend is 60/40 lexical/semantic.

**MMR reranking.** Taking the top-4 by score alone tends to return four near-identical passages from the same section. Maximal Marginal Relevance (λ=0.7) picks each subsequent passage for relevance *minus* similarity to what is already selected, so the context window carries four genuinely different pieces of evidence.

**A refusal path.** If the best passage scores below a threshold, the bot says it doesn't know instead of forcing an answer out of irrelevant context. This is the single most important behaviour in a RAG system — an ungrounded confident answer is worse than no answer.

**Citations are mandatory.** Every claim carries `[n]`, and the sources block names the file, heading, and relevance score, so any answer can be traced back to its passage.

## Results

### Retrieval evaluation (20 labelled questions)

Each question is labelled with the document that genuinely contains its answer.

| Metric | Score |
|---|---|
| **Hit Rate @1** | **19/20 — 95.0%** |
| **Hit Rate @3** | **20/20 — 100.0%** |
| **MRR** | **0.9667** |

Index: 4 documents → 28 chunks; LSA retains 98.2% of TF-IDF variance in 27 components.

The single @1 miss is *"What does PCA actually do?"*, which returns `neural_networks.md` first. That is a genuine near-miss rather than a failure: both documents discuss dimensionality and scaling, and the correct document is still retrieved at rank 2 — which is why Hit@3 is 100%. The metric is deliberately strict (exact source document at rank 1).

### Sample interaction

```
Q: Why do neural networks need hidden layers?

Hidden layers compose the previous layer's outputs into progressively more
abstract features. [1] Without a hidden layer and a non-linear activation, a
network collapses to a linear model no matter how many layers it has. [1] An
artificial neural network is composed of layers of units. [2]

Sources:
  [1] neural_networks.md — Why hidden layers matter  (relevance 0.418)
  [2] neural_networks.md — Neural Networks and Deep Learning  (relevance 0.382)
  [3] neural_networks.md — Convolutional Neural Networks  (relevance 0.220)
  [4] neural_networks.md — Activation functions  (relevance 0.201)
```

Off-topic guard:

```
Q: What is the capital of France?

I don't have information about that in my knowledge base. Try asking about
supervised learning, neural networks, clustering, recommendation systems,
reinforcement learning, or RAG.
```

## Observations

1. **Retrieval is the bottleneck, not generation.** If the correct passage is never retrieved, no prompt engineering recovers the answer. That is why retrieval is evaluated independently here (Hit@k, MRR) rather than only judging the final text.
2. **Hybrid beats either method alone.** Pure TF-IDF misses paraphrases; pure LSA loses precision on exact technical terms ("MMR", "ReLU"). The 60/40 blend keeps exact-term precision while still matching reworded questions.
3. **MMR changes what reaches the model.** Without it, the four retrieved passages are frequently overlapping windows of the same section — the overlap that protects against boundary cuts also produces near-duplicate candidates.
4. **The refusal path is the most valuable feature.** The off-topic guard is what separates a RAG system from a search box with a language model attached: it makes "I don't know" a first-class output.

## Conclusion

This capstone implements the full RAG pipeline end to end — chunking, hybrid indexing, retrieval, MMR reranking, grounded generation, and citation — and evaluates the retrieval stage quantitatively at **95% Hit@1 / 100% Hit@3 / 0.967 MRR** across 20 labelled questions.

The design priority was that the system be **verifiable and reproducible**: it runs with no API key, no model download, and no network access, so its retrieval numbers can be reproduced by anyone. The Claude path is an upgrade to the *generation* stage only, leaving retrieval — and therefore the measured metrics — unchanged.

The main limitation is inherent to RAG: the system cannot answer what is not in the corpus, and retrieval that returns plausible-but-wrong passages can make a confident wrong answer *more* likely, not less. That is precisely why the score threshold and mandatory citations matter. In production the next steps would be a cross-encoder reranker over the MMR candidates, transformer embeddings in place of LSA, and a faithfulness check that verifies each generated claim against its cited passage.

## How to Run

```bash
pip install -r requirements.txt

python rag_chatbot.py                # interactive chat
python rag_chatbot.py --evaluate     # retrieval evaluation suite
python rag_chatbot.py --ask "Why do neural networks need hidden layers?"
```

Optional — upgrade generation to Claude:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...     # Windows: set ANTHROPIC_API_KEY=...
python rag_chatbot.py
```

Without the key the bot runs fully offline; the banner reports which generation mode is active.

## Files

| File | Purpose |
|---|---|
| `rag_chatbot.py` | Full pipeline — chunking, hybrid index, MMR, generation, evaluation |
| `knowledge_base/*.md` | The 4-document corpus |
| `requirements.txt` | Dependencies |
