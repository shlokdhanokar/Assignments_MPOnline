# Retrieval-Augmented Generation and Large Language Models

## What RAG is

Retrieval-Augmented Generation combines a retrieval system with a language model. Instead of relying only on knowledge baked into a model's weights, the system first retrieves relevant passages from an external corpus and then conditions the generated answer on those passages.

## Why RAG is used

A language model's parametric knowledge is fixed at training time, cannot be updated without retraining, and gives no way to check where an answer came from. RAG addresses three problems at once: it grounds answers in a source that can be cited, it lets the knowledge base be updated instantly by editing documents, and it reduces hallucination because the model is asked to answer from supplied evidence rather than memory.

## The pipeline

Ingestion splits source documents into chunks. Chunk size is a trade-off: chunks that are too small lose the context needed to answer, while chunks that are too large dilute the relevant sentence among irrelevant text and waste context budget. Overlapping chunks avoid cutting a relevant passage in half at a boundary.

Indexing converts each chunk into a vector. Sparse lexical methods such as TF-IDF and BM25 match on exact terms and are strong when the query shares vocabulary with the document. Dense embedding methods place semantically similar text near each other, so they match paraphrases that share no words. Hybrid retrieval combines both and generally outperforms either alone.

Retrieval embeds the user query with the same method and finds the nearest chunks, usually by cosine similarity. Maximal Marginal Relevance re-ranks the candidates to balance relevance against diversity, preventing the context window from being filled with several near-identical passages.

Generation places the retrieved chunks into a prompt with the question, and instructs the model to answer only from the provided context and to say so when the context is insufficient.

## Evaluating a RAG system

Retrieval and generation are evaluated separately. Retrieval quality is measured with hit rate at k, which asks whether the correct document appears in the top k results, and with Mean Reciprocal Rank, which rewards placing it higher. Generation quality is measured on faithfulness — whether every claim is supported by the retrieved context — and answer relevance.

A failure can come from either stage. If the right passage was never retrieved, no amount of prompt engineering fixes the answer.

## Limitations

RAG cannot answer what is not in the corpus. It is sensitive to chunking strategy, and retrieval that returns plausible but wrong passages can make a confident, wrong answer more likely rather than less.

## Embeddings and vector similarity

An embedding maps text to a vector so that similar meanings sit close together. Cosine similarity measures the angle between two vectors and is preferred over Euclidean distance for text because it ignores document length. Latent Semantic Analysis produces dense vectors by applying truncated Singular Value Decomposition to a TF-IDF matrix, capturing which terms co-occur and so matching related words that never appear together in the same chunk.
