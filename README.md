<h1 align="center">Code Docs RAG</h1>

This project is meant for experimenting with RAG techniques on code documentation.

Currently using: Prometheus documentation
[prometheus/docs](https://github.com/prometheus/docs/tree/main/docs)

## Features

- Document loading and chunking via LangChain
- Embeddings via Ollama + `nomic_embed_text` + ChromaDB vector store
- Eval via 30 Claude-generated QA pairs based on Prometheus docs

## Eval

Evaluation on QA pairs currently uses a basic keyword matching scheme in retrieved docs via `similarity_search`.

- Given threshold=0.3, retrival hit rate = 83.33%. Adjusting number of retrieved results (k) does not affect `similarity_search_with_score` results, clustered between ~0.47 and ~0.56. This may indicate embedding blind spot or need for advanced retrival including ranking, alternate chunking techniques, etc.