# LLM Projects Repository

This repository aggregates learning materials and hands-on experiments related to large language models. In addition to the existing course content, it now includes a fully working Retrieval-Augmented Generation (RAG) project for answering questions about customer review datasets.

## Projects

### `rag_customer_review_chatbot/`

A modular RAG pipeline that ingests customer reviews, builds a TF-IDF semantic index, and exposes a command-line chatbot for interactive question answering. The project ships with:

- Data ingestion and chunking utilities (`src/ingestion/loader.py`).
- A scikit-learn based vector store (`src/retriever/vectorstore.py`).
- An orchestration layer with an interactive CLI (`src/app/chatbot.py`).
- Configuration defaults (`config/default.yaml`) and a sample dataset under `data/`.
- A starter notebook for exploratory data analysis (`notebooks/01_eda.ipynb`).

Refer to the project README for setup instructions and extension ideas.
