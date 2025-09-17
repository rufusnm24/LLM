# RAG Customer Review Chatbot

This project provides a Retrieval-Augmented Generation (RAG) pipeline that turns a corpus of customer reviews into a question-answering (QA) experience. It focuses on keeping the setup lightweight while still showcasing the essential building blocks of a production-grade RAG stack:

1. **Ingestion** – clean and normalize raw review data.
2. **Chunking** – split long reviews into retrievable passages.
3. **Vector Store** – create a semantic index over the passages.
4. **Chatbot Orchestration** – retrieve supporting evidence and compose a natural-language answer.

The code base is intentionally modular, so you can swap components (e.g. embedding models or LLM backends) as your needs evolve.

## Project Structure

```
rag_customer_review_chatbot/
├── config/
│   └── default.yaml
├── data/
│   └── .gitkeep
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── app/
│   │   └── chatbot.py
│   ├── ingestion/
│   │   └── loader.py
│   └── retriever/
│       └── vectorstore.py
├── tests/
│   ├── conftest.py
│   ├── test_chatbot.py
│   ├── test_loader.py
│   └── test_vectorstore.py
├── README.md
└── requirements.txt
```

## Getting Started

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install --upgrade pip
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your dataset

Add a CSV file under `data/` with at least the following columns:

| Column | Description |
| --- | --- |
| `review_id` | Unique identifier for each review. |
| `product_id` | Identifier of the product being reviewed. |
| `review_text` | The free-form review content. |
| `rating` | Numerical rating (e.g. 1-5). |
| `review_date` | ISO timestamp or date string (optional but useful). |

You can add additional metadata columns as needed. The ingestion utilities automatically preserve them as document metadata.

### 4. Run the CLI demo

```bash
python -m rag_customer_review_chatbot.src.app.chatbot \
    --data-path data/sample_reviews.csv \
    --top-k 3
```

The CLI loads the dataset, builds a TF-IDF vector index, and launches an interactive shell where you can ask questions about the reviews. Type `exit` or press `Ctrl+D` to quit.

### 5. Run the automated checks

Execute the project's unit tests to validate the ingestion, retrieval, and chatbot orchestration flows:

```bash
pytest rag_customer_review_chatbot/tests
```

Running the suite regularly is the fastest way to ensure production readiness when you adjust configurations or extend the code.

### 6. Explore the notebook

### 5. Explore the notebook


Open `notebooks/01_eda.ipynb` in JupyterLab or VS Code to explore the dataset, inspect rating distributions, and validate text quality before feeding it to the RAG pipeline.

## Configuration

Default parameters live in `config/default.yaml`. You can override them by passing a custom YAML file via `--config-path` when running the CLI, or by loading them programmatically:

```python
from rag_customer_review_chatbot.src.app.chatbot import load_config
config = load_config("config/default.yaml")
```

The configuration controls chunk sizes, vectorizer settings, and response formatting heuristics.

## Extending the Project

- Replace the TF-IDF vector store with embeddings from libraries such as `sentence-transformers` or `langchain`.
- Integrate an LLM provider (OpenAI, Azure, Anthropic, etc.) to generate richer answers using the retrieved context.
- Deploy the chatbot as an API or streamlit app by reusing the `ReviewChatbot` class.

## License

This project is provided as-is for educational purposes. Feel free to adapt it for commercial use, but review the licenses of third-party dependencies.
