import pytest

from rag_customer_review_chatbot.src.ingestion.loader import ReviewDocument
from rag_customer_review_chatbot.src.retriever.vectorstore import create_vector_store


def _sample_documents():
    return [
        ReviewDocument(
            doc_id="1_0",
            text="The battery life of this vacuum is outstanding",
            metadata={"review_id": "1", "rating": 5},
        ),
        ReviewDocument(
            doc_id="2_0",
            text="Terrible battery life and loud noise",
            metadata={"review_id": "2", "rating": 1},
        ),
    ]


def test_create_vector_store_and_query():
    documents = _sample_documents()
    store = create_vector_store(documents, min_df=1)

    results = store.query("battery life", top_k=1)
    assert results
    assert results[0].document.doc_id in {"1_0", "2_0"}


def test_query_rejects_invalid_input():
    documents = _sample_documents()
    store = create_vector_store(documents, min_df=1)

    with pytest.raises(ValueError):
        store.query(" ", top_k=1)

    with pytest.raises(ValueError):
        store.query("battery", top_k=0)


def test_create_vector_store_validates_parameters():
    documents = _sample_documents()

    with pytest.raises(ValueError):
        create_vector_store(documents, min_df=0)

    with pytest.raises(ValueError):
        create_vector_store(documents, max_features=0)

    with pytest.raises(ValueError):
        create_vector_store(documents, ngram_range=(2, 1))


def test_review_chatbot_handles_top_k_larger_than_corpus():
    documents = _sample_documents()
    store = create_vector_store(documents, min_df=1)
    results = store.query("battery life", top_k=10)
    assert len(results) <= len(documents)
