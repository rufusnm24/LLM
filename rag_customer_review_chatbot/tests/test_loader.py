import pandas as pd
import pytest

from rag_customer_review_chatbot.src.ingestion.loader import (
    ReviewDocument,
    build_documents_from_csv,
    chunk_reviews,
    load_reviews,
    preprocess_reviews,
    summarize_documents,
)


def test_load_reviews_missing_file(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_reviews(missing)


def test_preprocess_reviews_validates_column():
    df = pd.DataFrame({"text": ["sample"]})
    with pytest.raises(KeyError):
        preprocess_reviews(df, text_column="review_text")


def test_preprocess_reviews_filters_short_text():
    df = pd.DataFrame({"review_text": ["short", "this is long enough"]})
    cleaned = preprocess_reviews(df, text_column="review_text", min_text_length=6)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["review_text"] == "this is long enough"


def test_chunk_reviews_validates_parameters():
    df = pd.DataFrame(
        {
            "review_text": ["great product"],
            "review_id": ["1"],
        }
    )

    with pytest.raises(ValueError):
        chunk_reviews(df, chunk_size=0)

    with pytest.raises(ValueError):
        chunk_reviews(df, chunk_overlap=-1)

    with pytest.raises(ValueError):
        chunk_reviews(df, chunk_overlap=5, chunk_size=5)


def _build_sample_reviews(tmp_path):
    data = pd.DataFrame(
        {
            "review_id": ["a", "b"],
            "review_text": [
                "I love the battery life of this vacuum cleaner, it lasts forever",
                "The noise level is too high and the battery dies quickly",
            ],
            "product_id": ["vacuum-1", "vacuum-2"],
            "rating": [5, 2],
        }
    )
    csv_path = tmp_path / "reviews.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


def test_build_documents_from_csv_roundtrip(tmp_path):
    csv_path = _build_sample_reviews(tmp_path)
    documents = build_documents_from_csv(
        csv_path,
        text_column="review_text",
        review_id_column="review_id",
        product_column="product_id",
        rating_column="rating",
        min_text_length=10,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert documents, "expected at least one document chunk"
    first_doc = documents[0]
    assert isinstance(first_doc, ReviewDocument)
    assert first_doc.metadata["review_id"] == "a"
    assert "product_id" in first_doc.metadata
    assert "rating" in first_doc.metadata


def test_summarize_documents(tmp_path):
    csv_path = _build_sample_reviews(tmp_path)
    documents = build_documents_from_csv(csv_path)
    summary = summarize_documents(documents)
    assert {"doc_id", "text", "review_id"}.issubset(summary.columns)
    assert len(summary) == len(documents)
