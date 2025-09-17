from argparse import Namespace
from typing import List

import pandas as pd
import pytest

from rag_customer_review_chatbot.src.app.chatbot import (
    ReviewChatbot,
    build_chatbot_from_config,
    configure_from_args,
)
from rag_customer_review_chatbot.src.ingestion.loader import ReviewDocument
from rag_customer_review_chatbot.src.retriever.vectorstore import create_vector_store


def _sample_documents() -> List[ReviewDocument]:
    return [
        ReviewDocument(
            doc_id="1_0",
            text="The vacuum has excellent battery life and is quiet",
            metadata={"review_id": "1", "rating": 5},
        ),
        ReviewDocument(
            doc_id="2_0",
            text="Battery life is poor and the device is noisy",
            metadata={"review_id": "2", "rating": 2},
        ),
    ]


def test_review_chatbot_requires_non_empty_question():
    store = create_vector_store(_sample_documents())
    chatbot = ReviewChatbot(store)

    with pytest.raises(ValueError):
        chatbot.answer("   ")


def test_review_chatbot_returns_message_when_no_results():
    class EmptyStore:
        def query(self, question: str, top_k: int = 3):
            return []

    chatbot = ReviewChatbot(EmptyStore())
    response = chatbot.answer("What do people say?")
    assert "could not find" in response


def test_configure_from_args_does_not_mutate_original_config():
    original = {
        "paths": {"data_path": "data.csv"},
        "retriever": {"top_k": 2},
    }
    args = Namespace(data_path="override.csv", top_k=5)

    updated = configure_from_args(original, args)

    assert updated["paths"]["data_path"] == "override.csv"
    assert updated["retriever"]["top_k"] == 5
    assert original["paths"]["data_path"] == "data.csv"
    assert original["retriever"]["top_k"] == 2


def test_build_chatbot_from_config_with_real_data(tmp_path):
    data = pd.DataFrame(
        {
            "review_id": ["1", "2"],
            "review_text": [
                "Battery life lasts a long time and the suction is powerful",
                "Battery drains fast and it is very loud",
            ],
            "product_id": ["vacuum-1", "vacuum-2"],
            "rating": [5, 2],
        }
    )
    csv_path = tmp_path / "reviews.csv"
    data.to_csv(csv_path, index=False)

    config = {
        "paths": {"data_path": str(csv_path)},
        "ingestion": {
            "text_column": "review_text",
            "review_id_column": "review_id",
            "product_column": "product_id",
            "rating_column": "rating",
            "chunk_size": 8,
            "chunk_overlap": 2,
            "min_text_length": 5,
        },
        "retriever": {
            "top_k": 2,
            "vectorizer": {"min_df": 1, "ngram_range": [1, 2]},
        },
        "chatbot": {"max_context_reviews": 2, "include_sentiment_hint": True},
    }

    chatbot = build_chatbot_from_config(config)
    answer = chatbot.answer("What about the battery life?")
    assert "Question:" in answer
    assert "Supporting reviews:" in answer
