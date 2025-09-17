"""Simple RAG chatbot for answering questions about customer reviews."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from rag_customer_review_chatbot.src.ingestion.loader import (
    build_documents_from_csv,
)
from rag_customer_review_chatbot.src.retriever.vectorstore import (
    RetrievalResult,
    SklearnVectorStore,
    create_vector_store,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, dict]:
    """Load a YAML configuration file, falling back to the default config."""

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must define a YAML mapping.")

    return config


class ReviewChatbot:
    """High-level orchestrator that wraps retrieval and answer composition."""

    def __init__(
        self,
        vector_store: SklearnVectorStore,
        *,
        default_top_k: int = 3,
        max_context_reviews: int = 3,
        include_rating_summary: bool = True,
        include_sentiment_hint: bool = True,
    ) -> None:
        self._vector_store = vector_store
        self._default_top_k = max(1, default_top_k)
        self._max_context_reviews = max(1, max_context_reviews)
        self._include_rating_summary = include_rating_summary
        self._include_sentiment_hint = include_sentiment_hint

    def retrieve(self, question: str, *, top_k: Optional[int] = None) -> List[RetrievalResult]:
        k = top_k or self._default_top_k
        return self._vector_store.query(question, top_k=k)

    def answer(self, question: str, *, top_k: Optional[int] = None) -> str:
        results = self.retrieve(question, top_k=top_k)
        if not results:
            return "I could not find any reviews relevant to your question."

        return self._compose_answer(question, results)

    # ------------------------------------------------------------------
    def _compose_answer(self, question: str, results: List[RetrievalResult]) -> str:
        context_reviews = results[: self._max_context_reviews]
        summary_lines: List[str] = []

        if self._include_rating_summary:
            ratings = [
                res.document.metadata.get("rating")
                for res in context_reviews
                if isinstance(res.document.metadata.get("rating"), (int, float))
            ]
            if ratings:
                avg_rating = statistics.fmean(ratings)
                summary_lines.append(
                    f"Average rating across {len(ratings)} relevant review(s): {avg_rating:.2f}/5."
                )

        if self._include_sentiment_hint:
            sentiment_line = _sentiment_hint(context_reviews)
            if sentiment_line:
                summary_lines.append(sentiment_line)

        supporting_lines = [
            _format_review_excerpt(index + 1, res) for index, res in enumerate(context_reviews)
        ]

        answer_parts = [f"Question: {question}"]
        if summary_lines:
            answer_parts.append("Insights:")
            answer_parts.extend(f"- {line}" for line in summary_lines)
        answer_parts.append("Supporting reviews:")
        answer_parts.extend(supporting_lines)

        return "\n".join(answer_parts)


def _format_review_excerpt(position: int, result: RetrievalResult) -> str:
    doc = result.document
    metadata = doc.metadata
    rating = metadata.get("rating")
    rating_fragment = f" | Rating: {rating:.1f}" if isinstance(rating, (int, float)) else ""
    product_fragment = f" | Product: {metadata.get('product_id')}" if metadata.get("product_id") else ""
    snippet = doc.text.strip()
    if len(snippet) > 300:
        snippet = snippet[:297].rstrip() + "..."
    return f"{position}. Score: {result.score:.2f}{rating_fragment}{product_fragment}\n   {snippet}"


def _sentiment_hint(results: Iterable[RetrievalResult]) -> Optional[str]:
    ratings = [
        res.document.metadata.get("rating")
        for res in results
        if isinstance(res.document.metadata.get("rating"), (int, float))
    ]
    if not ratings:
        return None

    positive = sum(1 for rating in ratings if rating >= 4)
    negative = sum(1 for rating in ratings if rating <= 2)
    neutral = len(ratings) - positive - negative

    parts = []
    if positive:
        parts.append(f"{positive} positive")
    if neutral:
        parts.append(f"{neutral} neutral")
    if negative:
        parts.append(f"{negative} negative")

    return "Sentiment mix: " + ", ".join(parts)


# ----------------------------------------------------------------------
# Factory helpers
# ----------------------------------------------------------------------

def build_chatbot_from_config(config: Dict[str, dict]) -> ReviewChatbot:
    paths_cfg = config.get("paths", {})
    data_path = paths_cfg.get("data_path")
    if not data_path:
        raise ValueError("`paths.data_path` must be provided in the configuration.")

    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    ingestion_cfg = config.get("ingestion", {})
    documents = build_documents_from_csv(
        data_path,
        text_column=ingestion_cfg.get("text_column", "review_text"),
        review_id_column=ingestion_cfg.get("review_id_column", "review_id"),
        product_column=ingestion_cfg.get("product_column", "product_id"),
        rating_column=ingestion_cfg.get("rating_column", "rating"),
        min_text_length=int(ingestion_cfg.get("min_text_length", 40)),
        chunk_size=int(ingestion_cfg.get("chunk_size", 220)),
        chunk_overlap=int(ingestion_cfg.get("chunk_overlap", 40)),
    )

    retriever_cfg = config.get("retriever", {})
    vectorizer_cfg = retriever_cfg.get("vectorizer", {})
    vector_store = create_vector_store(
        documents,
        max_features=vectorizer_cfg.get("max_features"),
        ngram_range=tuple(vectorizer_cfg.get("ngram_range", (1, 2))),
        min_df=int(vectorizer_cfg.get("min_df", 1)),
    )

    chatbot_cfg = config.get("chatbot", {})
    chatbot = ReviewChatbot(
        vector_store,
        default_top_k=int(retriever_cfg.get("top_k", 3)),
        max_context_reviews=int(chatbot_cfg.get("max_context_reviews", 3)),
        include_rating_summary=bool(chatbot_cfg.get("include_rating_summary", True)),
        include_sentiment_hint=bool(chatbot_cfg.get("include_sentiment_hint", True)),
    )

    return chatbot


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with your customer reviews using a RAG pipeline.")
    parser.add_argument("--config-path", type=str, help="Path to a YAML config file.")
    parser.add_argument("--data-path", type=str, help="Override the dataset path defined in the config.")
    parser.add_argument("--top-k", type=int, help="Number of review chunks to retrieve for each question.")
    return parser.parse_args(argv)


def configure_from_args(config: Dict[str, dict], args: argparse.Namespace) -> Dict[str, dict]:
    config = dict(config)  # shallow copy
    config.setdefault("paths", {})
    config.setdefault("retriever", {})

    if args.data_path:
        config["paths"]["data_path"] = args.data_path
    if args.top_k is not None:
        config["retriever"]["top_k"] = args.top_k

    return config


def interactive_loop(chatbot: ReviewChatbot, *, top_k: Optional[int] = None) -> None:
    print("Type a question about your reviews. Enter 'exit' or press Ctrl+D to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except EOFError:  # Ctrl+D
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        answer = chatbot.answer(question, top_k=top_k)
        print(f"\n{answer}\n")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config_path)
    config = configure_from_args(config, args)
    chatbot = build_chatbot_from_config(config)

    retriever_cfg = config.get("retriever", {})
    top_k = int(retriever_cfg.get("top_k", 3))
    interactive_loop(chatbot, top_k=top_k)


if __name__ == "__main__":  # pragma: no cover
    main()
