"""Application layer utilities for the customer review RAG chatbot."""

from __future__ import annotations

import argparse
import ast
import copy
import logging
import statistics
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence

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

logger = logging.getLogger(__name__)

ConfigDict = Dict[str, Any]


def load_config(config_path: Optional[str | Path] = None) -> ConfigDict:
    """Load a YAML configuration file, falling back to the default config."""

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_text = handle.read()

    config = _load_yaml_text(raw_text)

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
        self._default_top_k = _ensure_positive_int(default_top_k, "default_top_k")
        self._max_context_reviews = _ensure_positive_int(
            max_context_reviews, "max_context_reviews"
        )
        self._include_rating_summary = include_rating_summary
        self._include_sentiment_hint = include_sentiment_hint

    def retrieve(self, question: str, *, top_k: Optional[int] = None) -> List[RetrievalResult]:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        k = _ensure_positive_int(top_k, "top_k") if top_k is not None else self._default_top_k
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
    _ensure_mapping(config, "config root")

    paths_cfg = _ensure_mapping(config.get("paths", {}), "paths")
    data_path = paths_cfg.get("data_path")
    if not data_path:
        raise ValueError("`paths.data_path` must be provided in the configuration.")

    data_path = Path(str(data_path)).expanduser()
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    data_path = data_path.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Review dataset not found at {data_path}")

    ingestion_cfg = _ensure_mapping(config.get("ingestion", {}), "ingestion")
    documents = build_documents_from_csv(
        data_path,
        text_column=_ensure_non_empty_str(
            ingestion_cfg.get("text_column", "review_text"), "ingestion.text_column"
        ),
        review_id_column=_ensure_non_empty_str(
            ingestion_cfg.get("review_id_column", "review_id"),
            "ingestion.review_id_column",
        ),
        product_column=_coerce_optional_str(
            ingestion_cfg.get("product_column", "product_id")
        ),
        rating_column=_coerce_optional_str(ingestion_cfg.get("rating_column", "rating")),
        min_text_length=_ensure_positive_int(
            ingestion_cfg.get("min_text_length", 40), "ingestion.min_text_length"
        ),
        chunk_size=_ensure_positive_int(
            ingestion_cfg.get("chunk_size", 220), "ingestion.chunk_size"
        ),
        chunk_overlap=_ensure_non_negative_int(
            ingestion_cfg.get("chunk_overlap", 40), "ingestion.chunk_overlap"
        ),
    )

    retriever_cfg = _ensure_mapping(config.get("retriever", {}), "retriever")
    vectorizer_cfg = _ensure_mapping(
        retriever_cfg.get("vectorizer", {}), "retriever.vectorizer"
    )
    ngram_range = vectorizer_cfg.get("ngram_range", (1, 2))
    ngram_range_tuple = _ensure_ngram_range(ngram_range)
    max_features = vectorizer_cfg.get("max_features")
    if max_features is not None:
        max_features = _ensure_positive_int(max_features, "retriever.vectorizer.max_features")

    vector_store = create_vector_store(
        documents,
        max_features=max_features,
        ngram_range=ngram_range_tuple,
        min_df=_ensure_positive_int(vectorizer_cfg.get("min_df", 1), "retriever.vectorizer.min_df"),
    )

    chatbot_cfg = _ensure_mapping(config.get("chatbot", {}), "chatbot")
    chatbot = ReviewChatbot(
        vector_store,
        default_top_k=_ensure_positive_int(
            retriever_cfg.get("top_k", 3), "retriever.top_k"
        ),
        max_context_reviews=_ensure_positive_int(
            chatbot_cfg.get("max_context_reviews", 3), "chatbot.max_context_reviews"
        ),
        include_rating_summary=bool(chatbot_cfg.get("include_rating_summary", True)),
        include_sentiment_hint=bool(chatbot_cfg.get("include_sentiment_hint", True)),
    )

    return chatbot


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with your customer reviews using a RAG pipeline."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help=f"Path to a YAML config file. Defaults to {DEFAULT_CONFIG_PATH} if omitted.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Override the dataset path defined in the config.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Number of review chunks to retrieve for each question.",
    )
    return parser.parse_args(argv)


def configure_from_args(config: ConfigDict, args: argparse.Namespace) -> ConfigDict:
    copied: ConfigDict = copy.deepcopy(config)
    copied.setdefault("paths", {})
    copied.setdefault("retriever", {})

    if args.data_path:
        copied["paths"]["data_path"] = args.data_path
    if args.top_k is not None:
        copied["retriever"]["top_k"] = args.top_k

    return copied


def interactive_loop(chatbot: ReviewChatbot, *, top_k: Optional[int] = None) -> None:
    print("Type a question about your reviews. Enter 'exit' or press Ctrl+D to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except EOFError:  # Ctrl+D
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:  # pragma: no cover - user initiated interruption
            print("\nInterrupted. Goodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            answer = chatbot.answer(question, top_k=top_k)
        except Exception as exc:  # pragma: no cover - interactive safeguard
            logger.exception("Error while answering question")
            print(f"An error occurred: {exc}")
            continue

        print(f"\n{answer}\n")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config_path)
    config = configure_from_args(config, args)
    chatbot = build_chatbot_from_config(config)

    retriever_cfg = _ensure_mapping(config.get("retriever", {}), "retriever")
    top_k = _ensure_positive_int(retriever_cfg.get("top_k", 3), "retriever.top_k")
    interactive_loop(chatbot, top_k=top_k)


def _ensure_positive_int(value: Any, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc
    if number <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return number


def _ensure_non_negative_int(value: Any, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer.") from exc
    if number < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return number


def _ensure_mapping(value: Any, field_name: str) -> MutableMapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MutableMapping):
        raise ValueError(f"{field_name} section must be a mapping.")
    return dict(value)


def _ensure_ngram_range(value: Any) -> tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("ngram_range must contain exactly two integers.")
        start, end = value
    else:
        raise ValueError("ngram_range must be a sequence of two integers.")

    start_int = _ensure_positive_int(start, "ngram_range start")
    end_int = _ensure_positive_int(end, "ngram_range end")
    if start_int > end_int:
        raise ValueError("ngram_range start must be less than or equal to end.")
    return (start_int, end_int)


def _ensure_non_empty_str(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must be provided.")

    result = str(value).strip()
    if not result:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return result


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _load_yaml_text(text: str) -> Any:
    """Parse YAML using PyYAML if available, otherwise use a minimal parser."""

    yaml_module = _try_import_yaml()
    if yaml_module is not None:
        loaded = yaml_module.safe_load(text) or {}
        return loaded

    return _parse_simple_yaml(text)


def _try_import_yaml():
    """Attempt to import :mod:`yaml`, returning ``None`` if unavailable."""

    try:
        return import_module("yaml")
    except ModuleNotFoundError:
        return None


def _parse_simple_yaml(text: str) -> Any:
    """Parse a limited subset of YAML used by the default configuration.

    The fallback parser supports nested mappings with two-space indentation and
    scalar values such as integers, floats, booleans, strings, and inline lists.
    It intentionally rejects unsupported constructs (e.g. multi-line strings,
    anchors) to fail fast if a more complex configuration is provided.
    """

    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError("Configuration indentation must use multiples of two spaces.")

        stripped = line.strip()
        if stripped.startswith("-"):
            raise ValueError("List syntax with '-' is not supported by the fallback YAML parser.")

        key, sep, value_part = stripped.partition(":")
        if not sep:
            raise ValueError(f"Invalid configuration line: {raw_line}")

        key = key.strip()
        value_text = value_part.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation in configuration near: {raw_line}")

        parent = stack[-1][1]
        if value_text == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value_text)

    return root


def _parse_scalar(value_text: str) -> Any:
    lowered = value_text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None

    try:
        return ast.literal_eval(value_text)
    except (ValueError, SyntaxError):
        pass

    # Attempt to coerce numeric strings that literal_eval could not parse due
    # to missing decimal points or other formatting nuances.
    try:
        if "." in value_text:
            return float(value_text)
        return int(value_text)
    except ValueError:
        return value_text


if __name__ == "__main__":  # pragma: no cover
    main()
