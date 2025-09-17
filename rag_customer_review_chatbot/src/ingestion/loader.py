"""Utilities for loading and preparing customer review datasets for RAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class ReviewDocument:
    """Lightweight container for review chunks and their metadata."""

    doc_id: str
    text: str
    metadata: Dict[str, object]


def load_reviews(csv_path: Path | str, *, encoding: str = "utf-8") -> pd.DataFrame:
    """Load a CSV file containing customer reviews.

    Parameters
    ----------
    csv_path:
        Location of the CSV file.
    encoding:
        Encoding used when reading the file. Defaults to UTF-8.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the loaded reviews.

    Raises
    ------
    FileNotFoundError
        If the CSV path does not exist.
    ValueError
        If the file cannot be parsed as CSV.
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Review dataset not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding=encoding)
    except Exception as exc:  # pragma: no cover - provides a friendlier error
        raise ValueError(f"Failed to parse reviews CSV: {csv_path}") from exc

    if df.empty:
        raise ValueError("The review dataset is empty.")

    return df


def preprocess_reviews(
    df: pd.DataFrame,
    *,
    text_column: str = "review_text",
    min_text_length: int = 40,
) -> pd.DataFrame:
    """Clean review text and drop records that are too short."""

    if text_column not in df.columns:
        raise KeyError(f"`{text_column}` column is required in the dataset.")

    cleaned = df.copy()
    cleaned[text_column] = (
        cleaned[text_column]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    cleaned = cleaned[cleaned[text_column].str.len() >= max(1, min_text_length)].reset_index(drop=True)

    if cleaned.empty:
        raise ValueError(
            "No reviews remain after preprocessing. Lower `min_text_length` or inspect the dataset."
        )

    return cleaned


def chunk_reviews(
    df: pd.DataFrame,
    *,
    text_column: str = "review_text",
    review_id_column: str = "review_id",
    product_column: Optional[str] = "product_id",
    rating_column: Optional[str] = "rating",
    chunk_size: int = 220,
    chunk_overlap: int = 40,
) -> List[ReviewDocument]:
    """Split reviews into overlapping text chunks for retrieval."""

    if chunk_size <= 0:
        raise ValueError("`chunk_size` must be positive.")
    if chunk_overlap >= chunk_size:
        raise ValueError("`chunk_overlap` must be smaller than `chunk_size`.")

    required_columns = {text_column, review_id_column}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns required for chunking: {sorted(missing)}")

    documents: List[ReviewDocument] = []

    for _, row in df.iterrows():
        text = str(row[text_column])
        tokens = text.split()
        if not tokens:
            continue

        step = chunk_size - chunk_overlap
        chunk_index = 0

        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start : start + chunk_size]
            if not chunk_tokens:
                continue

            chunk_text = " ".join(chunk_tokens)
            doc_id = f"{row[review_id_column]}_{chunk_index}"
            metadata: Dict[str, object] = {"review_id": row[review_id_column]}

            if product_column and product_column in df.columns:
                metadata["product_id"] = row[product_column]
            if rating_column and rating_column in df.columns:
                try:
                    metadata["rating"] = float(row[rating_column])
                except (TypeError, ValueError):
                    metadata["rating"] = None

            metadata["position"] = chunk_index
            metadata["original_text_length"] = len(text)

            documents.append(ReviewDocument(doc_id=doc_id, text=chunk_text, metadata=metadata))
            chunk_index += 1

    if not documents:
        raise ValueError("No chunks were produced from the dataset. Check preprocessing parameters.")

    return documents


def build_documents_from_csv(
    csv_path: Path | str,
    *,
    text_column: str = "review_text",
    review_id_column: str = "review_id",
    product_column: Optional[str] = "product_id",
    rating_column: Optional[str] = "rating",
    min_text_length: int = 40,
    chunk_size: int = 220,
    chunk_overlap: int = 40,
    encoding: str = "utf-8",
) -> List[ReviewDocument]:
    """Convenience helper that loads, preprocesses, and chunks a dataset."""

    df = load_reviews(csv_path, encoding=encoding)
    df = preprocess_reviews(df, text_column=text_column, min_text_length=min_text_length)
    documents = chunk_reviews(
        df,
        text_column=text_column,
        review_id_column=review_id_column,
        product_column=product_column,
        rating_column=rating_column,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return documents


def summarize_documents(documents: Iterable[ReviewDocument]) -> pd.DataFrame:
    """Build a DataFrame summary from the produced documents."""

    rows = []
    for doc in documents:
        row = {"doc_id": doc.doc_id, "text": doc.text}
        row.update(doc.metadata)
        rows.append(row)

    return pd.DataFrame(rows)
