"""Vector store implementation backed by scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_customer_review_chatbot.src.ingestion.loader import ReviewDocument


@dataclass
class RetrievalResult:
    """Container bundling a retrieved document and its similarity score."""

    document: ReviewDocument
    score: float


class SklearnVectorStore:
    """A minimal in-memory vector store built on top of TF-IDF."""

    def __init__(self, *, vectorizer: TfidfVectorizer | None = None) -> None:
        self._vectorizer = vectorizer or TfidfVectorizer()
        self._doc_matrix: sparse.spmatrix | None = None
        self._documents: List[ReviewDocument] = []

    @property
    def is_fitted(self) -> bool:
        return self._doc_matrix is not None and self._doc_matrix.shape[0] > 0

    def fit(self, documents: Sequence[ReviewDocument]) -> None:
        if not documents:
            raise ValueError("Cannot fit vector store on an empty document collection.")

        texts = [doc.text for doc in documents]
        self._doc_matrix = self._vectorizer.fit_transform(texts)
        self._documents = list(documents)

    def add_documents(self, documents: Sequence[ReviewDocument]) -> None:
        if not documents:
            return

        if not self.is_fitted:
            self.fit(documents)
            return

        texts = [doc.text for doc in documents]
        new_matrix = self._vectorizer.transform(texts)
        self._doc_matrix = sparse.vstack([self._doc_matrix, new_matrix])
        self._documents.extend(documents)

    def query(self, question: str, *, top_k: int = 3) -> List[RetrievalResult]:
        if not self.is_fitted:
            raise RuntimeError("Vector store has not been fitted with any documents yet.")

        if top_k <= 0:
            raise ValueError("`top_k` must be positive.")

        query_vec = self._vectorizer.transform([question])
        similarities = cosine_similarity(query_vec, self._doc_matrix)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results: List[RetrievalResult] = []
        for idx in top_indices:
            results.append(
                RetrievalResult(document=self._documents[int(idx)], score=float(similarities[int(idx)]))
            )
        return results


def create_vector_store(
    documents: Iterable[ReviewDocument],
    *,
    max_features: int | None = None,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
) -> SklearnVectorStore:
    """Build and return a fitted `SklearnVectorStore`."""

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    store = SklearnVectorStore(vectorizer=vectorizer)
    store.fit(list(documents))
    return store
