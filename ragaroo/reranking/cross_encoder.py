from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from ..base import BaseReranker
from ..retrieval.types import RetrievedDocument
from .st_reranker import SentenceTransformerCrossEncoder


class CrossEncoderReranker(BaseReranker):
    """Rerank candidate documents with a Sentence Transformers cross-encoder."""

    def __init__(
        self,
        model: SentenceTransformerCrossEncoder | None = None,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        device: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        self.model = model or SentenceTransformerCrossEncoder(
            model_name,
            device=device,
            hf_token=hf_token,
        )
        self.top_k = top_k
        self.last_rerank_stats: dict[str, float] = {}

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        if not documents:
            self.last_rerank_stats = {
                "rerank_time_s": 0.0,
                "input_candidates": 0.0,
                "output_candidates": 0.0,
            }
            return []

        started_at = perf_counter()
        scores = self.model.score(query, [document.text for document in documents])
        ranked_indices = np.argsort(scores)[::-1]
        reranked_documents = [
            RetrievedDocument(
                corpus_id=documents[index].corpus_id,
                score=float(scores[index]),
                text=documents[index].text,
                metadata=documents[index].metadata,
            )
            for index in ranked_indices[: self.top_k]
        ]
        finished_at = perf_counter()

        self.last_rerank_stats = {
            "rerank_time_s": finished_at - started_at,
            "input_candidates": float(len(documents)),
            "output_candidates": float(len(reranked_documents)),
        }

        return reranked_documents

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "model": self.model.__class__.__name__,
            "model_name_or_path": getattr(self.model, "model_name_or_path", None),
            "top_k": self.top_k,
        }
