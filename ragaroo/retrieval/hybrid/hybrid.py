from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Any

from ...base import BaseRetriever
from ..types import RetrievedDocument


class HybridRetriever(BaseRetriever):
    """Fuse two retriever result lists with reciprocal-rank or normalized-score fusion."""

    def __init__(
        self,
        retriever_1: BaseRetriever,
        retriever_2: BaseRetriever,
        *,
        top_k: int = 10,
        fusion_technique: str = "rrf",
        rrf_k: int = 60,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if fusion_technique not in {"rrf", "average"}:
            raise ValueError("fusion_technique must be one of ['rrf', 'average']")

        self.retriever_1 = retriever_1
        self.retriever_2 = retriever_2
        self.top_k = top_k
        self.fusion_technique = fusion_technique
        self.rrf_k = rrf_k
        self.last_build_stats: dict[str, float] = {}
        self.last_query_stats: dict[str, float] = {}
        self._is_built = False

    def build_index(self, corpus: dict[str, dict[str, Any]]) -> None:
        self.retriever_1.build_index(corpus)
        self.retriever_2.build_index(corpus)
        self.last_build_stats = {
            "retriever_1_total_build_time_s": float(self.retriever_1.last_build_stats.get("total_build_time_s", 0.0)),
            "retriever_2_total_build_time_s": float(self.retriever_2.last_build_stats.get("total_build_time_s", 0.0)),
            "total_build_time_s": float(self.retriever_1.last_build_stats.get("total_build_time_s", 0.0))
            + float(self.retriever_2.last_build_stats.get("total_build_time_s", 0.0)),
        }
        self._is_built = True

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDocument]:
        started_at = perf_counter()
        k = top_k or self.top_k
        if k <= 0:
            raise ValueError("top_k must be > 0")

        results_1 = self.retriever_1.retrieve(query, top_k=k)
        results_2 = self.retriever_2.retrieve(query, top_k=k)

        fusion_started_at = perf_counter()
        if self.fusion_technique == "average":
            fused = self._average_fusion(results_1, results_2)
        else:
            fused = self._rrf_fusion(results_1, results_2)
        finished_at = perf_counter()

        self.last_query_stats = {
            "retriever_1_total_query_time_s": float(self.retriever_1.last_query_stats.get("total_query_time_s", 0.0)),
            "retriever_2_total_query_time_s": float(self.retriever_2.last_query_stats.get("total_query_time_s", 0.0)),
            "fusion_time_s": finished_at - fusion_started_at,
            "total_query_time_s": finished_at - started_at,
            "top_k": float(k),
        }

        return fused[:k]

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "top_k": self.top_k,
            "fusion_technique": self.fusion_technique,
            "rrf_k": self.rrf_k,
            "retriever_1": self.retriever_1.config_dict(),
            "retriever_2": self.retriever_2.config_dict(),
        }

    def _rrf_fusion(
        self,
        results_1: list[RetrievedDocument],
        results_2: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        fused_scores: dict[str, float] = defaultdict(float)
        documents: dict[str, RetrievedDocument] = {}

        for results in (results_1, results_2):
            for rank, document in enumerate(results, start=1):
                fused_scores[document.corpus_id] += 1.0 / (self.rrf_k + rank)
                documents.setdefault(document.corpus_id, document)

        return [
            RetrievedDocument(
                corpus_id=documents[corpus_id].corpus_id,
                score=score,
                text=documents[corpus_id].text,
                metadata=documents[corpus_id].metadata,
            )
            for corpus_id, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        ]

    def _average_fusion(
        self,
        results_1: list[RetrievedDocument],
        results_2: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        normalized_1 = self._normalize_scores(results_1)
        normalized_2 = self._normalize_scores(results_2)

        fused_scores: dict[str, float] = defaultdict(float)
        documents: dict[str, RetrievedDocument] = {}

        for normalized_scores, results in ((normalized_1, results_1), (normalized_2, results_2)):
            for document in results:
                fused_scores[document.corpus_id] += normalized_scores[document.corpus_id]
                documents.setdefault(document.corpus_id, document)

        averaged_scores = {
            corpus_id: fused_scores[corpus_id] / 2.0
            for corpus_id in fused_scores
        }
        return [
            RetrievedDocument(
                corpus_id=documents[corpus_id].corpus_id,
                score=score,
                text=documents[corpus_id].text,
                metadata=documents[corpus_id].metadata,
            )
            for corpus_id, score in sorted(averaged_scores.items(), key=lambda item: item[1], reverse=True)
        ]

    @staticmethod
    def _normalize_scores(results: list[RetrievedDocument]) -> dict[str, float]:
        if not results:
            return {}
        scores = [document.score for document in results]
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return {document.corpus_id: 1.0 for document in results}
        return {
            document.corpus_id: (document.score - min_score) / (max_score - min_score)
            for document in results
        }

    @property
    def corpus_size(self) -> int:
        return getattr(self.retriever_1, "corpus_size", 0)

    @property
    def is_built(self) -> bool:
        return self._is_built
