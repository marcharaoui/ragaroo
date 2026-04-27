from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from ..base import BaseQueryTransform, BaseReranker, BaseRetriever
from ..query_augmentation import BaseLLMProvider, QueryTransformSpec
from ..retrieval.types import RetrievedDocument


QueryAugmentation = Callable[[str], str]
QueryAugmentationItem = BaseQueryTransform | QueryTransformSpec | QueryAugmentation
RerankerItem = BaseReranker | BaseRetriever


@dataclass(slots=True)
class Pipeline:
    """A retrieval pipeline with optional query augmentation and reranking."""

    name: str
    retriever: BaseRetriever
    reranker: RerankerItem | None = None
    query_augmentation: QueryAugmentationItem | list[QueryAugmentationItem] | None = None
    llm_provider: BaseLLMProvider | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    last_query_stats: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    last_build_stats: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def prepare(self, corpus: dict[str, dict[str, Any]]) -> None:
        """Build all indexes needed by this pipeline for a corpus."""
        self.retriever.build_index(corpus)
        retriever_build_stats = dict(getattr(self.retriever, "last_build_stats", {}))
        self.last_build_stats = retriever_build_stats
        if isinstance(self.reranker, BaseRetriever) and self.reranker is not self.retriever:
            self.reranker.build_index(corpus)
            reranker_build_stats = dict(getattr(self.reranker, "last_build_stats", {}))
            self.last_build_stats = {
                **retriever_build_stats,
                **{f"reranker_{key}": value for key, value in reranker_build_stats.items()},
                "total_build_time_s": float(retriever_build_stats.get("total_build_time_s", 0.0))
                + float(reranker_build_stats.get("total_build_time_s", 0.0)),
            }

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Run query augmentation, retrieval, and optional reranking for one query."""
        pipeline_started_at = perf_counter()
        query_augmentation_time_s = 0.0
        if self.query_augmentation is None:
            prepared_query = query
        else:
            augmentation_started_at = perf_counter()
            prepared_query = self.apply_query_augmentation(query)
            augmentation_finished_at = perf_counter()
            query_augmentation_time_s = augmentation_finished_at - augmentation_started_at

        retrieval_started_at = perf_counter()
        retrieved_documents = self.retriever.retrieve(prepared_query, top_k=top_k)
        retrieval_finished_at = perf_counter()
        retrieval_time_s = retrieval_finished_at - retrieval_started_at
        if self.reranker is None:
            finished_at = perf_counter()
            self.last_query_stats = {
                "query_augmentation_time_s": query_augmentation_time_s,
                "retrieval_time_s": retrieval_time_s,
                "rerank_time_s": 0.0,
                "total_pipeline_time_s": finished_at - pipeline_started_at,
            }
            return retrieved_documents

        rerank_started_at = perf_counter()
        reranked_documents = self.reranker.rerank(prepared_query, retrieved_documents)
        finished_at = perf_counter()
        rerank_time_s = finished_at - rerank_started_at
        self.last_query_stats = {
            "query_augmentation_time_s": query_augmentation_time_s,
            "retrieval_time_s": retrieval_time_s,
            "rerank_time_s": max(rerank_time_s, 0.0),
            "total_pipeline_time_s": finished_at - pipeline_started_at,
        }
        return reranked_documents

    def apply_query_augmentation(self, query: str) -> str:
        """Apply configured query transforms without retrieving documents."""
        if self.query_augmentation is None:
            return query
        if isinstance(self.query_augmentation, list):
            transformed = query
            for transform in self.query_augmentation:
                transformed = self._apply_single_query_augmentation(transform, transformed)
            return transformed
        return self._apply_single_query_augmentation(self.query_augmentation, query)

    @property
    def config_hash(self) -> str:
        serialized = json.dumps(self.config_dict(), sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    def config_dict(self) -> dict[str, Any]:
        """Return the serializable configuration used for experiment manifests."""
        return {
            "name": self.name,
            "retriever": self.retriever.config_dict(),
            "reranker": None if self.reranker is None else self.reranker.config_dict(),
            "query_augmentation": self._query_augmentation_config(),
            "llm_provider": None if self.llm_provider is None else self.llm_provider.config_dict(),
            "metadata": self.metadata,
        }

    def _apply_single_query_augmentation(
        self,
        transform: QueryAugmentationItem,
        query: str,
    ) -> str:
        if isinstance(transform, QueryTransformSpec):
            transform = self._build_query_transform(transform)
        result = transform.transform(query) if isinstance(transform, BaseQueryTransform) else transform(query)
        if not isinstance(result, str):
            raise TypeError("Pipeline query augmentation must return a string for a single query")
        return result

    def _query_augmentation_config(self) -> Any:
        if self.query_augmentation is None:
            return None
        if isinstance(self.query_augmentation, list):
            return [self._single_query_augmentation_config(transform) for transform in self.query_augmentation]
        return self._single_query_augmentation_config(self.query_augmentation)

    @staticmethod
    def _single_query_augmentation_config(
        transform: QueryAugmentationItem,
    ) -> Any:
        if isinstance(transform, QueryTransformSpec):
            return transform.config_dict()
        if isinstance(transform, BaseQueryTransform):
            return transform.config_dict()
        return getattr(transform, "__name__", transform.__class__.__name__)

    def _build_query_transform(self, spec: QueryTransformSpec) -> BaseQueryTransform:
        kwargs = dict(spec.kwargs)
        if "provider" not in kwargs:
            if self.llm_provider is None:
                raise ValueError(
                    f"Pipeline '{self.name}' uses query augmentation '{spec.transform_class.__name__}' "
                    "but no llm_provider was provided."
                )
            kwargs["provider"] = self.llm_provider
        return spec.transform_class(**kwargs)
