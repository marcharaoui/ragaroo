from __future__ import annotations

from dataclasses import dataclass, field
import sys
from time import perf_counter
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ..dataset import Dataset
from ..pipeline.pipeline import Pipeline
from ..retrieval.types import RetrievedDocument
from .metrics import metric_value, parse_metric_name


DEFAULT_RANKING_METRICS = [
    "recall",
    "precision",
    "mrr",
    "map",
    "hit_rate",
    "ndcg",
]
LATENCY_METRICS = {
    "latency_ms",
    "query_augmentation_latency_ms",
    "retrieval_latency_ms",
    "rerank_latency_ms",
    "avg_query_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "total_time_s",
}


@dataclass(frozen=True, slots=True)
class RankingMetricSpec:
    name: str
    base_name: str
    k: int


@dataclass(slots=True)
class QueryResult:
    query_id: str
    query: str
    latency_ms: float
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    results: list[RetrievedDocument] = field(default_factory=list)
    metric_values: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    pipeline_name: str
    pipeline_hash: str
    metrics: dict[str, float]
    query_results: list[QueryResult]
    build_stats: dict[str, float]
    query_count: int


class Evaluator:
    """Evaluate retrieval pipelines with ranking and latency metrics."""

    def __init__(
        self,
        metrics: list[str] | None = None,
        *,
        store_query_results: bool = False,
    ) -> None:
        self.metrics = metrics
        self.store_query_results = store_query_results

    def evaluate(
        self,
        dataset: Dataset,
        pipeline: Pipeline,
        *,
        show_progress: bool = False,
        query_items: list[tuple[str, str]] | None = None,
        warmup_queries: int = 0,
        prepare_pipeline: bool = True,
    ) -> EvaluationResult:
        """Evaluate one pipeline against a loaded dataset."""
        if not dataset.loaded:
            dataset.load()

        if prepare_pipeline:
            pipeline.prepare(dataset.corpus)

        metric_names = self.metrics or self._default_metrics_for_pipeline(pipeline)
        retriever_top_k = getattr(pipeline.retriever, "top_k", 10)
        ranking_metrics = self._compile_ranking_metrics(metric_names, retriever_top_k)
        query_results: list[QueryResult] = []
        metric_sums = {metric.name: 0.0 for metric in ranking_metrics}
        latencies_ms: list[float] = []
        query_augmentation_latencies_ms: list[float] = []
        retrieval_latencies_ms: list[float] = []
        rerank_latencies_ms: list[float] = []
        selected_query_items = list(dataset.queries.items()) if query_items is None else query_items
        warmup_count = max(0, min(warmup_queries, len(selected_query_items)))

        total_started_at = perf_counter()
        query_iterator = tqdm(
            selected_query_items,
            total=len(selected_query_items),
            desc=f"Queries [{pipeline.name}]",
            leave=False,
            disable=not show_progress,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for query_index, (query_id, query_text) in enumerate(query_iterator):
            started_at = perf_counter()
            results = pipeline.retrieve(query_text, top_k=retriever_top_k)
            latency_ms = (perf_counter() - started_at) * 1000.0
            if query_index < warmup_count:
                continue

            latencies_ms.append(latency_ms)
            stage_latencies_ms = {
                "query_augmentation_latency_ms": float(
                    getattr(pipeline, "last_query_stats", {}).get("query_augmentation_time_s", 0.0)
                )
                * 1000.0,
                "retrieval_latency_ms": float(
                    getattr(pipeline, "last_query_stats", {}).get("retrieval_time_s", 0.0)
                )
                * 1000.0,
                "rerank_latency_ms": float(
                    getattr(pipeline, "last_query_stats", {}).get("rerank_time_s", 0.0)
                )
                * 1000.0,
            }
            query_augmentation_latencies_ms.append(stage_latencies_ms["query_augmentation_latency_ms"])
            retrieval_latencies_ms.append(stage_latencies_ms["retrieval_latency_ms"])
            rerank_latencies_ms.append(stage_latencies_ms["rerank_latency_ms"])

            qrels = dataset.qrels.get(query_id, {})
            relevant_ids = {
                corpus_id
                for corpus_id, score in qrels.items()
                if score > 0
            }
            metric_values: dict[str, float] = {}
            for metric in ranking_metrics:
                value = metric_value(
                    metric_name=metric.name,
                    results=results,
                    qrels=qrels,
                    default_k=metric.k,
                    base_name=metric.base_name,
                    relevant_ids=relevant_ids,
                )
                metric_sums[metric.name] += value
                if self.store_query_results:
                    metric_values[metric.name] = value

            if self.store_query_results:
                query_results.append(
                    QueryResult(
                        query_id=query_id,
                        query=query_text,
                        latency_ms=latency_ms,
                        stage_latencies_ms=stage_latencies_ms,
                        results=results,
                        metric_values=metric_values,
                    )
                )

        total_time_s = perf_counter() - total_started_at
        aggregated_metrics = self._aggregate_metrics(
            metric_names=metric_names,
            metric_sums=metric_sums,
            latencies_ms=latencies_ms,
            query_augmentation_latencies_ms=query_augmentation_latencies_ms,
            retrieval_latencies_ms=retrieval_latencies_ms,
            rerank_latencies_ms=rerank_latencies_ms,
            total_time_s=total_time_s,
        )

        build_stats = dict(
            getattr(
                pipeline,
                "last_build_stats",
                getattr(pipeline.retriever, "last_build_stats", {}),
            )
        )

        return EvaluationResult(
            pipeline_name=pipeline.name,
            pipeline_hash=pipeline.config_hash,
            metrics=aggregated_metrics,
            query_results=query_results,
            build_stats=build_stats,
            query_count=len(latencies_ms),
        )

    def _aggregate_metrics(
        self,
        metric_names: list[str],
        metric_sums: dict[str, float],
        latencies_ms: list[float],
        query_augmentation_latencies_ms: list[float],
        retrieval_latencies_ms: list[float],
        rerank_latencies_ms: list[float],
        total_time_s: float,
    ) -> dict[str, float]:
        if not latencies_ms:
            return {}

        latencies = np.asarray(latencies_ms, dtype=np.float32)
        query_augmentation_latencies = np.asarray(query_augmentation_latencies_ms, dtype=np.float32)
        retrieval_latencies = np.asarray(retrieval_latencies_ms, dtype=np.float32)
        rerank_latencies = np.asarray(rerank_latencies_ms, dtype=np.float32)
        query_count = float(len(latencies_ms))
        aggregated: dict[str, float] = {}

        for metric_name in metric_names:
            if metric_name == "latency_ms" or metric_name == "avg_query_latency_ms":
                aggregated[metric_name] = float(latencies.mean())
                continue
            if metric_name == "query_augmentation_latency_ms":
                aggregated[metric_name] = float(query_augmentation_latencies.mean())
                continue
            if metric_name == "retrieval_latency_ms":
                aggregated[metric_name] = float(retrieval_latencies.mean())
                continue
            if metric_name == "rerank_latency_ms":
                aggregated[metric_name] = float(rerank_latencies.mean())
                continue
            if metric_name == "p50_latency_ms":
                aggregated[metric_name] = float(np.percentile(latencies, 50))
                continue
            if metric_name == "p95_latency_ms":
                aggregated[metric_name] = float(np.percentile(latencies, 95))
                continue
            if metric_name == "total_time_s":
                aggregated[metric_name] = total_time_s
                continue

            aggregated[metric_name] = metric_sums.get(metric_name, 0.0) / query_count

        return aggregated

    @staticmethod
    def _default_metrics_for_pipeline(pipeline: Pipeline) -> list[str]:
        top_k = getattr(pipeline.reranker, "top_k", None) or getattr(pipeline.retriever, "top_k", 10)
        metrics = [f"{metric}@{top_k}" for metric in DEFAULT_RANKING_METRICS]
        metrics.extend(
            [
                "latency_ms",
                "query_augmentation_latency_ms",
                "retrieval_latency_ms",
                "rerank_latency_ms",
                "p50_latency_ms",
                "p95_latency_ms",
                "total_time_s",
            ]
        )
        return metrics

    @staticmethod
    def _compile_ranking_metrics(metric_names: list[str], default_k: int) -> list[RankingMetricSpec]:
        compiled: list[RankingMetricSpec] = []
        for metric_name in metric_names:
            if metric_name in LATENCY_METRICS:
                continue
            base_name, metric_k = parse_metric_name(metric_name, default_k)
            if metric_k is None:
                continue
            compiled.append(
                RankingMetricSpec(
                    name=metric_name,
                    base_name=base_name,
                    k=metric_k,
                )
            )
        return compiled


def evaluate(
    dataset: Dataset,
    pipeline: Pipeline,
    metrics: list[str] | None = None,
    *,
    show_progress: bool = False,
    store_query_results: bool = False,
    query_items: list[tuple[str, str]] | None = None,
    warmup_queries: int = 0,
    prepare_pipeline: bool = True,
) -> EvaluationResult:
    evaluator = Evaluator(metrics=metrics, store_query_results=store_query_results)
    return evaluator.evaluate(
        dataset,
        pipeline,
        show_progress=show_progress,
        query_items=query_items,
        warmup_queries=warmup_queries,
        prepare_pipeline=prepare_pipeline,
    )


Evaluation = Evaluator
