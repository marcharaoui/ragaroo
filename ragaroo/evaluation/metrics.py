from __future__ import annotations

import math
from collections.abc import Sequence

from ..retrieval.types import RetrievedDocument


def parse_metric_name(metric_name: str, default_k: int) -> tuple[str, int | None]:
    normalized = metric_name.strip().lower()
    if "@" in normalized:
        base, raw_k = normalized.split("@", maxsplit=1)
        return base, int(raw_k)
    return normalized, None if normalized.endswith("_s") else default_k


def relevant_ids_for_query(qrels: dict[str, float]) -> set[str]:
    return {corpus_id for corpus_id, score in qrels.items() if score > 0}


def metric_value(
    metric_name: str,
    results: Sequence[RetrievedDocument],
    qrels: dict[str, float],
    default_k: int,
    *,
    base_name: str | None = None,
    relevant_ids: set[str] | None = None,
) -> float:
    parsed_base_name, metric_k = parse_metric_name(metric_name, default_k)
    if base_name is None:
        base_name = parsed_base_name
    if metric_k is None:
        raise ValueError(f"Metric '{metric_name}' is not a ranking metric")

    ranked_results = list(results[:metric_k])
    if relevant_ids is None:
        relevant_ids = relevant_ids_for_query(qrels)

    if base_name == "recall":
        return recall_at_k(ranked_results, relevant_ids)
    if base_name == "precision":
        return precision_at_k(ranked_results, relevant_ids, metric_k)
    if base_name == "mrr":
        return mrr_at_k(ranked_results, relevant_ids)
    if base_name == "map":
        return map_at_k(ranked_results, relevant_ids)
    if base_name in {"hit_rate", "hitrate"}:
        return hit_rate_at_k(ranked_results, relevant_ids)
    if base_name == "ndcg":
        return ndcg_at_k(ranked_results, qrels, metric_k)

    raise ValueError(f"Unsupported metric '{metric_name}'")


def recall_at_k(
    results: Sequence[RetrievedDocument],
    relevant_ids: set[str],
) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for result in results if result.corpus_id in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(
    results: Sequence[RetrievedDocument],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for result in results if result.corpus_id in relevant_ids)
    return hits / k


def mrr_at_k(
    results: Sequence[RetrievedDocument],
    relevant_ids: set[str],
) -> float:
    for rank, result in enumerate(results, start=1):
        if result.corpus_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def map_at_k(
    results: Sequence[RetrievedDocument],
    relevant_ids: set[str],
) -> float:
    if not relevant_ids:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, result in enumerate(results, start=1):
        if result.corpus_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank

    if hits == 0:
        return 0.0
    return precision_sum / len(relevant_ids)


def hit_rate_at_k(
    results: Sequence[RetrievedDocument],
    relevant_ids: set[str],
) -> float:
    return float(any(result.corpus_id in relevant_ids for result in results))


def ndcg_at_k(
    results: Sequence[RetrievedDocument],
    qrels: dict[str, float],
    k: int,
) -> float:
    ideal_scores = sorted((score for score in qrels.values() if score > 0), reverse=True)[:k]
    ideal_dcg = discounted_cumulative_gain(ideal_scores)
    if ideal_dcg == 0:
        return 0.0

    actual_scores = [qrels.get(result.corpus_id, 0.0) for result in results[:k]]
    actual_dcg = discounted_cumulative_gain(actual_scores)
    return actual_dcg / ideal_dcg


def discounted_cumulative_gain(scores: Sequence[float]) -> float:
    total = 0.0
    for rank, score in enumerate(scores, start=1):
        total += (2 ** score - 1) / math.log2(rank + 1)
    return total
