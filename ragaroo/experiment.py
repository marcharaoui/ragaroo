from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import json
import os
import platform
import random
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .base import BaseRetriever
from .dataset import Dataset
from .evaluation.evaluation import Evaluator
from .pipeline.pipeline import Pipeline
from .report import Report
from .retrieval.hybrid import HybridRetriever
from .retrieval.lexical import BM25Retriever
from .retrieval.sparse import SparseRetriever
from .retrieval.dense import DenseRetriever


@dataclass(slots=True)
class Experiment:
    """Run one dataset through one or more pipelines and save a reproducible report."""

    dataset: Dataset
    pipelines: list[Pipeline]
    metrics: list[str] | None = None
    experiment_name: str | None = None
    output_dir: str | Path | None = None
    show_progress: bool = True
    store_query_results: bool = False
    warmup_queries: int = 0
    query_limit: int | None = None
    query_ids: list[str] | None = None
    notes: str | None = None
    tags: list[str] | None = None
    random_seed: int | None = None

    def run(self) -> Report:
        """Build pipeline indexes, evaluate selected queries, and save artifacts."""
        if not self.pipelines:
            raise ValueError("Experiment requires at least one pipeline")
        if self.query_limit is not None and self.query_limit < 0:
            raise ValueError("query_limit must be >= 0")
        if self.warmup_queries < 0:
            raise ValueError("warmup_queries must be >= 0")

        if not self.dataset.loaded:
            self.dataset.load()
        if self.random_seed is not None:
            self._set_random_seed(self.random_seed)

        run_timestamp = datetime.now(timezone.utc)
        resolved_output_dir = self._resolve_output_dir(run_timestamp)
        evaluator = Evaluator(
            metrics=self.metrics,
            store_query_results=self.store_query_results,
        )
        results = []
        retriever_registry: dict[tuple[str, str], Any] = {}
        selected_query_items = self._select_query_items()
        pipeline_iterator = tqdm(
            self.pipelines,
            total=len(self.pipelines),
            desc="Benchmarking pipelines",
            leave=True,
            disable=not self.show_progress,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for pipeline in pipeline_iterator:
            pipeline_iterator.set_postfix_str(f"{pipeline.name} | building index")
            pipeline.retriever = self._reuse_retriever(
                pipeline.retriever,
                retriever_registry,
            )
            if isinstance(pipeline.reranker, BaseRetriever):
                pipeline.reranker = self._reuse_retriever(
                    pipeline.reranker,
                    retriever_registry,
                )
            pipeline.prepare(self.dataset.corpus)
            build_status = self._build_status(pipeline.retriever)
            pipeline_iterator.set_postfix_str(f"{pipeline.name} | evaluating{build_status}")
            results.append(
                evaluator.evaluate(
                    self.dataset,
                    pipeline,
                    show_progress=self.show_progress,
                    query_items=selected_query_items,
                    warmup_queries=self.warmup_queries,
                    prepare_pipeline=False,
                )
            )

        report = Report(
            dataset_summary={
                **self.dataset.summary(),
                "experiment_query_count": len(selected_query_items),
                "warmup_queries": self.warmup_queries,
            },
            experiment_metadata=self._build_experiment_metadata(
                selected_query_items,
                resolved_output_dir,
                run_timestamp,
            ),
            results=results,
        )

        report.save(resolved_output_dir)

        return report

    def _select_query_items(self) -> list[tuple[str, str]]:
        if self.query_ids is None:
            items = [
                (query_id, query_text)
                for query_id, query_text in self.dataset.queries.items()
                if query_id in self.dataset.qrels
            ]
        else:
            missing = [query_id for query_id in self.query_ids if query_id not in self.dataset.queries]
            if missing:
                raise ValueError(f"Unknown query ids requested: {', '.join(missing)}")
            missing_qrels = [query_id for query_id in self.query_ids if query_id not in self.dataset.qrels]
            if missing_qrels:
                raise ValueError(f"Selected query ids have no qrels: {', '.join(missing_qrels)}")
            items = [(query_id, self.dataset.queries[query_id]) for query_id in self.query_ids]

        if self.query_limit is not None:
            return items[: self.query_limit]
        return items

    def _build_experiment_metadata(
        self,
        selected_query_items: list[tuple[str, str]],
        resolved_output_dir: Path,
        run_timestamp: datetime,
    ) -> dict[str, Any]:
        timestamp = run_timestamp.isoformat()
        return {
            "run_id": f"{self.dataset.dataset_hash}-{timestamp.replace(':', '').replace('-', '')}",
            "experiment_name": self._resolved_experiment_name(),
            "timestamp_utc": timestamp,
            "package_version": self._package_version("ragaroo"),
            "python_version": platform.python_version(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
            },
            "git": self._git_metadata(),
            "dependencies": self._dependency_versions(),
            "random_seed": self.random_seed,
            "notes": self.notes,
            "tags": self.tags or [],
            "settings": {
                "metrics": self.metrics,
                "query_limit": self.query_limit,
                "query_ids": self.query_ids,
                "selected_query_count": len(selected_query_items),
                "warmup_queries": self.warmup_queries,
                "store_query_results": self.store_query_results,
                "output_dir": str(resolved_output_dir),
            },
            "pipelines": [
                {
                    "name": pipeline.name,
                    "hash": pipeline.config_hash,
                    "config": pipeline.config_dict(),
                }
                for pipeline in self.pipelines
            ],
        }

    def _resolve_output_dir(self, run_timestamp: datetime) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir).resolve()
        dated_name = f"{self._resolved_experiment_name()}_{run_timestamp.strftime('%Y%m%d_%H%M%S')}"
        return (Path("results") / dated_name).resolve()

    def _resolved_experiment_name(self) -> str:
        base_name = self.experiment_name or self.dataset.summary().get("dataset_name") or "experiment"
        cleaned = re.sub(r"[\\/]+", "_", str(base_name)).strip()
        return cleaned or "experiment"

    def _reuse_retriever(
        self,
        retriever: Any,
        registry: dict[tuple[str, str], Any],
    ) -> Any:
        if isinstance(retriever, HybridRetriever):
            retriever.retriever_1 = self._reuse_retriever(retriever.retriever_1, registry)
            retriever.retriever_2 = self._reuse_retriever(retriever.retriever_2, registry)
            return retriever

        key = self._retriever_cache_key(retriever)
        if key is None:
            return retriever
        if key in registry:
            return registry[key]
        registry[key] = retriever
        return retriever

    def _retriever_cache_key(self, retriever: Any) -> tuple[str, str] | None:
        corpus = self.dataset.corpus
        config_signature = json.dumps(retriever.config_dict(), sort_keys=True, default=str)
        if isinstance(retriever, DenseRetriever | SparseRetriever):
            return retriever.__class__.__name__, f"{retriever._cache_path(corpus)}|{config_signature}"
        if isinstance(retriever, BM25Retriever):
            return retriever.__class__.__name__, f"{retriever._build_signature(corpus)}|{config_signature}"
        return None

    @staticmethod
    def _build_status(retriever: Any) -> str:
        cache_hit = getattr(retriever, "last_build_stats", {}).get("cache_hit")
        if cache_hit is None:
            return ""
        return " (cache hit)" if bool(cache_hit) else " (built)"

    @staticmethod
    def _set_random_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            return

    @staticmethod
    def _package_version(package_name: str) -> str | None:
        try:
            return importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            return None

    def _dependency_versions(self) -> dict[str, str | None]:
        packages = [
            "numpy",
            "pandas",
            "faiss-cpu",
            "sentence-transformers",
            "bm25s",
            "python-dotenv",
            "tqdm",
        ]
        return {
            package_name: self._package_version(package_name)
            for package_name in packages
        }

    @staticmethod
    def _git_metadata() -> dict[str, Any]:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
            return {
                "commit": commit,
                "dirty": dirty,
            }
        except Exception:
            return {
                "commit": None,
                "dirty": None,
            }
