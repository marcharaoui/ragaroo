from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .evaluation.evaluation import EvaluationResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(slots=True)
class Report:
    """Experiment results plus helpers for tabular exports, JSON, CSV, and plots."""

    dataset_summary: dict[str, Any]
    experiment_metadata: dict[str, Any]
    results: list[EvaluationResult]

    def summary(
        self,
        *,
        sort_by: str | None = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        frame = self.to_dataframe(sort_by=sort_by, ascending=ascending)
        print(frame.to_string(index=False))
        return frame

    def to_dataframe(
        self,
        *,
        sort_by: str | None = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result in self.results:
            row = {
                "pipeline": result.pipeline_name,
                "pipeline_hash": result.pipeline_hash,
                "query_count": result.query_count,
            }
            row.update(result.metrics)
            row.update(
                {
                    f"build_{key}": value
                    for key, value in result.build_stats.items()
                }
            )
            rows.append(row)

        frame = pd.DataFrame(rows)
        if sort_by is not None and sort_by in frame.columns:
            return frame.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
        return frame

    def to_query_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result in self.results:
            for query_result in result.query_results:
                row = {
                    "pipeline": result.pipeline_name,
                    "pipeline_hash": result.pipeline_hash,
                    "query_id": query_result.query_id,
                    "query": query_result.query,
                    "latency_ms": query_result.latency_ms,
                    **query_result.stage_latencies_ms,
                    **query_result.metric_values,
                    "retrieved_ids": json.dumps(
                        [document.corpus_id for document in query_result.results]
                    ),
                }
                rows.append(row)
        return pd.DataFrame(rows)

    def to_json(self, path: str | Path | None = None) -> str:
        payload = self.as_dict()
        rendered = json.dumps(payload, indent=2)
        if path is not None:
            Path(path).write_text(rendered, encoding="utf-8")
        return rendered

    def to_csv(self, path: str | Path | None = None) -> str:
        frame = self.to_dataframe()
        rendered = frame.to_csv(index=False)
        if path is not None:
            Path(path).write_text(rendered, encoding="utf-8", newline="")
        return rendered

    def save(self, directory: str | Path) -> Path:
        """Write report files, manifests, plots, and optional per-query artifacts."""
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "dataset": self.dataset_summary,
            "experiment": self.experiment_metadata,
        }

        metric_plot_paths = self._save_metric_plots(plots_dir)
        self.plot_latency(plots_dir / "latency_bar.png")
        self.plot_tradeoffs(plots_dir / "tradeoff_scatter.png")
        self.plot_quality_overview(plots_dir / "quality_overview.png")
        self.plot_latency_breakdown(plots_dir / "latency_breakdown.png")
        self.plot_build_times(plots_dir / "build_time_breakdown.png")
        self._save_query_artifacts(output_dir)
        if metric_plot_paths:
            manifest["plots"] = {
                "quality_metric_bars": [str(path.relative_to(output_dir)) for path in metric_plot_paths],
            }

        self.to_json(output_dir / "report.json")
        self.to_csv(output_dir / "report.csv")
        manifest_json = json.dumps(manifest, indent=2)
        (output_dir / "manifest.json").write_text(manifest_json, encoding="utf-8")
        (output_dir / "config.json").write_text(manifest_json, encoding="utf-8")

        return output_dir

    def plot_metrics(
        self,
        path: str | Path | None = None,
        *,
        metric_name: str | None = None,
    ) -> None:
        frame = self.to_dataframe()
        if frame.empty:
            return

        selected_metric = metric_name or self._pick_quality_metric(frame)
        if selected_metric is None:
            return

        figure, axis = plt.subplots(figsize=(8, 4))
        axis.bar(frame["pipeline"], frame[selected_metric], color="#2a6f97")
        axis.set_title(f"{selected_metric} by pipeline")
        axis.set_ylabel(selected_metric)
        axis.set_xlabel("pipeline")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)
        self._set_zoomed_axis_limits(axis, frame[selected_metric], axis_name="y", prefer_zero_floor=False)
        figure.tight_layout()

        if path is not None:
            figure.savefig(path, dpi=150)
        plt.close(figure)

    def plot_latency(
        self,
        path: str | Path | None = None,
        *,
        latency_column: str | None = None,
    ) -> None:
        frame = self.to_dataframe()
        if frame.empty:
            return

        selected_latency = latency_column or self._pick_latency_metric(frame)
        if selected_latency is None:
            return

        figure, axis = plt.subplots(figsize=(8, 4))
        axis.bar(frame["pipeline"], frame[selected_latency], color="#d62828")
        axis.set_title(f"{selected_latency} by pipeline")
        axis.set_ylabel(selected_latency)
        axis.set_xlabel("pipeline")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)
        self._set_zoomed_axis_limits(axis, frame[selected_latency], axis_name="y", prefer_zero_floor=False)
        figure.tight_layout()

        if path is not None:
            figure.savefig(path, dpi=150)
        plt.close(figure)

    def plot_quality_overview(self, path: str | Path | None = None) -> None:
        frame = self.to_dataframe()
        metric_columns = self._selected_quality_metrics(frame)
        if frame.empty or not metric_columns:
            return

        figure, axis = plt.subplots(figsize=(max(8, len(metric_columns) * 1.4), 4.8))
        x = np.arange(len(metric_columns))
        width = min(0.8 / len(frame), 0.18)
        for index, (_, row) in enumerate(frame.iterrows()):
            offset = (index - (len(frame) - 1) / 2) * width
            values = [row[metric_name] for metric_name in metric_columns]
            axis.bar(x + offset, values, width=width, label=row["pipeline"])

        axis.set_title("Quality metrics overview")
        axis.set_ylabel("score")
        axis.set_xlabel("metric")
        axis.set_xticks(x)
        axis.set_xticklabels(metric_columns, rotation=20)
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)
        axis.legend(loc="best", fontsize=8)
        combined_values = pd.concat([frame[column] for column in metric_columns], ignore_index=True)
        self._set_zoomed_axis_limits(axis, combined_values, axis_name="y", prefer_zero_floor=False)
        figure.tight_layout()

        if path is not None:
            figure.savefig(path, dpi=150)
        plt.close(figure)

    def plot_latency_breakdown(self, path: str | Path | None = None) -> None:
        frame = self.to_dataframe()
        latency_columns = [
            column
            for column in [
                "query_augmentation_latency_ms",
                "retrieval_latency_ms",
                "rerank_latency_ms",
            ]
            if column in frame.columns
        ]
        if frame.empty or not latency_columns:
            return

        figure, axis = plt.subplots(figsize=(max(8, len(frame) * 1.1), 4.8))
        x = np.arange(len(frame))
        bottom = np.zeros(len(frame), dtype=np.float32)
        colors = ["#8ecae6", "#219ebc", "#ffb703"]
        for index, column in enumerate(latency_columns):
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            axis.bar(x, values, bottom=bottom, label=column, color=colors[index % len(colors)])
            bottom += values

        axis.set_title("Latency breakdown by pipeline")
        axis.set_ylabel("milliseconds")
        axis.set_xlabel("pipeline")
        axis.set_xticks(x)
        axis.set_xticklabels(frame["pipeline"], rotation=20)
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)
        axis.legend(loc="best", fontsize=8)
        self._set_zoomed_axis_limits(axis, pd.Series(bottom), axis_name="y", prefer_zero_floor=True)
        figure.tight_layout()

        if path is not None:
            figure.savefig(path, dpi=150)
        plt.close(figure)

    def plot_build_times(self, path: str | Path | None = None) -> None:
        frame = self.to_dataframe()
        build_columns = [
            column
            for column in [
                "build_embedding_time_s",
                "build_index_build_time_s",
                "build_load_time_s",
                "build_total_build_time_s",
            ]
            if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
        if frame.empty or not build_columns:
            return

        figure, axis = plt.subplots(figsize=(max(8, len(frame) * 1.2), 4.8))
        x = np.arange(len(frame))
        width = min(0.8 / len(build_columns), 0.18)
        for index, column in enumerate(build_columns):
            offset = (index - (len(build_columns) - 1) / 2) * width
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            axis.bar(x + offset, values, width=width, label=column)

        axis.set_title("Build-time metrics by pipeline")
        axis.set_ylabel("seconds")
        axis.set_xlabel("pipeline")
        axis.set_xticks(x)
        axis.set_xticklabels(frame["pipeline"], rotation=20)
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)
        axis.legend(loc="best", fontsize=8)
        combined_values = pd.concat([pd.to_numeric(frame[column], errors="coerce") for column in build_columns], ignore_index=True)
        self._set_zoomed_axis_limits(axis, combined_values.dropna(), axis_name="y", prefer_zero_floor=True)
        figure.tight_layout()

        if path is not None:
            figure.savefig(path, dpi=150)
        plt.close(figure)

    def plot_tradeoffs(
        self,
        path: str | Path | None = None,
        *,
        quality_metric: str | None = None,
        latency_metric: str | None = None,
    ) -> None:
        frame = self.to_dataframe()
        if frame.empty:
            return

        selected_quality = quality_metric or self._pick_quality_metric(frame, preferred_prefix="mrr@")
        selected_latency = latency_metric or self._pick_latency_metric(frame)
        if selected_quality is None or selected_latency is None:
            return

        figure, axis = plt.subplots(figsize=(6, 5))
        colors = plt.cm.tab10.colors
        markers = ["o", "s", "^", "D", "P", "X", "v", "*", "<", ">"]
        for index, (_, row) in enumerate(frame.iterrows()):
            axis.scatter(
                row[selected_latency],
                row[selected_quality],
                color=colors[index % len(colors)],
                marker=markers[index % len(markers)],
                s=90,
                edgecolors="black",
                linewidths=0.5,
            )
            axis.annotate(row["pipeline"], (row[selected_latency], row[selected_quality]))
        axis.set_xlabel(selected_latency)
        axis.set_ylabel(selected_quality)
        axis.set_title("Quality / latency tradeoff")
        axis.grid(True, linestyle="--", alpha=0.35)
        self._set_zoomed_axis_limits(axis, frame[selected_latency], axis_name="x", prefer_zero_floor=False)
        self._set_zoomed_axis_limits(axis, frame[selected_quality], axis_name="y", prefer_zero_floor=False)
        figure.tight_layout()

        if path is not None:
            figure.savefig(path, dpi=150)
        plt.close(figure)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_summary,
            "experiment": self.experiment_metadata,
            "results": [self._result_to_dict(result) for result in self.results],
        }

    @staticmethod
    def _result_to_dict(result: EvaluationResult) -> dict[str, Any]:
        return asdict(result)

    def _save_query_artifacts(self, output_dir: Path) -> None:
        query_frame = self.to_query_dataframe()
        if query_frame.empty:
            return

        query_frame.to_csv(output_dir / "query_metrics.csv", index=False)
        with (output_dir / "query_results.jsonl").open("w", encoding="utf-8") as handle:
            for result in self.results:
                for query_result in result.query_results:
                    payload = {
                        "pipeline": result.pipeline_name,
                        "pipeline_hash": result.pipeline_hash,
                        **asdict(query_result),
                    }
                    handle.write(json.dumps(payload) + "\n")

    def _save_metric_plots(self, plots_dir: Path) -> list[Path]:
        frame = self.to_dataframe()
        metric_columns = self._selected_quality_metrics(frame)
        if not metric_columns:
            return []

        saved_paths: list[Path] = []
        for metric_name in metric_columns:
            metric_path = plots_dir / f"metrics_bar_{self._safe_metric_filename(metric_name)}.png"
            self.plot_metrics(metric_path, metric_name=metric_name)
            saved_paths.append(metric_path)
        return saved_paths

    def _pick_quality_metric(
        self,
        frame: pd.DataFrame,
        *,
        preferred_prefix: str | None = None,
    ) -> str | None:
        metric_columns = self._selected_quality_metrics(frame)
        if preferred_prefix is not None:
            for column in metric_columns:
                if column.startswith(preferred_prefix):
                    return column
        for prefix in ["mrr@", "ndcg@", "recall@", "precision@", "map@", "hit_rate@"]:
            for column in metric_columns:
                if column.startswith(prefix):
                    return column
        return None

    @staticmethod
    def _pick_latency_metric(frame: pd.DataFrame) -> str | None:
        for column in ["latency_ms", "retrieval_latency_ms", "p95_latency_ms", "avg_query_latency_ms"]:
            if column in frame.columns:
                return column
        return None

    @staticmethod
    def _quality_metric_columns(frame: pd.DataFrame) -> list[str]:
        return [
            column
            for column in frame.columns
            if any(
                column.startswith(prefix)
                for prefix in ["mrr@", "ndcg@", "recall@", "precision@", "map@", "hit_rate@"]
            )
        ]

    def _selected_quality_metrics(self, frame: pd.DataFrame) -> list[str]:
        available_metrics = self._quality_metric_columns(frame)
        requested_metrics = self.experiment_metadata.get("settings", {}).get("metrics")
        if not isinstance(requested_metrics, list) or not requested_metrics:
            return available_metrics
        return [
            metric_name
            for metric_name in requested_metrics
            if metric_name in available_metrics
        ]

    @staticmethod
    def _safe_metric_filename(metric_name: str) -> str:
        return metric_name.replace("@", "_at_").replace("/", "_").replace(" ", "_")

    @staticmethod
    def _set_zoomed_axis_limits(
        axis: Any,
        values: pd.Series,
        *,
        axis_name: str,
        prefer_zero_floor: bool,
    ) -> None:
        numeric_values = pd.to_numeric(values, errors="coerce").dropna()
        if numeric_values.empty:
            return

        minimum = float(numeric_values.min())
        maximum = float(numeric_values.max())
        if math.isclose(minimum, maximum):
            span = max(abs(minimum) * 0.05, 1e-3)
            lower = minimum - span
            upper = maximum + span
        else:
            span = maximum - minimum
            margin = max(span * 0.1, 1e-3)
            lower = minimum - margin
            upper = maximum + margin

        if prefer_zero_floor and lower > 0.0:
            lower = 0.0

        if axis_name == "x":
            axis.set_xlim(lower, upper)
        else:
            axis.set_ylim(lower, upper)
