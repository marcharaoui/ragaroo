import tempfile
import unittest
from pathlib import Path

from ragaroo.evaluation.evaluation import EvaluationResult, QueryResult
from ragaroo.report import Report
from ragaroo.retrieval.types import RetrievedDocument


class TestReport(unittest.TestCase):
    def test_to_query_dataframe_and_save_query_artifacts(self):
        report = Report(
            dataset_summary={"dataset_name": "toy", "dataset_hash": "abc123"},
            experiment_metadata={
                "run_id": "run-1",
                "settings": {
                    "store_query_results": True,
                    "metrics": ["mrr@10", "latency_ms"],
                },
            },
            results=[
                EvaluationResult(
                    pipeline_name="dense",
                    pipeline_hash="hash1",
                    metrics={"mrr@10": 1.0, "latency_ms": 12.5},
                    query_results=[
                        QueryResult(
                            query_id="q1",
                            query="apple facts",
                            latency_ms=12.5,
                            stage_latencies_ms={
                                "query_augmentation_latency_ms": 0.0,
                                "retrieval_latency_ms": 12.0,
                                "rerank_latency_ms": 0.5,
                            },
                            results=[
                                RetrievedDocument(
                                    corpus_id="d1",
                                    score=0.9,
                                    text="apple nutrition facts",
                                    metadata={"source": "toy"},
                                )
                            ],
                            metric_values={"mrr@10": 1.0},
                        )
                    ],
                    build_stats={"cache_hit": 1.0},
                    query_count=1,
                )
            ],
        )

        query_frame = report.to_query_dataframe()
        self.assertEqual(len(query_frame), 1)
        self.assertIn("retrieved_ids", query_frame.columns)
        self.assertIn("retrieval_latency_ms", query_frame.columns)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report.save(output_dir)
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "query_metrics.csv").exists())
            self.assertTrue((output_dir / "query_results.jsonl").exists())
            self.assertFalse((output_dir / "plots" / "metrics_bar.png").exists())
            self.assertTrue((output_dir / "plots" / "metrics_bar_mrr_at_10.png").exists())
            self.assertTrue((output_dir / "plots" / "latency_bar.png").exists())
            self.assertTrue((output_dir / "plots" / "tradeoff_scatter.png").exists())

    def test_to_dataframe_can_sort(self):
        report = Report(
            dataset_summary={"dataset_name": "toy"},
            experiment_metadata={"run_id": "run-1"},
            results=[
                EvaluationResult(
                    pipeline_name="b",
                    pipeline_hash="hash-b",
                    metrics={"mrr@10": 0.1},
                    query_results=[],
                    build_stats={},
                    query_count=1,
                ),
                EvaluationResult(
                    pipeline_name="a",
                    pipeline_hash="hash-a",
                    metrics={"mrr@10": 0.9},
                    query_results=[],
                    build_stats={},
                    query_count=1,
                ),
            ],
        )

        frame = report.to_dataframe(sort_by="mrr@10")
        self.assertEqual(frame.iloc[0]["pipeline"], "a")

    def test_plot_methods_save_chart_files(self):
        report = Report(
            dataset_summary={"dataset_name": "toy"},
            experiment_metadata={"run_id": "run-1"},
            results=[
                EvaluationResult(
                    pipeline_name="a",
                    pipeline_hash="hash-a",
                    metrics={
                        "ndcg@10": 0.91,
                        "mrr@10": 0.89,
                        "latency_ms": 12.0,
                        "query_augmentation_latency_ms": 0.0,
                        "retrieval_latency_ms": 11.5,
                        "rerank_latency_ms": 0.5,
                    },
                    query_results=[],
                    build_stats={"embedding_time_s": 0.3, "index_build_time_s": 0.8, "total_build_time_s": 1.1},
                    query_count=10,
                ),
                EvaluationResult(
                    pipeline_name="b",
                    pipeline_hash="hash-b",
                    metrics={
                        "ndcg@10": 0.94,
                        "mrr@10": 0.92,
                        "latency_ms": 14.5,
                        "query_augmentation_latency_ms": 0.2,
                        "retrieval_latency_ms": 12.0,
                        "rerank_latency_ms": 2.3,
                    },
                    query_results=[],
                    build_stats={"embedding_time_s": 0.2, "index_build_time_s": 0.6, "total_build_time_s": 0.8},
                    query_count=10,
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report.plot_metrics(output_dir / "metrics.png", metric_name="ndcg@10")
            report.plot_latency(output_dir / "latency.png", latency_column="latency_ms")
            report.plot_tradeoffs(
                output_dir / "tradeoff.png",
                quality_metric="ndcg@10",
                latency_metric="latency_ms",
            )
            report.plot_quality_overview(output_dir / "quality_overview.png")
            report.plot_latency_breakdown(output_dir / "latency_breakdown.png")
            report.plot_build_times(output_dir / "build_time_breakdown.png")

            self.assertTrue((output_dir / "metrics.png").exists())
            self.assertTrue((output_dir / "latency.png").exists())
            self.assertTrue((output_dir / "tradeoff.png").exists())
            self.assertTrue((output_dir / "quality_overview.png").exists())
            self.assertTrue((output_dir / "latency_breakdown.png").exists())
            self.assertTrue((output_dir / "build_time_breakdown.png").exists())

    def test_save_creates_metric_plot_for_each_requested_quality_metric(self):
        report = Report(
            dataset_summary={"dataset_name": "toy"},
            experiment_metadata={
                "run_id": "run-1",
                "settings": {
                    "metrics": ["recall@10", "precision@10", "ndcg@10", "latency_ms"],
                },
            },
            results=[
                EvaluationResult(
                    pipeline_name="a",
                    pipeline_hash="hash-a",
                    metrics={
                        "recall@10": 0.8,
                        "precision@10": 0.3,
                        "ndcg@10": 0.9,
                        "mrr@10": 0.88,
                        "latency_ms": 12.0,
                    },
                    query_results=[],
                    build_stats={},
                    query_count=10,
                ),
                EvaluationResult(
                    pipeline_name="b",
                    pipeline_hash="hash-b",
                    metrics={
                        "recall@10": 0.82,
                        "precision@10": 0.28,
                        "ndcg@10": 0.92,
                        "mrr@10": 0.9,
                        "latency_ms": 13.0,
                    },
                    query_results=[],
                    build_stats={},
                    query_count=10,
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report.save(output_dir)

            self.assertFalse((output_dir / "plots" / "metrics_bar.png").exists())
            self.assertTrue((output_dir / "plots" / "metrics_bar_recall_at_10.png").exists())
            self.assertTrue((output_dir / "plots" / "metrics_bar_precision_at_10.png").exists())
            self.assertTrue((output_dir / "plots" / "metrics_bar_ndcg_at_10.png").exists())
            self.assertFalse((output_dir / "plots" / "metrics_bar_mrr_at_10.png").exists())


if __name__ == "__main__":
    unittest.main()
