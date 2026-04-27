import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ragaroo import BM25Retriever, Dataset, DenseRetriever, Experiment, Pipeline, SparseRetriever
from ragaroo.base import BaseEmbedder


class DummyEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        self.embedding_dim = 2
        self.encode_documents_calls = 0

    def encode_documents(self, texts, normalize_embeddings=True):
        self.encode_documents_calls += 1
        vectors = []
        for text in texts:
            if "apple" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            if "apple" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class DummySparseEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        self.encode_documents_calls = 0

    def encode_documents(self, texts, normalize_embeddings=True):
        self.encode_documents_calls += 1
        vectors = []
        for text in texts:
            if "apple" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return torch.tensor(vectors, dtype=torch.float32).to_sparse_coo()

    def encode_queries(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            if "apple" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return torch.tensor(vectors, dtype=torch.float32).to_sparse_coo()


class TestExperiment(unittest.TestCase):
    def test_run_and_save_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                        json.dumps({"id": "d2", "text": "banana potassium benefits"}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "q1", "text": "apple facts"}),
                        json.dumps({"id": "q2", "text": "banana benefits"}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            pipeline = Pipeline(
                name="dummy_dense",
                retriever=DenseRetriever(embedder=DummyEmbedder(), top_k=2),
            )

            output_dir = Path(tmpdir) / "results"
            report = Experiment(
                dataset=dataset,
                pipelines=[pipeline],
                output_dir=output_dir,
                metrics=["recall@2", "mrr@2", "ndcg@2", "latency_ms"],
            ).run()

            frame = report.to_dataframe()
            self.assertEqual(len(frame), 1)
            self.assertIn("mrr@2", frame.columns)
            self.assertIn("latency_ms", frame.columns)
            self.assertNotIn("retrieval_latency_ms", frame.columns)
            self.assertTrue((output_dir / "report.json").exists())
            self.assertTrue((output_dir / "report.csv").exists())
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertFalse((output_dir / "plots" / "metrics_bar.png").exists())
            self.assertTrue((output_dir / "plots" / "metrics_bar_recall_at_2.png").exists())
            self.assertTrue((output_dir / "plots" / "metrics_bar_mrr_at_2.png").exists())
            self.assertTrue((output_dir / "plots" / "metrics_bar_ndcg_at_2.png").exists())
            self.assertTrue((output_dir / "plots" / "latency_bar.png").exists())
            self.assertTrue((output_dir / "plots" / "tradeoff_scatter.png").exists())
            self.assertTrue((output_dir / "plots" / "quality_overview.png").exists())
            self.assertFalse((output_dir / "plots" / "latency_breakdown.png").exists())
            self.assertTrue((output_dir / "plots" / "build_time_breakdown.png").exists())
            self.assertEqual(report.results[0].query_results, [])
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("experiment", manifest)
            self.assertIn("dataset_hash", manifest["dataset"])
            self.assertEqual(manifest["experiment"]["settings"]["store_query_results"], False)
            self.assertEqual(
                manifest["plots"]["quality_metric_bars"],
                [
                    str(Path("plots") / "metrics_bar_recall_at_2.png"),
                    str(Path("plots") / "metrics_bar_mrr_at_2.png"),
                    str(Path("plots") / "metrics_bar_ndcg_at_2.png"),
                ],
            )

    def test_can_opt_in_to_store_query_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                json.dumps({"id": "q1", "text": "apple facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            pipeline = Pipeline(
                name="dummy_dense",
                retriever=DenseRetriever(embedder=DummyEmbedder(), top_k=1),
            )

            report = Experiment(
                dataset=dataset,
                pipelines=[pipeline],
                store_query_results=True,
                output_dir=Path(tmpdir) / "results",
            ).run()

            self.assertEqual(len(report.results[0].query_results), 1)
            self.assertIn("retrieval_latency_ms", report.results[0].query_results[0].stage_latencies_ms)
            self.assertTrue((Path(tmpdir) / "results" / "query_metrics.csv").exists())
            self.assertTrue((Path(tmpdir) / "results" / "query_results.jsonl").exists())

    def test_query_limit_and_warmup_reduce_evaluated_queries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                        json.dumps({"id": "d2", "text": "banana potassium benefits"}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "q1", "text": "apple facts"}),
                        json.dumps({"id": "q2", "text": "banana benefits"}),
                        json.dumps({"id": "q3", "text": "apple nutrition"}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\nq3\td1\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            pipeline = Pipeline(
                name="dummy_dense",
                retriever=DenseRetriever(embedder=DummyEmbedder(), top_k=1),
            )

            report = Experiment(
                dataset=dataset,
                pipelines=[pipeline],
                query_limit=2,
                warmup_queries=1,
                store_query_results=True,
                output_dir=Path(tmpdir) / "results",
            ).run()

            self.assertEqual(report.results[0].query_count, 1)
            self.assertEqual(len(report.results[0].query_results), 1)
            self.assertEqual(report.results[0].query_results[0].query_id, "q2")

    def test_query_limit_zero_evaluates_zero_queries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                json.dumps({"id": "q1", "text": "apple facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            pipeline = Pipeline(
                name="dummy_dense",
                retriever=DenseRetriever(embedder=DummyEmbedder(), top_k=1),
            )

            report = Experiment(
                dataset=dataset,
                pipelines=[pipeline],
                query_limit=0,
                store_query_results=True,
                output_dir=Path(tmpdir) / "results",
            ).run()

            self.assertEqual(report.results[0].query_count, 0)
            self.assertEqual(report.results[0].metrics, {})
            self.assertEqual(report.results[0].query_results, [])

    def test_negative_query_limit_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                json.dumps({"id": "q1", "text": "apple facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            pipeline = Pipeline(
                name="dummy_dense",
                retriever=DenseRetriever(embedder=DummyEmbedder(), top_k=1),
            )

            with self.assertRaisesRegex(ValueError, "query_limit must be >= 0"):
                Experiment(
                    dataset=dataset,
                    pipelines=[pipeline],
                    query_limit=-1,
                    output_dir=Path(tmpdir) / "results",
                ).run()

    def test_negative_warmup_queries_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                json.dumps({"id": "q1", "text": "apple facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            pipeline = Pipeline(
                name="dummy_dense",
                retriever=DenseRetriever(embedder=DummyEmbedder(), top_k=1),
            )

            with self.assertRaisesRegex(ValueError, "warmup_queries must be >= 0"):
                Experiment(
                    dataset=dataset,
                    pipelines=[pipeline],
                    warmup_queries=-1,
                    output_dir=Path(tmpdir) / "results",
                ).run()

    def test_query_ids_select_specific_queries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "q1", "text": "apple facts"}),
                        json.dumps({"id": "q2", "text": "apple nutrition"}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td1\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            pipeline = Pipeline(
                name="dummy_dense",
                retriever=DenseRetriever(embedder=DummyEmbedder(), top_k=1),
            )

            report = Experiment(
                dataset=dataset,
                pipelines=[pipeline],
                query_ids=["q2"],
                store_query_results=True,
                output_dir=Path(tmpdir) / "results",
            ).run()

            self.assertEqual(report.results[0].query_count, 1)
            self.assertEqual(report.results[0].query_results[0].query_id, "q2")

    def test_default_query_selection_uses_only_queries_with_qrels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "q1", "text": "apple facts"}),
                        json.dumps({"id": "q2", "text": "unlabeled query"}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            pipeline = Pipeline(
                name="dummy_dense",
                retriever=DenseRetriever(embedder=DummyEmbedder(), top_k=1),
            )

            report = Experiment(
                dataset=dataset,
                pipelines=[pipeline],
                store_query_results=True,
                output_dir=Path(tmpdir) / "results",
            ).run()

            self.assertEqual(report.results[0].query_count, 1)
            self.assertEqual(report.results[0].query_results[0].query_id, "q1")
            self.assertEqual(report.results[0].metrics["query_augmentation_latency_ms"], 0.0)
            self.assertEqual(report.results[0].metrics["rerank_latency_ms"], 0.0)

    def test_query_ids_without_qrels_raise_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "q1", "text": "apple facts"}),
                        json.dumps({"id": "q2", "text": "unlabeled query"}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            pipeline = Pipeline(
                name="dummy_dense",
                retriever=DenseRetriever(embedder=DummyEmbedder(), top_k=1),
            )

            with self.assertRaises(ValueError):
                Experiment(
                    dataset=dataset,
                    pipelines=[pipeline],
                    query_ids=["q2"],
                    output_dir=Path(tmpdir) / "results",
                ).run()

    def test_default_output_dir_uses_results_and_dataset_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "nfcorpus"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                json.dumps({"id": "q1", "text": "apple facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            pipeline = Pipeline(
                name="dummy_dense",
                retriever=DenseRetriever(embedder=DummyEmbedder(), top_k=1),
            )

            original_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                Experiment(
                    dataset=dataset,
                    pipelines=[pipeline],
                ).run()
            finally:
                os.chdir(original_cwd)

            results_root = Path(tmpdir) / "results"
            matching_dirs = [path for path in results_root.iterdir() if path.is_dir() and path.name.startswith("nfcorpus_")]
            self.assertEqual(len(matching_dirs), 1)
            default_output_dir = matching_dirs[0]
            self.assertTrue((default_output_dir / "report.json").exists())
            manifest = json.loads((default_output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["experiment"]["experiment_name"], "nfcorpus")
            self.assertTrue(
                Path(manifest["experiment"]["settings"]["output_dir"]).samefile(default_output_dir)
            )

    def test_reuses_identical_dense_retriever_instances_within_one_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                        json.dumps({"id": "d2", "text": "banana potassium benefits"}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                json.dumps({"id": "q1", "text": "apple facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            cache_dir = Path(tmpdir) / "indexes"
            embedder_1 = DummyEmbedder()
            embedder_2 = DummyEmbedder()

            pipeline_1 = Pipeline(
                name="dense_a",
                retriever=DenseRetriever(embedder=embedder_1, top_k=2, cache_dir=cache_dir),
            )
            pipeline_2 = Pipeline(
                name="dense_b",
                retriever=DenseRetriever(embedder=embedder_2, top_k=2, cache_dir=cache_dir),
            )

            Experiment(
                dataset=dataset,
                pipelines=[pipeline_1, pipeline_2],
                output_dir=Path(tmpdir) / "results",
            ).run()

            self.assertIs(pipeline_1.retriever, pipeline_2.retriever)
            self.assertEqual(embedder_1.encode_documents_calls, 1)
            self.assertEqual(embedder_2.encode_documents_calls, 0)

    def test_does_not_reuse_retrievers_with_different_top_k(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                        json.dumps({"id": "d2", "text": "apple clinical trial"}),
                        json.dumps({"id": "d3", "text": "banana potassium benefits"}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                json.dumps({"id": "q1", "text": "apple facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            cache_dir = Path(tmpdir) / "indexes"
            pipeline_1 = Pipeline(
                name="bm25_top1",
                retriever=BM25Retriever(top_k=1, cache_dir=cache_dir),
            )
            pipeline_2 = Pipeline(
                name="bm25_top3",
                retriever=BM25Retriever(top_k=3, cache_dir=cache_dir),
            )

            report = Experiment(
                dataset=dataset,
                pipelines=[pipeline_1, pipeline_2],
                store_query_results=True,
                output_dir=Path(tmpdir) / "results",
            ).run()

            self.assertIsNot(pipeline_1.retriever, pipeline_2.retriever)
            self.assertEqual(pipeline_1.retriever.top_k, 1)
            self.assertEqual(pipeline_2.retriever.top_k, 3)
            self.assertEqual(len(report.results[0].query_results[0].results), 1)
            self.assertEqual(len(report.results[1].query_results[0].results), 3)

    def test_reuses_identical_sparse_retriever_instances_within_one_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                        json.dumps({"id": "d2", "text": "banana potassium benefits"}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                json.dumps({"id": "q1", "text": "apple facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            cache_dir = Path(tmpdir) / "indexes"
            embedder_1 = DummySparseEmbedder()
            embedder_2 = DummySparseEmbedder()

            pipeline_1 = Pipeline(
                name="sparse_a",
                retriever=SparseRetriever(embedder=embedder_1, top_k=2, cache_dir=cache_dir),
            )
            pipeline_2 = Pipeline(
                name="sparse_b",
                retriever=SparseRetriever(embedder=embedder_2, top_k=2, cache_dir=cache_dir),
            )

            Experiment(
                dataset=dataset,
                pipelines=[pipeline_1, pipeline_2],
                output_dir=Path(tmpdir) / "results",
            ).run()

            self.assertIs(pipeline_1.retriever, pipeline_2.retriever)
            self.assertEqual(embedder_1.encode_documents_calls, 1)
            self.assertEqual(embedder_2.encode_documents_calls, 0)


if __name__ == "__main__":
    unittest.main()
