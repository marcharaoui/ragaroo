import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ragaroo import BM25Retriever, Dataset, DenseRetriever, Experiment, HybridRetriever, Pipeline
from ragaroo.base import BaseEmbedder
from ragaroo.reranking.cross_encoder import CrossEncoderReranker


class DummyEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        self.embedding_dim = 2

    def encode_documents(self, texts, normalize_embeddings=True):
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


class FakeCrossEncoderModel:
    model_name_or_path = "fake-cross-encoder"

    def score(self, query, documents):
        return np.asarray(
            [10.0 if "apple" in document else 1.0 for document in documents],
            dtype=np.float32,
        )


class TestMixedPipelines(unittest.TestCase):
    def test_experiment_runs_mixed_pipeline_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "corpus.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                        json.dumps({"id": "d2", "text": "banana potassium benefits"}),
                        json.dumps({"id": "d3", "text": "apple fruit overview"}),
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
                "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td3\t1\nq2\td2\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            embedder = DummyEmbedder()
            pipelines = [
                Pipeline(name="bm25", retriever=BM25Retriever(top_k=2)),
                Pipeline(name="dense", retriever=DenseRetriever(embedder=embedder, top_k=2)),
                Pipeline(
                    name="hybrid",
                    retriever=HybridRetriever(
                        retriever_1=DenseRetriever(embedder=DummyEmbedder(), top_k=2),
                        retriever_2=BM25Retriever(top_k=2),
                        top_k=2,
                    ),
                ),
                Pipeline(
                    name="dense_reranked",
                    retriever=DenseRetriever(embedder=DummyEmbedder(), top_k=3),
                    reranker=CrossEncoderReranker(model=FakeCrossEncoderModel(), top_k=2),
                ),
                Pipeline(
                    name="bm25_reranked_by_dense",
                    retriever=BM25Retriever(top_k=3),
                    reranker=DenseRetriever(embedder=DummyEmbedder(), top_k=2),
                ),
            ]

            report = Experiment(
                dataset=dataset,
                pipelines=pipelines,
                query_limit=2,
                output_dir=Path(tmpdir) / "results",
            ).run()

            frame = report.to_dataframe()
            self.assertEqual(len(frame), 5)
            self.assertIn("pipeline", frame.columns)
            self.assertTrue(all(count == 2 for count in frame["query_count"].tolist()))
