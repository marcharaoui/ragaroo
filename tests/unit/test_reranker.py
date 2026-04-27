import unittest

import numpy as np

from ragaroo import Pipeline
from ragaroo.base import BaseEmbedder, BaseRetriever
from ragaroo.reranking.cross_encoder import CrossEncoderReranker
from ragaroo.retrieval.dense.dense import DenseRetriever
from ragaroo.retrieval.types import RetrievedDocument


class FakeEmbedder(BaseEmbedder):
    def __init__(self):
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
        scores = []
        for document in documents:
            if "best" in document:
                scores.append(10.0)
            elif "apple" in document:
                scores.append(5.0)
            else:
                scores.append(1.0)
        return np.asarray(scores, dtype=np.float32)


class FixedRetriever(BaseRetriever):
    def __init__(self, results, top_k=None):
        self.results = results
        self.top_k = top_k or len(results)
        self.build_calls = 0
        self.last_build_stats = {}
        self.last_query_stats = {}

    def build_index(self, corpus):
        self.build_calls += 1
        self.last_build_stats = {"total_build_time_s": 0.0}

    def retrieve(self, query, top_k=None):
        k = top_k or self.top_k
        self.last_query_stats = {"total_query_time_s": 0.0, "top_k": float(k)}
        return self.results[:k]

    def config_dict(self):
        return {"type": "FixedRetriever", "top_k": self.top_k}


class TestReranker(unittest.TestCase):
    def test_reranker_reorders_retrieved_documents(self):
        corpus = {
            "doc1": {"text": "apple baseline document", "metadata": {}},
            "doc2": {"text": "apple best document", "metadata": {}},
            "doc3": {"text": "banana document", "metadata": {}},
        }

        pipeline = Pipeline(
            name="dense_plus_reranker",
            retriever=DenseRetriever(embedder=FakeEmbedder(), top_k=3),
            reranker=CrossEncoderReranker(model=FakeCrossEncoderModel(), top_k=2),
        )

        pipeline.prepare(corpus)
        results = pipeline.retrieve("apple query")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].corpus_id, "doc2")
        self.assertEqual(results[1].corpus_id, "doc1")
        self.assertEqual(pipeline.last_query_stats["query_augmentation_time_s"], 0.0)
        self.assertGreaterEqual(pipeline.last_query_stats["retrieval_time_s"], 0.0)
        self.assertGreaterEqual(pipeline.last_query_stats["rerank_time_s"], 0.0)

    def test_retriever_can_be_used_as_reranker(self):
        corpus = {
            "doc1": {"text": "apple document", "metadata": {"source": "candidate"}},
            "doc2": {"text": "banana document", "metadata": {}},
            "doc3": {"text": "banana fallback", "metadata": {}},
        }
        primary_retriever = FixedRetriever(
            [
                RetrievedDocument("doc3", 0.9, "banana fallback"),
                RetrievedDocument("doc1", 0.8, "apple document", {"source": "candidate"}),
            ],
            top_k=2,
        )
        dense_reranker = DenseRetriever(embedder=FakeEmbedder(), top_k=2)
        pipeline = Pipeline(
            name="fixed_plus_dense",
            retriever=primary_retriever,
            reranker=dense_reranker,
        )

        pipeline.prepare(corpus)
        results = pipeline.retrieve("apple query")

        self.assertEqual(primary_retriever.build_calls, 1)
        self.assertTrue(dense_reranker.is_built)
        self.assertIn("reranker_total_build_time_s", pipeline.last_build_stats)
        self.assertEqual([result.corpus_id for result in results], ["doc1", "doc3"])
        self.assertEqual(results[0].metadata, {"source": "candidate"})
        self.assertIn("matched_candidates", dense_reranker.last_rerank_stats)

    def test_retriever_reranker_keeps_unmatched_candidates_at_the_end(self):
        corpus = {
            "doc1": {"text": "alpha", "metadata": {}},
            "doc2": {"text": "beta", "metadata": {}},
            "doc3": {"text": "gamma", "metadata": {}},
        }
        primary_retriever = FixedRetriever(
            [
                RetrievedDocument("doc1", 0.9, "alpha"),
                RetrievedDocument("doc2", 0.8, "beta"),
                RetrievedDocument("doc3", 0.7, "gamma"),
            ],
            top_k=3,
        )
        reranker = FixedRetriever(
            [
                RetrievedDocument("doc2", 10.0, "beta"),
            ],
            top_k=3,
        )
        pipeline = Pipeline(
            name="fixed_plus_fixed",
            retriever=primary_retriever,
            reranker=reranker,
        )

        pipeline.prepare(corpus)
        results = pipeline.retrieve("query")

        self.assertEqual([result.corpus_id for result in results], ["doc2", "doc1", "doc3"])
        self.assertEqual(results[0].score, 10.0)

    def test_empty_cross_encoder_rerank_clears_stats(self):
        reranker = CrossEncoderReranker(model=FakeCrossEncoderModel(), top_k=2)
        reranker.last_rerank_stats = {"rerank_time_s": 99.0}

        results = reranker.rerank("query", [])

        self.assertEqual(results, [])
        self.assertEqual(reranker.last_rerank_stats["rerank_time_s"], 0.0)
        self.assertEqual(reranker.last_rerank_stats["input_candidates"], 0.0)

    def test_empty_retriever_rerank_clears_stats(self):
        reranker = FixedRetriever([], top_k=2)
        reranker.last_rerank_stats = {"rerank_time_s": 99.0}

        results = reranker.rerank("query", [])

        self.assertEqual(results, [])
        self.assertEqual(reranker.last_rerank_stats["rerank_time_s"], 0.0)
        self.assertEqual(reranker.last_rerank_stats["input_candidates"], 0.0)


if __name__ == "__main__":
    unittest.main()
