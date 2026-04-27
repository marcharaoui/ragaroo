import unittest
from pathlib import Path
import tempfile

import numpy as np

from ragaroo.base import BaseEmbedder
from ragaroo.retrieval.dense.dense import DenseRetriever


class FakeEmbedder(BaseEmbedder):
    def __init__(self, *, embedding_dim=2):
        self.embedding_dim = embedding_dim
        self.encode_documents_calls = 0

    def encode_documents(self, texts, normalize_embeddings=True):
        self.encode_documents_calls += 1
        vectors = []
        for text in texts:
            if text == "alpha":
                vectors.append([1.0, 0.0])
            elif text == "beta":
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            if "alpha" in text:
                vectors.append([1.0, 0.0])
            elif "beta" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return np.asarray(vectors, dtype=np.float32)


class TestDenseRetriever(unittest.TestCase):
    def test_build_index_and_retrieve(self):
        retriever = DenseRetriever(embedder=FakeEmbedder(), top_k=2)
        corpus = {
            "doc1": {"text": "alpha", "metadata": {"source": "test"}},
            "doc2": {"text": "beta", "metadata": {"source": "test"}},
        }

        retriever.build_index(corpus)
        results = retriever.retrieve("alpha query")

        self.assertEqual(retriever.corpus_size, 2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].corpus_id, "doc1")
        self.assertEqual(results[0].text, "alpha")

    def test_ivf_index_builds_and_retrieves(self):
        retriever = DenseRetriever(
            embedder=FakeEmbedder(),
            top_k=2,
            index_technique="ivf",
            distance_metric="cosine",
            nlist=2,
            nprobe=2,
        )
        corpus = {
            "doc1": {"text": "alpha", "metadata": {"source": "test"}},
            "doc2": {"text": "beta", "metadata": {"source": "test"}},
            "doc3": {"text": "gamma", "metadata": {"source": "test"}},
        }

        retriever.build_index(corpus)
        results = retriever.retrieve("alpha query")

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].corpus_id, "doc1")
        self.assertEqual(retriever.vector_index.index_technique, "ivf")

    def test_reuses_cached_index(self):
        corpus = {
            "doc1": {"text": "alpha", "metadata": {"source": "test"}},
            "doc2": {"text": "beta", "metadata": {"source": "test"}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            first_embedder = FakeEmbedder()
            first_retriever = DenseRetriever(
                embedder=first_embedder,
                top_k=2,
                cache_dir=Path(tmpdir),
            )
            first_retriever.build_index(corpus)
            self.assertEqual(first_embedder.encode_documents_calls, 1)

            second_embedder = FakeEmbedder()
            second_retriever = DenseRetriever(
                embedder=second_embedder,
                top_k=2,
                cache_dir=Path(tmpdir),
            )
            second_retriever.build_index(corpus)

            self.assertEqual(second_embedder.encode_documents_calls, 0)
            self.assertEqual(second_retriever.last_build_stats["cache_hit"], 1.0)

    def test_batched_index_build_splits_document_encoding(self):
        corpus = {
            "doc1": {"text": "alpha", "metadata": {"source": "test"}},
            "doc2": {"text": "beta", "metadata": {"source": "test"}},
            "doc3": {"text": "gamma", "metadata": {"source": "test"}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = DenseRetriever(
                embedder=FakeEmbedder(),
                top_k=2,
                index_batch_size=1,
                cache_dir=Path(tmpdir),
            )
            retriever.build_index(corpus)
            results = retriever.retrieve("alpha query")

            self.assertEqual(retriever.embedder.encode_documents_calls, 3)
            self.assertEqual(results[0].corpus_id, "doc1")

    def test_runtime_tuning_settings_do_not_change_cache_key(self):
        corpus = {
            "doc1": {"text": "alpha", "metadata": {"source": "test"}},
            "doc2": {"text": "beta", "metadata": {"source": "test"}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            first_embedder = FakeEmbedder()
            first_retriever = DenseRetriever(
                embedder=first_embedder,
                top_k=2,
                cache_dir=Path(tmpdir),
                index_batch_size=1,
                faiss_threads=1,
            )
            first_retriever.build_index(corpus)

            second_embedder = FakeEmbedder()
            second_retriever = DenseRetriever(
                embedder=second_embedder,
                top_k=2,
                cache_dir=Path(tmpdir),
                index_batch_size=8,
                faiss_threads=2,
            )
            second_retriever.build_index(corpus)

            self.assertEqual(second_embedder.encode_documents_calls, 0)
            self.assertEqual(second_retriever.last_build_stats["cache_hit"], 1.0)

    def test_dimension_mismatch_raises_clear_error(self):
        corpus = {
            "doc1": {"text": "alpha", "metadata": {"source": "test"}},
            "doc2": {"text": "beta", "metadata": {"source": "test"}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = DenseRetriever(
                embedder=FakeEmbedder(embedding_dim=3),
                top_k=2,
                cache_dir=Path(tmpdir),
            )

            with self.assertRaisesRegex(ValueError, "Expected dense embedding dimension 3, got 2"):
                retriever.build_index(corpus)


if __name__ == "__main__":
    unittest.main()
