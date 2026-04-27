import unittest
from pathlib import Path
import tempfile

import torch

from ragaroo.base import BaseEmbedder
from ragaroo.retrieval.sparse import SparseRetriever


class FakeSparseEmbedder(BaseEmbedder):
    def __init__(self):
        self.encode_documents_calls = 0

    def encode_documents(self, texts, normalize_embeddings=True):
        self.encode_documents_calls += 1
        vectors = []
        for text in texts:
            if "alpha" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "beta" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        dense = torch.tensor(vectors, dtype=torch.float32)
        return dense.to_sparse_coo()

    def encode_queries(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            if "alpha" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "beta" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        dense = torch.tensor(vectors, dtype=torch.float32)
        return dense.to_sparse_coo()


class TestSparseRetriever(unittest.TestCase):
    def test_build_index_and_retrieve(self):
        retriever = SparseRetriever(embedder=FakeSparseEmbedder(), top_k=2)
        corpus = {
            "doc1": {"text": "alpha topic", "metadata": {"source": "test"}},
            "doc2": {"text": "beta topic", "metadata": {"source": "test"}},
        }

        retriever.build_index(corpus)
        results = retriever.retrieve("alpha query")

        self.assertEqual(retriever.corpus_size, 2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].corpus_id, "doc1")
        self.assertEqual(results[0].text, "alpha topic")

    def test_reuses_cached_sparse_index(self):
        corpus = {
            "doc1": {"text": "alpha topic", "metadata": {"source": "test"}},
            "doc2": {"text": "beta topic", "metadata": {"source": "test"}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            first_embedder = FakeSparseEmbedder()
            first_retriever = SparseRetriever(
                embedder=first_embedder,
                top_k=2,
                cache_dir=Path(tmpdir),
            )
            first_retriever.build_index(corpus)
            self.assertEqual(first_embedder.encode_documents_calls, 1)

            second_embedder = FakeSparseEmbedder()
            second_retriever = SparseRetriever(
                embedder=second_embedder,
                top_k=2,
                cache_dir=Path(tmpdir),
            )
            second_retriever.build_index(corpus)

            self.assertEqual(second_embedder.encode_documents_calls, 0)
            self.assertEqual(second_retriever.last_build_stats["cache_hit"], 1.0)


if __name__ == "__main__":
    unittest.main()
