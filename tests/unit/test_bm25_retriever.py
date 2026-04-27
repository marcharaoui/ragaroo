import unittest
from pathlib import Path
import tempfile

from ragaroo.retrieval.lexical.bm25 import BM25Retriever


class TestBM25Retriever(unittest.TestCase):
    def test_build_index_and_retrieve(self):
        retriever = BM25Retriever(top_k=2)
        corpus = {
            "doc1": {"text": "alpha beta", "metadata": {"source": "test"}},
            "doc2": {"text": "gamma delta", "metadata": {"source": "test"}},
            "doc3": {"text": "alpha gamma", "metadata": {"source": "test"}},
        }

        retriever.build_index(corpus)
        results = retriever.retrieve("alpha")

        self.assertEqual(retriever.corpus_size, 3)
        self.assertGreater(len(results), 0)
        self.assertIn(results[0].corpus_id, {"doc1", "doc3"})
        self.assertIn("total_build_time_s", retriever.last_build_stats)
        self.assertIn("total_query_time_s", retriever.last_query_stats)

    def test_reuses_cached_bm25_index(self):
        corpus = {
            "doc1": {"text": "alpha beta", "metadata": {"source": "test"}},
            "doc2": {"text": "gamma delta", "metadata": {"source": "test"}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            first_retriever = BM25Retriever(top_k=2, cache_dir=Path(tmpdir))
            first_retriever.build_index(corpus)
            self.assertEqual(first_retriever.last_build_stats["cache_hit"], 0.0)

            second_retriever = BM25Retriever(top_k=2, cache_dir=Path(tmpdir))
            second_retriever.build_index(corpus)

            self.assertEqual(second_retriever.last_build_stats["cache_hit"], 1.0)

    def test_cache_signature_includes_tokenization_options(self):
        corpus = {
            "doc1": {"text": "alpha beta", "metadata": {"source": "test"}},
            "doc2": {"text": "gamma delta", "metadata": {"source": "test"}},
        }
        default_retriever = BM25Retriever()
        custom_retriever = BM25Retriever(stopwords=[])

        self.assertNotEqual(
            default_retriever._cache_path(corpus),
            custom_retriever._cache_path(corpus),
        )
        self.assertEqual(custom_retriever.config_dict()["stopwords"], [])


if __name__ == "__main__":
    unittest.main()
