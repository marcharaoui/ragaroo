import unittest

from ragaroo.base import BaseRetriever
from ragaroo.retrieval.hybrid import HybridRetriever
from ragaroo.retrieval.types import RetrievedDocument


class DummyRetriever(BaseRetriever):
    def __init__(self, results):
        self.results = results
        self.last_build_stats = {}
        self.last_query_stats = {"total_query_time_s": 0.01}

    def build_index(self, corpus):
        self.corpus = corpus
        self.last_build_stats = {"total_build_time_s": 0.01}

    def retrieve(self, query, top_k=None):
        k = top_k or len(self.results)
        return self.results[:k]

    def config_dict(self):
        return {"type": "DummyRetriever"}


class TestHybridRetriever(unittest.TestCase):
    def test_rrf_fusion_combines_rankings(self):
        retriever_1 = DummyRetriever(
            [
                RetrievedDocument("d1", 0.9, "doc 1"),
                RetrievedDocument("d2", 0.8, "doc 2"),
            ]
        )
        retriever_2 = DummyRetriever(
            [
                RetrievedDocument("d2", 0.95, "doc 2"),
                RetrievedDocument("d3", 0.7, "doc 3"),
            ]
        )
        hybrid = HybridRetriever(retriever_1, retriever_2, top_k=3, fusion_technique="rrf", rrf_k=10)
        hybrid.build_index({"d1": {"text": "doc 1"}, "d2": {"text": "doc 2"}, "d3": {"text": "doc 3"}})

        results = hybrid.retrieve("query")

        self.assertEqual(results[0].corpus_id, "d2")
        self.assertEqual(len(results), 3)
        self.assertIn("fusion_time_s", hybrid.last_query_stats)
        self.assertGreaterEqual(hybrid.last_query_stats["total_query_time_s"], hybrid.last_query_stats["fusion_time_s"])

    def test_average_fusion_combines_scores(self):
        retriever_1 = DummyRetriever(
            [
                RetrievedDocument("d1", 10.0, "doc 1"),
                RetrievedDocument("d2", 9.0, "doc 2"),
                RetrievedDocument("d3", 0.0, "doc 3"),
            ]
        )
        retriever_2 = DummyRetriever(
            [
                RetrievedDocument("d2", 10.0, "doc 2"),
                RetrievedDocument("d3", 1.0, "doc 3"),
            ]
        )
        hybrid = HybridRetriever(retriever_1, retriever_2, top_k=3, fusion_technique="average")
        hybrid.build_index({"d1": {"text": "doc 1"}, "d2": {"text": "doc 2"}, "d3": {"text": "doc 3"}})

        results = hybrid.retrieve("query")

        self.assertEqual(results[0].corpus_id, "d2")


if __name__ == "__main__":
    unittest.main()
