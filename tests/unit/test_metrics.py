import unittest

from ragaroo.evaluation.metrics import (
    hit_rate_at_k,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ragaroo.retrieval.types import RetrievedDocument


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.results = [
            RetrievedDocument(corpus_id="d1", score=0.9, text="doc 1"),
            RetrievedDocument(corpus_id="d2", score=0.8, text="doc 2"),
            RetrievedDocument(corpus_id="d3", score=0.7, text="doc 3"),
        ]
        self.relevant_ids = {"d2", "d3"}
        self.graded_qrels = {"d2": 2.0, "d3": 1.0}

    def test_recall_at_k(self):
        self.assertEqual(recall_at_k(self.results, self.relevant_ids), 1.0)

    def test_precision_at_k(self):
        self.assertAlmostEqual(precision_at_k(self.results, self.relevant_ids, 3), 2 / 3)

    def test_mrr_at_k(self):
        self.assertAlmostEqual(mrr_at_k(self.results, self.relevant_ids), 0.5)

    def test_map_at_k(self):
        self.assertAlmostEqual(map_at_k(self.results, self.relevant_ids), (0.5 + (2 / 3)) / 2)

    def test_hit_rate_at_k(self):
        self.assertEqual(hit_rate_at_k(self.results, self.relevant_ids), 1.0)

    def test_ndcg_at_k(self):
        score = ndcg_at_k(self.results, self.graded_qrels, 3)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
