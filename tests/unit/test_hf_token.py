import os
import unittest
from unittest.mock import patch

from ragaroo.reranking.st_reranker import SentenceTransformerCrossEncoder
from ragaroo.retrieval.dense.st_embedder import SentenceTransformerEmbedder
from ragaroo.retrieval.sparse.st_sparse_embedder import SentenceTransformerSparseEmbedder


class FakeSentenceTransformer:
    def __init__(self, model_name_or_path, device=None, token=None):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.token = token

    def get_sentence_embedding_dimension(self):
        return 8


class FakeSparseEncoder:
    def __init__(self, model_name_or_path, device=None, token=None):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.token = token


class FakeCrossEncoder:
    def __init__(self, model_name_or_path, device=None, token=None):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.token = token


class TestHFTokenSupport(unittest.TestCase):
    def test_dense_embedder_forwards_explicit_hf_token(self):
        with patch("ragaroo.retrieval.dense.st_embedder.SentenceTransformer", FakeSentenceTransformer):
            embedder = SentenceTransformerEmbedder("model/name", hf_token="secret-token")

        self.assertEqual(embedder.model.token, "secret-token")

    def test_dense_embedder_uses_env_hf_token(self):
        with patch.dict(os.environ, {"HF_TOKEN": "env-token"}, clear=False):
            with patch("ragaroo.retrieval.dense.st_embedder.SentenceTransformer", FakeSentenceTransformer):
                embedder = SentenceTransformerEmbedder("model/name")

        self.assertEqual(embedder.model.token, "env-token")

    def test_sparse_embedder_forwards_explicit_hf_token(self):
        with patch("ragaroo.retrieval.sparse.st_sparse_embedder.SparseEncoder", FakeSparseEncoder):
            embedder = SentenceTransformerSparseEmbedder("model/name", hf_token="secret-token")

        self.assertEqual(embedder.model.token, "secret-token")

    def test_cross_encoder_forwards_explicit_hf_token(self):
        with patch("ragaroo.reranking.st_reranker.CrossEncoder", FakeCrossEncoder):
            reranker_model = SentenceTransformerCrossEncoder("model/name", hf_token="secret-token")

        self.assertEqual(reranker_model.model.token, "secret-token")


if __name__ == "__main__":
    unittest.main()
