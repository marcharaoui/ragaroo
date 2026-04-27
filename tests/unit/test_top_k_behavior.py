import unittest

import numpy as np
import torch

from ragaroo.base import BaseEmbedder, BaseRetriever
from ragaroo.pipeline.pipeline import Pipeline
from ragaroo.retrieval.dense.dense import DenseRetriever
from ragaroo.retrieval.hybrid import HybridRetriever
from ragaroo.retrieval.lexical.bm25 import BM25Retriever
from ragaroo.retrieval.sparse import SparseRetriever
from ragaroo.retrieval.types import RetrievedDocument
from ragaroo.reranking.cross_encoder import CrossEncoderReranker


class DenseLikeEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        self.embedding_dim = 2

    def encode_documents(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            if "alpha" in text:
                vectors.append([1.0, 0.0])
            elif "beta" in text:
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


class SparseLikeEmbedder(BaseEmbedder):
    def encode_documents(self, texts, normalize_embeddings=True):
        dense = torch.tensor(
            [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts],
            dtype=torch.float32,
        )
        return dense.to_sparse_coo()

    def encode_queries(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        dense = torch.tensor(
            [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts],
            dtype=torch.float32,
        )
        return dense.to_sparse_coo()


class FixedRetriever(BaseRetriever):
    def __init__(self, results):
        self.results = results
        self.last_build_stats = {}
        self.last_query_stats = {}

    def build_index(self, corpus):
        self.last_build_stats = {"total_build_time_s": 0.0}

    def retrieve(self, query, top_k=None):
        k = top_k or len(self.results)
        self.last_query_stats = {"total_query_time_s": 0.0}
        return self.results[:k]

    def config_dict(self):
        return {"type": "FixedRetriever"}


class FakeCrossEncoderModel:
    model_name_or_path = "fake-cross-encoder"

    def score(self, query, documents):
        return np.asarray([float(len(doc)) for doc in documents], dtype=np.float32)


class TestTopKBehavior(unittest.TestCase):
    def setUp(self):
        self.corpus = {
            "doc1": {"text": "alpha one", "metadata": {}},
            "doc2": {"text": "alpha two", "metadata": {}},
            "doc3": {"text": "beta three", "metadata": {}},
        }

    def test_dense_retriever_respects_top_k(self):
        retriever = DenseRetriever(embedder=DenseLikeEmbedder(), top_k=2)
        retriever.build_index(self.corpus)

        results = retriever.retrieve("alpha")

        self.assertEqual(len(results), 2)

    def test_bm25_retriever_respects_top_k(self):
        retriever = BM25Retriever(top_k=2)
        retriever.build_index(self.corpus)

        results = retriever.retrieve("alpha")

        self.assertLessEqual(len(results), 2)

    def test_sparse_retriever_respects_top_k(self):
        retriever = SparseRetriever(embedder=SparseLikeEmbedder(), top_k=2)
        retriever.build_index(self.corpus)

        results = retriever.retrieve("alpha")

        self.assertEqual(len(results), 2)

    def test_hybrid_retriever_respects_top_k(self):
        retriever_1 = FixedRetriever(
            [
                RetrievedDocument("doc1", 1.0, "alpha one"),
                RetrievedDocument("doc2", 0.9, "alpha two"),
                RetrievedDocument("doc3", 0.8, "beta three"),
            ]
        )
        retriever_2 = FixedRetriever(
            [
                RetrievedDocument("doc2", 1.0, "alpha two"),
                RetrievedDocument("doc3", 0.9, "beta three"),
                RetrievedDocument("doc1", 0.8, "alpha one"),
            ]
        )
        hybrid = HybridRetriever(retriever_1, retriever_2, top_k=2, fusion_technique="rrf")
        hybrid.build_index(self.corpus)

        results = hybrid.retrieve("alpha")

        self.assertEqual(len(results), 2)

    def test_reranker_respects_top_k(self):
        pipeline = Pipeline(
            name="dense_plus_reranker",
            retriever=DenseRetriever(embedder=DenseLikeEmbedder(), top_k=3),
            reranker=CrossEncoderReranker(model=FakeCrossEncoderModel(), top_k=1),
        )
        pipeline.prepare(self.corpus)

        results = pipeline.retrieve("alpha")

        self.assertEqual(len(results), 1)
