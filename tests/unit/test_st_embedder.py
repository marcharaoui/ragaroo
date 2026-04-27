import os
import unittest
from pathlib import Path

import ragaroo as rr
from ragaroo.retrieval.dense.st_embedder import SentenceTransformerEmbedder


rr.store_models("./models")


@unittest.skipUnless(
    Path("models").exists(),
    "Sentence Transformers cache directory is not available locally.",
)
class TestSTEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = SentenceTransformerEmbedder(model_name_or_path="intfloat/e5-small-v2")

    def test_init(self):
        self.assertIsNotNone(self.embedder)

    def test_encode_documents(self):
        documents = ["This is a test document.", "Another test document."]
        embeddings = self.embedder.encode_documents(documents)
        self.assertEqual(embeddings.shape[0], len(documents))
        self.assertEqual(embeddings.shape[1], self.embedder.embedding_dim)

    def test_encode_queries(self):
        queries = ["This is a test query.", "Another test query."]
        embeddings = self.embedder.encode_queries(queries)
        self.assertEqual(embeddings.shape[0], len(queries))
        self.assertEqual(embeddings.shape[1], self.embedder.embedding_dim)


if __name__ == "__main__":
    unittest.main()
