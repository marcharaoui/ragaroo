import json
import unittest
from unittest.mock import patch

import numpy as np

from ragaroo.retrieval.dense import ProprietaryEmbedder


class TestProprietaryEmbedder(unittest.TestCase):
    def test_missing_api_key_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "api_key is required"):
            ProprietaryEmbedder(api_key=None, model="demo-model", embedding_dim=3)

        with self.assertRaisesRegex(ValueError, "api_key is required"):
            ProprietaryEmbedder(api_key="  ", model="demo-model", embedding_dim=3)

    def test_encode_documents_parses_embedding_response(self):
        payload = {
            "data": [
                {"embedding": [1.0, 0.0, 0.5]},
                {"embedding": [0.0, 1.0, 0.5]},
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with patch("ragaroo.retrieval.dense.proprietary_embedder.request.urlopen", return_value=FakeResponse()):
            embedder = ProprietaryEmbedder(api_key="test-key", model="demo-model", embedding_dim=3)
            embeddings = embedder.encode_documents(["a", "b"])

        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (2, 3))
        self.assertEqual(embedder.embedding_dim, 3)
