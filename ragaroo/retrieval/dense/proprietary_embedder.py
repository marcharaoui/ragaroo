from __future__ import annotations

import json
from urllib import request

import numpy as np

from ...base import BaseEmbedder


class ProprietaryEmbedder(BaseEmbedder):
    """Example embedder adapter for users integrating proprietary embedding APIs.

    This class is intentionally simple and uses OpenRouter's embedding endpoint
    as a concrete example. Users can copy this file and adapt the request/response
    handling for their own provider.
    """

    def __init__(
        self,
        api_key: str | None,
        model: str = "openai/text-embedding-3-small",
        *,
        embedding_dim: int,
        base_url: str = "https://openrouter.ai/api/v1/embeddings",
        batch_size: int = 32,
        timeout_s: float = 30.0,
        referer: str | None = None,
        title: str | None = None,
    ) -> None:
        if api_key is None:
            raise ValueError("api_key is required for ProprietaryEmbedder")
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")

        cleaned_api_key = api_key.strip()
        if not cleaned_api_key:
            raise ValueError("api_key is required for ProprietaryEmbedder")

        self.api_key = cleaned_api_key
        self.model = model
        self.model_name_or_path = model
        self.base_url = base_url
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.referer = referer
        self.title = title
        self.embedding_dim: int = embedding_dim

    def encode_documents(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        return self._encode(texts, normalize_embeddings=normalize_embeddings)

    def encode_queries(
        self,
        texts: list[str] | str,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self._encode(texts, normalize_embeddings=normalize_embeddings)

    def _encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title

        req = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_s) as response:
            raw_body = response.read().decode("utf-8")

        body = json.loads(raw_body)
        rows = body.get("data", [])
        if not rows:
            raise ValueError("Embedding response does not contain any vectors")

        embeddings = np.asarray(
            [row["embedding"] for row in rows],
            dtype=np.float32,
        )
        if embeddings.ndim != 2:
            raise ValueError("Embedding response has invalid shape")

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms

        return embeddings
