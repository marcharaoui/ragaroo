from __future__ import annotations

import os

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

from ..._hf import init_model_with_hf_token
from ...base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Dense embedder backed by any Sentence Transformers-compatible model."""

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 32,
        device: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.model = init_model_with_hf_token(
            SentenceTransformer,
            hf_token=hf_token,
            model_name_or_path=model_name_or_path,
            device=device,
        )

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        if self.embedding_dim is None:
            raise ValueError(
                f"Could not infer embedding dimension for model '{model_name_or_path}'."
            )

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
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )

        return np.asarray(embeddings, dtype=np.float32)

