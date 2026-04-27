from __future__ import annotations

import os

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

import torch
from sentence_transformers import SparseEncoder

from ..._hf import init_model_with_hf_token
from ...base import BaseEmbedder


class SentenceTransformerSparseEmbedder(BaseEmbedder):
    """Sparse embedder backed by Sentence Transformers SparseEncoder."""

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
            SparseEncoder,
            hf_token=hf_token,
            model_name_or_path=model_name_or_path,
            device=device,
        )

    def encode_documents(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
    ) -> torch.Tensor:
        return self._encode_documents(texts)

    def encode_queries(
        self,
        texts: list[str] | str,
        normalize_embeddings: bool = True,
    ) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        return self._encode_queries(texts)

    def _encode_documents(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32).to_sparse_coo()

        embeddings = self.model.encode_document(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
        )
        return embeddings.coalesce()

    def _encode_queries(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32).to_sparse_coo()

        embeddings = self.model.encode_query(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
        )
        return embeddings.coalesce()
