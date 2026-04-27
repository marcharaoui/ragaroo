from __future__ import annotations

import os

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

from typing import Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from .._hf import init_model_with_hf_token


class SentenceTransformerCrossEncoder:
    """Sentence Transformers CrossEncoder wrapper with a small scoring API."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.model = init_model_with_hf_token(
            CrossEncoder,
            hf_token=hf_token,
            model_name_or_path=model_name_or_path,
            device=device,
        )

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        if not documents:
            return np.empty((0,), dtype=np.float32)

        pairs = [[query, document] for document in documents]
        scores = self.model.predict(
            pairs,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float32)
