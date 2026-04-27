from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from ...base import BaseEmbedder, BaseRetriever
from ..cache import cache_root, corpus_hash, load_metadata, save_metadata
from ..types import RetrievedDocument

if hasattr(torch.sparse, "check_sparse_tensor_invariants"):
    torch.sparse.check_sparse_tensor_invariants.disable()


class SparseRetriever(BaseRetriever):
    """Sparse-vector retriever backed by Sentence Transformers sparse encoders."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        top_k: int = 10,
        cache_dir: str | Path = "indexes",
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        self.embedder = embedder
        self.top_k = top_k
        self.cache_dir = Path(cache_dir)
        self._corpus: dict[str, dict[str, Any]] = {}
        self._corpus_ids: list[str] = []
        self._corpus_embeddings: torch.Tensor | None = None
        self.last_build_stats: dict[str, float] = {}
        self.last_query_stats: dict[str, float] = {}
        self._is_built = False
        self._built_cache_path: Path | None = None

    def build_index(self, corpus: dict[str, dict[str, Any]]) -> None:
        if not corpus:
            raise ValueError("Corpus is empty")

        corpus_ids: list[str] = []
        corpus_texts: list[str] = []

        for corpus_id, item in corpus.items():
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Corpus item '{corpus_id}' is missing a valid 'text' field")

            corpus_ids.append(corpus_id)
            corpus_texts.append(text)

        self._corpus = corpus
        self._corpus_ids = corpus_ids

        cache_path = self._cache_path(corpus)
        if self._is_built and self._built_cache_path == cache_path:
            return
        metadata_path = cache_path / "metadata.json"
        tensor_path = cache_path / "embeddings.pt"

        if metadata_path.exists() and tensor_path.exists():
            t0 = perf_counter()
            metadata = load_metadata(metadata_path)
            loaded_embeddings = torch.load(tensor_path, map_location="cpu", weights_only=True)
            self._corpus_ids = list(metadata["corpus_ids"])
            self._corpus_embeddings = loaded_embeddings.coalesce() if loaded_embeddings.is_sparse else loaded_embeddings.to_sparse_coo().coalesce()
            t1 = perf_counter()
            self.last_build_stats = {
                "embedding_time_s": 0.0,
                "index_build_time_s": 0.0,
                "load_time_s": t1 - t0,
                "total_build_time_s": t1 - t0,
                "corpus_size": float(len(self._corpus_ids)),
                "embedding_dim": float(metadata["embedding_dim"]),
                "cache_hit": 1.0,
            }
            self._is_built = True
            self._built_cache_path = cache_path
            return

        t0 = perf_counter()
        corpus_embeddings = self.embedder.encode_documents(corpus_texts, normalize_embeddings=False)
        t1 = perf_counter()

        if not isinstance(corpus_embeddings, torch.Tensor):
            raise ValueError("Sparse embedder must return a torch.Tensor")
        if corpus_embeddings.ndim != 2:
            raise ValueError("Sparse document embeddings must be a 2D tensor")

        self._corpus_embeddings = corpus_embeddings.coalesce() if corpus_embeddings.is_sparse else corpus_embeddings.to_sparse_coo().coalesce()
        t2 = perf_counter()

        cache_path.mkdir(parents=True, exist_ok=True)
        torch.save(self._corpus_embeddings, tensor_path)
        save_metadata(
            metadata_path,
            {
                "corpus_ids": corpus_ids,
                "embedding_dim": int(corpus_embeddings.shape[1]),
            },
        )

        self.last_build_stats = {
            "embedding_time_s": t1 - t0,
            "index_build_time_s": t2 - t1,
            "total_build_time_s": t2 - t0,
            "corpus_size": float(len(corpus_ids)),
            "embedding_dim": float(corpus_embeddings.shape[1]),
            "cache_hit": 0.0,
        }
        self._is_built = True
        self._built_cache_path = cache_path

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDocument]:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if self._corpus_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        k = top_k or self.top_k
        if k <= 0:
            raise ValueError("top_k must be > 0")

        t0 = perf_counter()
        query_embedding = self.embedder.encode_queries([query], normalize_embeddings=False)
        t1 = perf_counter()

        if not isinstance(query_embedding, torch.Tensor):
            raise ValueError("Sparse embedder must return a torch.Tensor")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)

        dense_query = query_embedding.to_dense() if query_embedding.is_sparse else query_embedding
        dense_scores = torch.sparse.mm(self._corpus_embeddings, dense_query.t()).squeeze(1)
        top_scores, top_indices = torch.topk(dense_scores, k=min(k, len(self._corpus_ids)))
        t2 = perf_counter()

        results: list[RetrievedDocument] = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            corpus_id = self._corpus_ids[idx]
            corpus_item = self._corpus[corpus_id]
            results.append(
                RetrievedDocument(
                    corpus_id=corpus_id,
                    score=float(score),
                    text=corpus_item["text"],
                    metadata=corpus_item.get("metadata", {}),
                )
            )

        self.last_query_stats = {
            "query_embedding_time_s": t1 - t0,
            "search_time_s": t2 - t1,
            "total_query_time_s": t2 - t0,
            "top_k": float(k),
        }

        return results

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "top_k": self.top_k,
            "embedder": self.embedder.__class__.__name__,
            "model_name_or_path": getattr(self.embedder, "model_name_or_path", None),
        }

    def _cache_path(self, corpus: dict[str, dict[str, Any]]) -> Path:
        model_name = getattr(self.embedder, "model_name_or_path", self.embedder.__class__.__name__)
        config_signature = json.dumps(
            {
                "type": self.__class__.__name__,
                "embedder": self.embedder.__class__.__name__,
            },
            sort_keys=True,
        )
        return cache_root(
            self.cache_dir,
            "sparse",
            model_name,
            corpus_hash(corpus),
            config_signature,
        )

    @property
    def corpus_size(self) -> int:
        return len(self._corpus_ids)

    @property
    def is_built(self) -> bool:
        return self._is_built
