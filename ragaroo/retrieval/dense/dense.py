from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import numpy as np

from ...base import BaseEmbedder, BaseRetriever, BaseVectorIndex
from ..cache import cache_root, corpus_hash, load_metadata, save_metadata
from ..types import RetrievedDocument


def _resolve_faiss_threads(num_threads: int | str | None) -> int | None:
    if isinstance(num_threads, str):
        if num_threads != "auto":
            raise ValueError("faiss_threads must be an integer, None, or 'auto'")
        num_threads = os.cpu_count() or 1

    if num_threads is None:
        return None
    if num_threads <= 0:
        raise ValueError("faiss_threads must be > 0")
    return num_threads


def _as_2d_float_array(vectors: Any, embedding_dim: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Dense embeddings must be a 2D numpy array")
    if array.shape[1] != embedding_dim:
        raise ValueError(
            f"Expected dense embedding dimension {embedding_dim}, got {array.shape[1]}"
        )
    return array


class FaissVectorIndex(BaseVectorIndex):
    """Thin FAISS wrapper for exact, HNSW, and IVF dense indexes."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        distance_metric: str = "cosine",
        index_technique: str = "flat",
        hnsw_M: int = 32,
        efConstruction: int = 200,
        nlist: int = 256,
        nprobe: int = 10,
        faiss_threads: int | str | None = "auto",
    ) -> None:
        if distance_metric not in {"cosine", "dot", "euclidean"}:
            raise ValueError("distance_metric must be one of ['cosine', 'dot', 'euclidean']")
        if index_technique not in {"flat", "hnsw", "ivf"}:
            raise ValueError("index_technique must be one of ['flat', 'hnsw', 'ivf']")

        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        self.index_technique = index_technique
        self.hnsw_M = hnsw_M
        self.efConstruction = efConstruction
        self.nlist = nlist
        self.nprobe = nprobe
        self.normalize_embeddings = distance_metric == "cosine"
        self.faiss_threads = _resolve_faiss_threads(faiss_threads)
        self.actual_nlist: int | None = None
        self._apply_thread_setting()
        self.index = self._create_index()

    def build(self, embeddings: Any) -> None:
        vectors = _as_2d_float_array(embeddings, self.embedding_dim)
        self.initialize(vectors.shape[0])
        self.train(vectors)
        self.add(vectors)

    def search(self, query_embeddings: Any, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        vectors = _as_2d_float_array(query_embeddings, self.embedding_dim)
        self._apply_thread_setting()
        return self.index.search(vectors, top_k)

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)

    def load(self, path: str) -> None:
        self.index = faiss.read_index(path)
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.actual_nlist = int(self.index.nlist)
            self.index.nprobe = min(self.nprobe, self.actual_nlist)

    def initialize(self, corpus_size: int) -> None:
        self._apply_thread_setting()
        self.index = self._create_index(corpus_size=corpus_size)

    def train(self, embeddings: Any) -> None:
        vectors = _as_2d_float_array(embeddings, self.embedding_dim)
        self._apply_thread_setting()
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.train(vectors)
            self.index.nprobe = min(self.nprobe, self.actual_nlist or self.nlist)

    def add(self, embeddings: Any) -> None:
        vectors = _as_2d_float_array(embeddings, self.embedding_dim)
        self._apply_thread_setting()
        self.index.add(vectors)

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "index_technique": self.index_technique,
            "distance_metric": self.distance_metric,
            "normalize_embeddings": self.normalize_embeddings,
            "hnsw_M": self.hnsw_M,
            "efConstruction": self.efConstruction,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "actual_nlist": self.actual_nlist,
            "faiss_threads": self.faiss_threads,
        }

    def _create_index(self, corpus_size: int | None = None):
        if self.index_technique == "flat":
            return self._flat_index()
        if self.index_technique == "hnsw":
            return self._hnsw_index()
        return self._ivf_index(corpus_size or self.nlist)

    def _flat_index(self):
        if self.distance_metric == "euclidean":
            return faiss.IndexFlatL2(self.embedding_dim)
        return faiss.IndexFlatIP(self.embedding_dim)

    def _hnsw_index(self):
        metric = faiss.METRIC_L2 if self.distance_metric == "euclidean" else faiss.METRIC_INNER_PRODUCT
        index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_M, metric)
        index.hnsw.efConstruction = self.efConstruction
        return index

    def _ivf_index(self, corpus_size: int):
        self.actual_nlist = max(1, min(self.nlist, corpus_size))
        quantizer = self._flat_index()
        metric = faiss.METRIC_L2 if self.distance_metric == "euclidean" else faiss.METRIC_INNER_PRODUCT
        index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.actual_nlist, metric)
        index.nprobe = min(self.nprobe, self.actual_nlist)
        return index

    def training_sample_size(self, corpus_size: int) -> int:
        return min(corpus_size, max((self.actual_nlist or self.nlist or 1) * 39, 1000))

    def _apply_thread_setting(self) -> None:
        if self.faiss_threads is not None:
            faiss.omp_set_num_threads(self.faiss_threads)


class DenseRetriever(BaseRetriever):
    """Retrieve documents by embedding text and searching a FAISS index."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        top_k: int = 10,
        cache_dir: str | Path = "indexes",
        distance_metric: str = "cosine",
        index_technique: str = "flat",
        hnsw_M: int = 32,
        efConstruction: int = 200,
        nlist: int = 256,
        nprobe: int = 10,
        index_batch_size: int = 2048,
        faiss_threads: int | str | None = "auto",
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if index_batch_size <= 0:
            raise ValueError("index_batch_size must be > 0")

        embedding_dim = getattr(embedder, "embedding_dim", None)
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("Dense embedder must expose a positive integer 'embedding_dim'")

        self.embedder = embedder
        self.top_k = top_k
        self.cache_dir = Path(cache_dir)
        self.index_batch_size = index_batch_size
        self.vector_index = FaissVectorIndex(
            embedding_dim,
            distance_metric=distance_metric,
            index_technique=index_technique,
            hnsw_M=hnsw_M,
            efConstruction=efConstruction,
            nlist=nlist,
            nprobe=nprobe,
            faiss_threads=faiss_threads,
        )

        self._corpus: dict[str, dict[str, Any]] = {}
        self._corpus_ids: list[str] = []
        self.last_build_stats: dict[str, float] = {}
        self.last_query_stats: dict[str, float] = {}
        self._is_built = False
        self._built_cache_path: Path | None = None

    def build_index(self, corpus: dict[str, dict[str, Any]]) -> None:
        if not corpus:
            raise ValueError("Corpus is empty")

        corpus_ids: list[str] = []
        for corpus_id, item in corpus.items():
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Corpus item '{corpus_id}' is missing a valid 'text' field")
            corpus_ids.append(corpus_id)

        self._corpus = corpus
        self._corpus_ids = corpus_ids

        cache_path = self._cache_path(corpus)
        if self._is_built and self._built_cache_path == cache_path:
            return

        metadata_path = cache_path / "metadata.json"
        index_path = cache_path / "index.faiss"

        if metadata_path.exists() and index_path.exists():
            load_started_at = perf_counter()
            metadata = load_metadata(metadata_path)
            self.vector_index.load(str(index_path))
            self._corpus_ids = list(metadata["corpus_ids"])
            load_finished_at = perf_counter()
            self.last_build_stats = {
                "embedding_time_s": 0.0,
                "index_build_time_s": 0.0,
                "load_time_s": load_finished_at - load_started_at,
                "total_build_time_s": load_finished_at - load_started_at,
                "corpus_size": float(len(self._corpus_ids)),
                "embedding_dim": float(metadata["embedding_dim"]),
                "cache_hit": 1.0,
            }
            self._is_built = True
            self._built_cache_path = cache_path
            return

        build_started_at = perf_counter()
        total_embedding_time_s = 0.0
        index_started_at = perf_counter()
        self.vector_index.initialize(len(corpus_ids))

        next_offset = 0
        if self.vector_index.index_technique == "ivf":
            sample_size = self.vector_index.training_sample_size(len(corpus_ids))
            sample_ids = corpus_ids[:sample_size]
            sample_texts = [self._corpus[corpus_id]["text"] for corpus_id in sample_ids]
            embedding_started_at = perf_counter()
            sample_embeddings = self.embedder.encode_documents(
                sample_texts,
                self.vector_index.normalize_embeddings,
            )
            total_embedding_time_s += perf_counter() - embedding_started_at
            self.vector_index.train(sample_embeddings)
            self.vector_index.add(sample_embeddings)
            next_offset = sample_size

        for start in range(next_offset, len(corpus_ids), self.index_batch_size):
            batch_ids = corpus_ids[start : start + self.index_batch_size]
            batch_texts = [self._corpus[corpus_id]["text"] for corpus_id in batch_ids]
            embedding_started_at = perf_counter()
            batch_embeddings = self.embedder.encode_documents(
                batch_texts,
                self.vector_index.normalize_embeddings,
            )
            total_embedding_time_s += perf_counter() - embedding_started_at
            self.vector_index.add(batch_embeddings)

        index_finished_at = perf_counter()

        cache_path.mkdir(parents=True, exist_ok=True)
        save_metadata(
            metadata_path,
            {
                "corpus_ids": corpus_ids,
                "embedding_dim": int(self.vector_index.embedding_dim),
                "vector_index": self.vector_index.config_dict(),
            },
        )
        self.vector_index.save(str(index_path))

        self.last_build_stats = {
            "embedding_time_s": total_embedding_time_s,
            "index_build_time_s": index_finished_at - index_started_at,
            "total_build_time_s": index_finished_at - build_started_at,
            "corpus_size": float(len(corpus_ids)),
            "embedding_dim": float(self.vector_index.embedding_dim),
            "cache_hit": 0.0,
        }
        self._is_built = True
        self._built_cache_path = cache_path

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDocument]:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if not self._corpus_ids:
            raise ValueError("Index not built. Call build_index() first.")

        k = top_k or self.top_k
        if k <= 0:
            raise ValueError("top_k must be > 0")

        embedding_started_at = perf_counter()
        query_embedding = self.embedder.encode_queries(
            [query],
            self.vector_index.normalize_embeddings,
        )
        embedding_finished_at = perf_counter()

        search_started_at = perf_counter()
        scores, indices = self.vector_index.search(
            query_embedding,
            min(k, len(self._corpus_ids)),
        )
        search_finished_at = perf_counter()

        results: list[RetrievedDocument] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            corpus_id = self._corpus_ids[index]
            item = self._corpus[corpus_id]
            results.append(
                RetrievedDocument(
                    corpus_id=corpus_id,
                    score=float(score),
                    text=item["text"],
                    metadata=item.get("metadata", {}),
                )
            )

        self.last_query_stats = {
            "query_embedding_time_s": embedding_finished_at - embedding_started_at,
            "search_time_s": search_finished_at - search_started_at,
            "total_query_time_s": search_finished_at - embedding_started_at,
            "top_k": float(k),
        }
        return results

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "top_k": self.top_k,
            "embedder": self.embedder.__class__.__name__,
            "model_name_or_path": getattr(self.embedder, "model_name_or_path", None),
            "index_batch_size": self.index_batch_size,
            "vector_index": self.vector_index.config_dict(),
        }

    def _cache_path(self, corpus: dict[str, dict[str, Any]]) -> Path:
        model_name = getattr(self.embedder, "model_name_or_path", self.embedder.__class__.__name__)
        config_signature = json.dumps(
            {
                "type": self.__class__.__name__,
                "embedder": self.embedder.__class__.__name__,
                "vector_index": {
                    "type": self.vector_index.__class__.__name__,
                    "index_technique": self.vector_index.index_technique,
                    "distance_metric": self.vector_index.distance_metric,
                    "normalize_embeddings": self.vector_index.normalize_embeddings,
                    "hnsw_M": self.vector_index.hnsw_M,
                    "efConstruction": self.vector_index.efConstruction,
                    "nlist": self.vector_index.nlist,
                    "nprobe": self.vector_index.nprobe,
                },
            },
            sort_keys=True,
        )
        return cache_root(
            self.cache_dir,
            "dense",
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
