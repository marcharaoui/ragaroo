from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from ...base import BaseRetriever
from ..cache import cache_root, corpus_hash, load_metadata, save_metadata
from ..types import RetrievedDocument
from .lexical import BM25SLexicalSearch


class BM25Retriever(BaseRetriever):
    """BM25 retriever backed by bm25s with local index caching."""

    def __init__(
        self,
        top_k: int = 10,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: str | list[str] = "english",
        stemmer: Any = None,
        cache_dir: str | Path = "indexes",
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        self.top_k = top_k
        self.cache_dir = Path(cache_dir)
        self.lexical_search = BM25SLexicalSearch(
            k1=k1,
            b=b,
            stopwords=stopwords,
            stemmer=stemmer,
        )

        self._corpus: dict[str, dict[str, Any]] = {}
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []

        self.last_build_stats: dict[str, float] = {}
        self.last_query_stats: dict[str, float] = {}
        self._is_built = False
        self._built_signature: str | None = None
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
        build_signature = self._build_signature(corpus)
        cache_path = self._cache_path(corpus)
        metadata_path = cache_path / "metadata.json"
        if self._is_built and self._built_signature == build_signature and self._built_cache_path == cache_path:
            return

        if metadata_path.exists():
            t0 = perf_counter()
            metadata = load_metadata(metadata_path)
            self.lexical_search.load(cache_path)
            self._corpus_ids = list(metadata["corpus_ids"])
            t1 = perf_counter()
            self.last_build_stats = {
                "index_build_time_s": 0.0,
                "load_time_s": t1 - t0,
                "total_build_time_s": t1 - t0,
                "corpus_size": float(len(self._corpus_ids)),
                "cache_hit": 1.0,
            }
            self._is_built = True
            self._built_signature = build_signature
            self._built_cache_path = cache_path
            return

        if self._is_built and self._built_signature == build_signature:
            return

        t0 = perf_counter()
        self.lexical_search.build_index(corpus_texts)
        t1 = perf_counter()
        cache_path.mkdir(parents=True, exist_ok=True)
        self.lexical_search.save(cache_path)
        save_metadata(
            metadata_path,
            {
                "corpus_ids": corpus_ids,
                "corpus_size": len(corpus_ids),
            },
        )

        self.last_build_stats = {
            "index_build_time_s": t1 - t0,
            "total_build_time_s": t1 - t0,
            "corpus_size": float(len(corpus_ids)),
            "cache_hit": 0.0,
        }
        self._is_built = True
        self._built_signature = build_signature
        self._built_cache_path = cache_path

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDocument]:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if not self._corpus_ids:
            raise ValueError("Index not built. Call build_index() first.")

        k = top_k or self.top_k
        if k <= 0:
            raise ValueError("top_k must be > 0")

        t0 = perf_counter()
        document_ids, scores = self.lexical_search.search(query, self._corpus_ids, k)
        t1 = perf_counter()

        results: list[RetrievedDocument] = []
        for corpus_id, score in zip(document_ids, scores):
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
            "search_time_s": t1 - t0,
            "total_query_time_s": t1 - t0,
            "top_k": float(k),
        }

        return results

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "top_k": self.top_k,
            "backend": self.lexical_search.__class__.__name__,
            "model_name_or_path": getattr(self.lexical_search, "model_name_or_path", None),
            **self._lexical_config(),
        }

    @property
    def corpus_size(self) -> int:
        return len(self._corpus_ids)

    @property
    def is_built(self) -> bool:
        return self._is_built

    def _build_signature(self, corpus: dict[str, dict[str, Any]]) -> str:
        return json.dumps(
            {
                "corpus": corpus_hash(corpus),
                "backend": self.lexical_search.__class__.__name__,
                "model_name_or_path": getattr(self.lexical_search, "model_name_or_path", None),
                **self._lexical_config(),
            },
            sort_keys=True,
        )

    def _cache_path(self, corpus: dict[str, dict[str, Any]]) -> Path:
        return cache_root(
            self.cache_dir,
            "lexical",
            getattr(self.lexical_search, "model_name_or_path", self.lexical_search.__class__.__name__),
            corpus_hash(corpus),
            self._build_signature(corpus),
        )

    def _lexical_config(self) -> dict[str, Any]:
        stemmer = getattr(self.lexical_search, "stemmer", None)
        return {
            "k1": getattr(self.lexical_search, "k1", None),
            "b": getattr(self.lexical_search, "b", None),
            "stopwords": getattr(self.lexical_search, "stopwords", None),
            "stemmer": None if stemmer is None else self._object_signature(stemmer),
        }

    @staticmethod
    def _object_signature(value: Any) -> str:
        value_type = value.__class__
        return f"{value_type.__module__}.{value_type.__qualname__}:{value!r}"
