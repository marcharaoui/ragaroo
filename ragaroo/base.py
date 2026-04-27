from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval.types import RetrievedDocument


QueryValue = str | list[str]


class BaseRetriever(ABC):
    """Interface for retrieval backends that index a corpus and return ranked documents."""

    @abstractmethod
    def build_index(self, corpus: dict[str, dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list["RetrievedDocument"]:
        raise NotImplementedError

    def rerank(
        self,
        query: str,
        documents: list["RetrievedDocument"],
    ) -> list["RetrievedDocument"]:
        if not documents:
            self.last_rerank_stats = {
                "rerank_time_s": 0.0,
                "input_candidates": 0.0,
                "matched_candidates": 0.0,
                "output_candidates": 0.0,
                "search_top_k": 0.0,
            }
            return []

        started_at = perf_counter()
        output_k = getattr(self, "top_k", len(documents))
        if output_k <= 0:
            raise ValueError("top_k must be > 0")

        search_k = max(len(documents), output_k)
        corpus_size = getattr(self, "corpus_size", None)
        if isinstance(corpus_size, int) and corpus_size > 0:
            search_k = min(search_k, corpus_size)

        candidate_ids = {document.corpus_id for document in documents}
        candidates_by_id = {document.corpus_id: document for document in documents}
        reranker_results = self.retrieve(query, top_k=search_k)
        from .retrieval.types import RetrievedDocument

        reranked_documents: list[RetrievedDocument] = []
        seen_ids: set[str] = set()
        for result in reranker_results:
            if result.corpus_id not in candidate_ids or result.corpus_id in seen_ids:
                continue
            candidate = candidates_by_id[result.corpus_id]
            reranked_documents.append(
                RetrievedDocument(
                    corpus_id=candidate.corpus_id,
                    score=result.score,
                    text=candidate.text,
                    metadata=candidate.metadata,
                )
            )
            seen_ids.add(result.corpus_id)

        for candidate in documents:
            if candidate.corpus_id not in seen_ids:
                reranked_documents.append(candidate)

        reranked_documents = reranked_documents[:output_k]
        finished_at = perf_counter()
        self.last_rerank_stats = {
            "rerank_time_s": finished_at - started_at,
            "input_candidates": float(len(documents)),
            "matched_candidates": float(len(seen_ids)),
            "output_candidates": float(len(reranked_documents)),
            "search_top_k": float(search_k),
        }
        return reranked_documents

    @abstractmethod
    def config_dict(self) -> dict[str, Any]:
        raise NotImplementedError


class BaseVectorIndex(ABC):
    """Interface for vector indexes used by dense retrieval."""

    @abstractmethod
    def build(self, embeddings: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embeddings: Any, top_k: int) -> tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def config_dict(self) -> dict[str, Any]:
        raise NotImplementedError


class BaseEmbedder(ABC):
    """Interface for text embedders used by dense or sparse retrievers."""

    @abstractmethod
    def encode_documents(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def encode_queries(
        self,
        texts: list[str] | str,
        normalize_embeddings: bool = True,
    ) -> Any:
        raise NotImplementedError


class BaseReranker(ABC):
    """Interface for rerankers that reorder first-stage retrieval candidates."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list["RetrievedDocument"],
    ) -> list["RetrievedDocument"]:
        raise NotImplementedError

    @abstractmethod
    def config_dict(self) -> dict[str, Any]:
        raise NotImplementedError


class BaseQueryTransform(ABC):
    """Interface for deterministic or LLM-based query augmentation."""

    @staticmethod
    def _prompt_with_query(
        query: str,
        *,
        default_user_prompt: str,
        user_prompt: str | None,
    ) -> str:
        prompt = default_user_prompt if user_prompt is None else user_prompt
        query_text = f"Query: {query}" if user_prompt is None else query
        return f"{prompt}\n\n{query_text}"

    def transform(self, query: QueryValue) -> QueryValue:
        if isinstance(query, str):
            return self.transform_one(query)
        return [self.transform_one(item) for item in query]

    @abstractmethod
    def transform_one(self, query: str) -> str:
        raise NotImplementedError

    def config_dict(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__}

    def __call__(self, query: QueryValue) -> QueryValue:
        return self.transform(query)


@dataclass(slots=True)
class SequentialQueryTransform(BaseQueryTransform):
    """Apply multiple query transforms in order."""

    transforms: list[BaseQueryTransform] = field(default_factory=list)

    def transform(self, query: QueryValue) -> QueryValue:
        result = query
        for transform in self.transforms:
            result = transform.transform(result)
        return result

    def transform_one(self, query: str) -> str:
        result = self.transform(query)
        if not isinstance(result, str):
            raise TypeError("SequentialQueryTransform must return a string when given a single query")
        return result

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "transforms": [transform.config_dict() for transform in self.transforms],
        }
