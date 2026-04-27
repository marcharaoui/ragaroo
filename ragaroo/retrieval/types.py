from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievedDocument:
    """One ranked retrieval result returned by a retriever or reranker."""

    corpus_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
