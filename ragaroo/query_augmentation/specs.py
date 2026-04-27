from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..base import BaseQueryTransform
from .hyde import HyDEQueryTransform
from .intent_clarification import IntentClarificationTransform
from .spelling_correction import LLMSpellingCorrection


@dataclass(frozen=True, slots=True)
class QueryTransformSpec:
    """Deferred query-transform configuration resolved by Pipeline with llm_provider."""

    transform_class: type[BaseQueryTransform]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": self.transform_class.__name__,
            "kwargs": self.kwargs,
        }


def HyDE(**kwargs: Any) -> QueryTransformSpec:
    return QueryTransformSpec(HyDEQueryTransform, kwargs=dict(kwargs))


def SpellingCorrection(**kwargs: Any) -> QueryTransformSpec:
    return QueryTransformSpec(LLMSpellingCorrection, kwargs=dict(kwargs))


def IntentClarification(**kwargs: Any) -> QueryTransformSpec:
    return QueryTransformSpec(IntentClarificationTransform, kwargs=dict(kwargs))
