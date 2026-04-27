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
            "kwargs": _serialize_config_value(self.kwargs),
        }


def HyDE(**kwargs: Any) -> QueryTransformSpec:
    return QueryTransformSpec(HyDEQueryTransform, kwargs=dict(kwargs))


def SpellingCorrection(**kwargs: Any) -> QueryTransformSpec:
    return QueryTransformSpec(LLMSpellingCorrection, kwargs=dict(kwargs))


def IntentClarification(**kwargs: Any) -> QueryTransformSpec:
    return QueryTransformSpec(IntentClarificationTransform, kwargs=dict(kwargs))


def _serialize_config_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _serialize_config_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_serialize_config_value(item) for item in value]
    config_dict = getattr(value, "config_dict", None)
    if callable(config_dict):
        return _serialize_config_value(config_dict())
    if isinstance(value, type):
        return value.__name__
    return value.__class__.__name__
