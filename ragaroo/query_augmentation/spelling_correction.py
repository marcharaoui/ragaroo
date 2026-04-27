from __future__ import annotations

from ..base import BaseQueryTransform
from .llm_provider import BaseLLMProvider


class LLMSpellingCorrection(BaseQueryTransform):
    """Correct spelling in retrieval queries while preserving their meaning."""

    DEFAULT_USER_PROMPT = (
        "Correct spelling mistakes in this retrieval query. Return only the corrected query."
    )
    DEFAULT_SYSTEM_PROMPT = (
        "You improve search queries. Keep the meaning unchanged and return only the corrected query."
    )
    DEFAULT_TEMPERATURE = 0.0

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        user_prompt: str | None = None,
        system_prompt: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.provider = provider
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt
        self.temperature = temperature

    def transform_one(self, query: str) -> str:
        corrected = self.provider.generate(
            self._prompt_with_query(
                query,
                default_user_prompt=self.DEFAULT_USER_PROMPT,
                user_prompt=self.user_prompt,
            ),
            system_prompt=(
                self.DEFAULT_SYSTEM_PROMPT
                if self.system_prompt is None
                else self.system_prompt
            ),
            temperature=self.temperature,
            max_tokens=128,
        ).strip()
        return corrected or query

    def config_dict(self) -> dict[str, str | float | None]:
        return {
            "type": self.__class__.__name__,
            "provider": self.provider.__class__.__name__,
            "user_prompt": self.user_prompt,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
        }
