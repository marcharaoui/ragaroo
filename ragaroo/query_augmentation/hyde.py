from __future__ import annotations

from ..base import BaseQueryTransform
from .llm_provider import BaseLLMProvider


class HyDEQueryTransform(BaseQueryTransform):
    """Generate a hypothetical answer passage and use it as the retrieval query."""

    DEFAULT_USER_PROMPT = "Write a short passage that would answer this retrieval query."
    DEFAULT_SYSTEM_PROMPT = (
        "You write concise hypothetical passages for retrieval augmentation. Return only the passage."
    )
    DEFAULT_TEMPERATURE = 0.2

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        concat_strategy: str = "hyde",
        max_tokens: int = 512,
        user_prompt: str | None = None,
        system_prompt: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.provider = provider
        self.max_tokens = max_tokens
        self.concat_strategy = concat_strategy
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt
        self.temperature = temperature

    def transform_one(self, query: str) -> str:
        hypothetical_document = self.provider.generate(
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
            max_tokens=self.max_tokens,
        ).strip()
        hyde = hypothetical_document or query
        if self.concat_strategy == "query_hyde":
            return f"Initial query: {query}\nHypothetical document: {hyde}"
        elif self.concat_strategy == "hyde":
            return hyde
        else:
            raise ValueError(
                f"Unknown concat_strategy: {self.concat_strategy}. "
                "Choose from 'query_hyde' or 'hyde'."
            )

    def config_dict(self) -> dict[str, str | int | float | None]:
        return {
            "type": self.__class__.__name__,
            "provider": self.provider.__class__.__name__,
            "max_tokens": self.max_tokens,
            "concat_strategy": self.concat_strategy,
            "user_prompt": self.user_prompt,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
        }
