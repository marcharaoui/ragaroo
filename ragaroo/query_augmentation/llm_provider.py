from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any
from urllib import error, request


class BaseLLMProvider(ABC):
    """Interface for chat-style text generation providers used by query transforms."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        raise NotImplementedError

    def config_dict(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__}

    def generate_many(
        self,
        prompts: list[str],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> list[str]:
        return [
            self.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for prompt in prompts
        ]


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter chat-completion provider for LLM-based query augmentation."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout_s: float = 30.0,
        referer: str | None = None,
        title: str | None = None,
    ) -> None:
        if api_key is None:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file or pass it explicitly."
            )
        cleaned_api_key = api_key.strip()
        if not cleaned_api_key or cleaned_api_key.lower() in {
            "none",
            "your_openrouter_api_key_here",
            "your_real_openrouter_key",
        }:
            raise ValueError(
                "OPENROUTER_API_KEY is missing or invalid. Set a real key in your .env file before using HyDE pipelines."
            )
        self.api_key = cleaned_api_key
        self.model = model
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.referer = referer
        self.title = title

    def config_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "model": self.model,
            "base_url": self.base_url,
            "timeout_s": self.timeout_s,
        }

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title

        req = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 401:
                raise RuntimeError(
                    "OpenRouter returned HTTP 401 Unauthorized. "
                    "Check that OPENROUTER_API_KEY in .env is valid and that main.py loads .env before creating HyDE pipelines."
                ) from exc
            raise RuntimeError(
                f"OpenRouter request failed with HTTP {exc.code}: {response_body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Could not reach OpenRouter: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("OpenRouter request timed out") from exc
        data = json.loads(body)
        return _extract_message_content(data).strip()


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("LLM response does not contain any choices")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
        )
    raise ValueError("LLM response message content is missing or malformed")
