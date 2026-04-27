from __future__ import annotations

import os
from typing import Any, Callable


def resolve_hf_token(hf_token: str | None = None) -> str | None:
    token = (
        hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if token is None:
        return None

    cleaned = token.strip()
    return cleaned or None


def init_model_with_hf_token(
    factory: Callable[..., Any],
    *,
    hf_token: str | None = None,
    **kwargs: Any,
) -> Any:
    token = resolve_hf_token(hf_token)
    if token is None:
        return factory(**kwargs)

    try:
        return factory(**kwargs, token=token)
    except TypeError as exc:
        if "token" not in str(exc):
            raise
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        return factory(**kwargs)
