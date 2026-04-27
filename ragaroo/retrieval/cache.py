from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def corpus_hash(corpus: dict[str, dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for corpus_id in sorted(corpus):
        item = corpus[corpus_id]
        hasher.update(corpus_id.encode("utf-8"))
        hasher.update(item.get("text", "").encode("utf-8"))
        hasher.update(json.dumps(item.get("metadata", {}), sort_keys=True).encode("utf-8"))
        hasher.update(str(item.get("document_id", "")).encode("utf-8"))
        hasher.update(str(item.get("title", "")).encode("utf-8"))
    return hasher.hexdigest()[:16]


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "default"


def cache_root(
    base_dir: str | Path,
    family: str,
    model_name: str,
    corpus_signature: str,
    config_signature: str,
) -> Path:
    config_hash = hashlib.sha256(config_signature.encode("utf-8")).hexdigest()[:12]
    return (
        Path(base_dir)
        / family
        / slugify(model_name)
        / f"corpus_{corpus_signature}"
        / f"config_{config_hash}"
    )


def save_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def load_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
