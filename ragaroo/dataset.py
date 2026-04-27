from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean
from typing import Any

from .retrieval.cache import corpus_hash as canonical_corpus_hash


@dataclass(slots=True)
class DatasetStats:
    corpus_size: int
    query_count: int
    qrel_count: int
    avg_chunk_length: float
    avg_query_length: float


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


class Dataset:
    """Load and validate a retrieval benchmark dataset from local files."""

    def __init__(
        self,
        data_folder: str | Path = "data",
        corpus_file: str = "corpus.jsonl",
        queries_file: str = "queries.jsonl",
        qrels_file: str = "qrels.tsv",
        max_corpus: int | None = None,
        max_queries: int | None = None,
        drop_empty_text: bool = True,
    ) -> None:
        if max_corpus is not None and max_corpus <= 0:
            raise ValueError("max_corpus must be > 0")
        if max_queries is not None and max_queries <= 0:
            raise ValueError("max_queries must be > 0")

        self.data_folder = Path(data_folder)
        self.corpus_file = self.data_folder / corpus_file
        self.queries_file = self.data_folder / queries_file
        self.qrels_file = self.data_folder / qrels_file
        self.max_corpus = max_corpus
        self.max_queries = max_queries
        self.drop_empty_text = drop_empty_text

        self.corpus: dict[str, dict[str, Any]] = {}
        self.queries: dict[str, str] = {}
        self.qrels: dict[str, dict[str, float]] = {}

        self.stats: DatasetStats | None = None
        self.validation_report = ValidationReport()
        self.loaded = False
        self._hash_cache: dict[str, str] = {}
        self._dropped_empty_corpus_ids: set[str] = set()
        self._dropped_empty_query_ids: set[str] = set()

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        *,
        max_corpus: int | None = None,
        max_queries: int | None = None,
        drop_empty_text: bool = True,
    ) -> "Dataset":
        """Create and immediately load a dataset from a folder."""
        dataset = cls(
            folder,
            max_corpus=max_corpus,
            max_queries=max_queries,
            drop_empty_text=drop_empty_text,
        )
        dataset.load()
        return dataset

    def load(self) -> "Dataset":
        """Read corpus, queries, and qrels into memory and validate references."""
        self._check_required_files()
        self._dropped_empty_corpus_ids.clear()
        self._dropped_empty_query_ids.clear()

        self.corpus = self._load_corpus()
        self.queries = self._load_queries()
        self.qrels = self._load_qrels()
        self._prune_qrels_for_dropped_items()
        self._apply_query_limit()
        self._apply_corpus_limit()

        self.validation_report = self._validate()
        if not self.validation_report.is_valid:
            joined = "\n".join(f"- {error}" for error in self.validation_report.errors)
            raise ValueError(f"Dataset validation failed:\n{joined}")

        self.stats = self._build_stats()
        self._hash_cache.clear()
        self.loaded = True
        return self

    def summary(self) -> dict[str, Any]:
        """Return dataset counts, hashes, validation messages, and load settings."""
        if self.stats is None:
            return {
                "data_folder": str(self.data_folder),
                "loaded": False,
            }

        summary = {
            "dataset_name": self.data_folder.name,
            "data_folder": str(self.data_folder),
            "dataset_hash": self.dataset_hash,
            "corpus_hash": self.corpus_hash,
            "queries_hash": self.queries_hash,
            "qrels_hash": self.qrels_hash,
            "loaded": self.loaded,
            **asdict(self.stats),
            "validation_errors": list(self.validation_report.errors),
            "validation_warnings": list(self.validation_report.warnings),
        }
        if self.max_queries is not None:
            summary["max_queries"] = self.max_queries
        if self.max_corpus is not None:
            summary["max_corpus"] = self.max_corpus
        summary["drop_empty_text"] = self.drop_empty_text
        summary["dropped_empty_corpus_count"] = len(self._dropped_empty_corpus_ids)
        summary["dropped_empty_query_count"] = len(self._dropped_empty_query_ids)
        return summary

    @property
    def corpus_hash(self) -> str:
        if "corpus" not in self._hash_cache:
            self._hash_cache["corpus"] = canonical_corpus_hash(self.corpus)
        return self._hash_cache["corpus"]

    @property
    def queries_hash(self) -> str:
        if "queries" not in self._hash_cache:
            hasher = hashlib.sha256()
            for query_id in sorted(self.queries):
                hasher.update(query_id.encode("utf-8"))
                hasher.update(self.queries[query_id].encode("utf-8"))
            self._hash_cache["queries"] = hasher.hexdigest()[:16]
        return self._hash_cache["queries"]

    @property
    def qrels_hash(self) -> str:
        if "qrels" not in self._hash_cache:
            hasher = hashlib.sha256()
            for query_id in sorted(self.qrels):
                hasher.update(query_id.encode("utf-8"))
                for corpus_id, score in sorted(self.qrels[query_id].items()):
                    hasher.update(corpus_id.encode("utf-8"))
                    hasher.update(str(score).encode("utf-8"))
            self._hash_cache["qrels"] = hasher.hexdigest()[:16]
        return self._hash_cache["qrels"]

    @property
    def dataset_hash(self) -> str:
        if "dataset" not in self._hash_cache:
            hasher = hashlib.sha256()
            hasher.update(self.corpus_hash.encode("utf-8"))
            hasher.update(self.queries_hash.encode("utf-8"))
            hasher.update(self.qrels_hash.encode("utf-8"))
            self._hash_cache["dataset"] = hasher.hexdigest()[:16]
        return self._hash_cache["dataset"]

    def _check_required_files(self) -> None:
        required = [
            self.corpus_file,
            self.queries_file,
            self.qrels_file,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing dataset file(s): " + ", ".join(missing)
            )

    def _load_corpus(self) -> dict[str, dict[str, Any]]:
        corpus: dict[str, dict[str, Any]] = {}
        with self.corpus_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON in corpus file at line {line_number}: {exc}"
                    ) from exc

                corpus_id = self._extract_id(payload, line_number, "corpus")
                if corpus_id in corpus:
                    raise ValueError(f"Duplicate corpus id '{corpus_id}' at line {line_number}")
                text = payload.get("text")
                if not isinstance(text, str):
                    raise ValueError(
                        f"Corpus row {line_number} is missing a string 'text' field"
                    )
                if not text.strip():
                    if self.drop_empty_text:
                        self._dropped_empty_corpus_ids.add(corpus_id)
                        continue
                    raise ValueError(f"Corpus item '{corpus_id}' has empty text")

                metadata = payload.get("metadata")
                if metadata is None:
                    metadata = {}
                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"Corpus row {line_number} has a non-dict 'metadata' field"
                    )

                corpus[corpus_id] = {
                    "text": text,
                    "title": payload.get("title"),
                    "document_id": payload.get("document_id"),
                    "metadata": metadata,
                }

        return corpus

    def _load_queries(self) -> dict[str, str]:
        queries: dict[str, str] = {}
        with self.queries_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON in queries file at line {line_number}: {exc}"
                    ) from exc

                query_id = self._extract_id(payload, line_number, "query")
                if query_id in queries:
                    raise ValueError(f"Duplicate query id '{query_id}' at line {line_number}")
                text = payload.get("text")
                if not isinstance(text, str):
                    raise ValueError(
                        f"Query row {line_number} is missing a string 'text' field"
                    )
                if not text.strip():
                    if self.drop_empty_text:
                        self._dropped_empty_query_ids.add(query_id)
                        continue
                    raise ValueError(f"Query '{query_id}' has empty text")

                queries[query_id] = text

        return queries

    def _load_qrels(self) -> dict[str, dict[str, float]]:
        qrels: dict[str, dict[str, float]] = {}
        with self.qrels_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for line_number, row in enumerate(reader, start=1):
                if not row:
                    continue

                if line_number == 1 and self._looks_like_qrels_header(row):
                    continue

                if len(row) < 3:
                    raise ValueError(
                        f"Qrels row {line_number} must have at least 3 tab-separated columns"
                    )

                query_id = row[0].strip()
                corpus_id = row[1].strip()
                score_text = row[2].strip()

                if score_text == "0":
                    continue

                try:
                    score = float(score_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Qrels row {line_number} has a non-numeric score '{score_text}'"
                    ) from exc

                if corpus_id in qrels.get(query_id, {}):
                    raise ValueError(
                        f"Duplicate qrel pair for query '{query_id}' and corpus '{corpus_id}'"
                    )
                qrels.setdefault(query_id, {})[corpus_id] = score

        return qrels

    def _apply_query_limit(self) -> None:
        if self.max_queries is None:
            return

        selected_query_ids = [
            query_id
            for query_id in self.queries.keys()
            if query_id in self.qrels
        ][: self.max_queries]

        self.queries = {
            query_id: self.queries[query_id]
            for query_id in selected_query_ids
        }
        self.qrels = {
            query_id: self.qrels[query_id]
            for query_id in selected_query_ids
        }

    def _apply_corpus_limit(self) -> None:
        if self.max_corpus is None:
            return

        required_ids = {
            corpus_id
            for rels in self.qrels.values()
            for corpus_id in rels.keys()
            if corpus_id in self.corpus
        }

        selected_ids: list[str] = []
        for corpus_id in self.corpus.keys():
            if corpus_id in required_ids:
                selected_ids.append(corpus_id)

        if len(selected_ids) < self.max_corpus:
            for corpus_id in self.corpus.keys():
                if corpus_id in required_ids or corpus_id in selected_ids:
                    continue
                selected_ids.append(corpus_id)
                if len(selected_ids) >= self.max_corpus:
                    break

        self.corpus = {
            corpus_id: self.corpus[corpus_id]
            for corpus_id in selected_ids
        }

    def _prune_qrels_for_dropped_items(self) -> None:
        if not self._dropped_empty_corpus_ids and not self._dropped_empty_query_ids:
            return

        pruned_qrels: dict[str, dict[str, float]] = {}
        for query_id, rels in self.qrels.items():
            if query_id in self._dropped_empty_query_ids:
                continue
            kept_rels = {
                corpus_id: score
                for corpus_id, score in rels.items()
                if corpus_id not in self._dropped_empty_corpus_ids
            }
            if kept_rels:
                pruned_qrels[query_id] = kept_rels
        self.qrels = pruned_qrels

    def _validate(self) -> ValidationReport:
        report = ValidationReport()

        if not self.corpus:
            report.errors.append("Corpus is empty")
        if not self.queries:
            report.errors.append("Queries are empty")
        if not self.qrels:
            report.errors.append("Qrels are empty")

        for corpus_id, item in self.corpus.items():
            if not corpus_id.strip():
                report.errors.append("Corpus contains an empty id")
            if not item["text"].strip():
                report.errors.append(f"Corpus item '{corpus_id}' has empty text")

        for query_id, query_text in self.queries.items():
            if not query_id.strip():
                report.errors.append("Queries contain an empty id")
            if not query_text.strip():
                report.errors.append(f"Query '{query_id}' has empty text")

        for query_id, rels in self.qrels.items():
            if query_id not in self.queries:
                report.errors.append(f"Qrels reference unknown query id '{query_id}'")
                continue

            if not rels:
                report.warnings.append(f"Query '{query_id}' has no relevant documents")

            for corpus_id, score in rels.items():
                if corpus_id not in self.corpus:
                    report.errors.append(
                        f"Qrels for query '{query_id}' reference unknown corpus id '{corpus_id}'"
                    )
                if score <= 0:
                    report.warnings.append(
                        f"Qrels score for query '{query_id}' and corpus '{corpus_id}' is non-positive"
                    )

        if self._dropped_empty_corpus_ids:
            report.warnings.append(
                f"Dropped {len(self._dropped_empty_corpus_ids)} corpus rows with empty text"
            )
        if self._dropped_empty_query_ids:
            report.warnings.append(
                f"Dropped {len(self._dropped_empty_query_ids)} query rows with empty text"
            )

        return report

    def _build_stats(self) -> DatasetStats:
        chunk_lengths = [len(item["text"].split()) for item in self.corpus.values()]
        query_lengths = [len(text.split()) for text in self.queries.values()]
        qrel_count = sum(len(rels) for rels in self.qrels.values())

        return DatasetStats(
            corpus_size=len(self.corpus),
            query_count=len(self.queries),
            qrel_count=qrel_count,
            avg_chunk_length=mean(chunk_lengths) if chunk_lengths else 0.0,
            avg_query_length=mean(query_lengths) if query_lengths else 0.0,
        )

    @staticmethod
    def _extract_id(payload: dict[str, Any], line_number: int, kind: str) -> str:
        value = payload.get("id") or payload.get("_id")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{kind.capitalize()} row {line_number} is missing a valid 'id' or '_id' field"
            )
        return value

    @staticmethod
    def _looks_like_qrels_header(row: list[str]) -> bool:
        first = row[0].strip().lower()
        second = row[1].strip().lower() if len(row) > 1 else ""
        return "query" in first and "corpus" in second
