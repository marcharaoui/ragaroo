from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import bm25s


class BM25SLexicalSearch:
    """Small adapter around bm25s tokenization, indexing, persistence, and search."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        *,
        stopwords: str | list[str] = "english",
        stemmer: Any = None,
    ) -> None:
        self.model_name_or_path = "bm25s"
        self.k1 = k1
        self.b = b
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.searcher = bm25s.BM25(k1=k1, b=b)

    def build_index(self, texts: list[str]) -> None:
        tokenized_corpus = self._tokenize(texts)
        self.searcher = bm25s.BM25(k1=self.k1, b=self.b)
        self.searcher.index(tokenized_corpus, show_progress=False)

    def save(self, directory: str | Path) -> None:
        self.searcher.save(directory, corpus=None)

    def load(self, directory: str | Path) -> None:
        self.searcher = bm25s.BM25.load(directory, load_corpus=False)

    def search(
        self,
        query: str,
        corpus_ids: list[str],
        top_k: int,
    ) -> tuple[list[str], list[float]]:
        query_tokens = self._tokenize(query)
        documents, scores = self.searcher.retrieve(
            query_tokens,
            corpus=corpus_ids,
            k=min(top_k, len(corpus_ids)),
            show_progress=False,
        )

        document_ids = [str(value) for value in documents[0]]
        score_values = [float(value) for value in scores[0]]
        return document_ids, score_values

    def _tokenize(self, texts: list[str] | str) -> Any:
        return bm25s.tokenize(
            texts,
            stopwords=self.stopwords,
            stemmer=self.stemmer,
            show_progress=False,
        )
