from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .base import BaseEmbedder, BaseReranker, BaseRetriever, BaseVectorIndex

try:
    __version__ = version("ragaroo")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .dataset import Dataset
from .evaluation.evaluation import Evaluation, Evaluator, evaluate
from .experiment import Experiment
from .pipeline.pipeline import Pipeline
from .query_augmentation import (
    BaseLLMProvider,
    BaseQueryTransform,
    HyDE,
    HyDEQueryTransform,
    IntentClarification,
    IntentClarificationTransform,
    LLMSpellingCorrection,
    OpenRouterProvider,
    QueryTransformSpec,
    SequentialQueryTransform,
    SpellingCorrection,
)
from .report import Report
from .reranking import CrossEncoderReranker, SentenceTransformerCrossEncoder
from .retrieval.dense.dense import DenseRetriever
from .retrieval.hybrid import HybridRetriever
from .retrieval.dense import ProprietaryEmbedder, SentenceTransformerEmbedder
from .retrieval.lexical import BM25Retriever, BM25SLexicalSearch
from .retrieval.sparse import SentenceTransformerSparseEmbedder, SparseRetriever


def store_models(path_folder: str | os.PathLike[str] | None = None) -> str | None:
    """Load local environment variables and set or clear the Sentence Transformers cache directory."""
    from dotenv import load_dotenv

    load_dotenv()

    if path_folder is None:
        os.environ.pop("SENTENCE_TRANSFORMERS_HOME", None)
        return None

    resolved = str(Path(path_folder))
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = resolved
    return resolved


__all__ = [
    "__version__",
    "BM25Retriever",
    "BM25SLexicalSearch",
    "BaseEmbedder",
    "BaseLLMProvider",
    "BaseQueryTransform",
    "BaseReranker",
    "BaseRetriever",
    "BaseVectorIndex",
    "CrossEncoderReranker",
    "Dataset",
    "DenseRetriever",
    "Evaluation",
    "Evaluator",
    "Experiment",
    "HyDE",
    "HyDEQueryTransform",
    "HybridRetriever",
    "IntentClarification",
    "IntentClarificationTransform",
    "LLMSpellingCorrection",
    "OpenRouterProvider",
    "Pipeline",
    "ProprietaryEmbedder",
    "QueryTransformSpec",
    "Report",
    "SequentialQueryTransform",
    "SentenceTransformerEmbedder",
    "SentenceTransformerCrossEncoder",
    "SentenceTransformerSparseEmbedder",
    "SpellingCorrection",
    "SparseRetriever",
    "store_models",
    "evaluate",
]
