import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ragaroo.retrieval.types import RetrievedDocument


class FakeDenseEmbedder:
    embedding_dim = 2
    model_name_or_path = "fake-dense"

    def __init__(self, *args, **kwargs):
        pass

    def encode_documents(self, texts, normalize_embeddings=True):
        return self._encode(texts)

    def encode_queries(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        return self._encode(texts)

    def _encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "apple" in lowered or "fruit" in lowered:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class FakeCrossEncoderReranker:
    top_k = 10

    def __init__(self, *args, top_k=10, **kwargs):
        self.top_k = top_k

    def rerank(self, query, documents):
        ranked = sorted(documents, key=lambda document: ("apple" not in document.text.lower(), -document.score))
        return [
            RetrievedDocument(document.corpus_id, document.score, document.text, document.metadata)
            for document in ranked[: self.top_k]
        ]

    def config_dict(self):
        return {"type": "FakeCrossEncoderReranker", "top_k": self.top_k}


class FakeOpenRouterProvider:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt, *, system_prompt=None, temperature=0.0, max_tokens=256):
        return "apple nutrition facts and fruit evidence"

    def config_dict(self):
        return {"type": "FakeOpenRouterProvider"}


class TestExamples(unittest.TestCase):
    repo_root = Path(__file__).resolve().parents[2]

    def _load_script(self, relative_path: str):
        path = self.repo_root / relative_path
        module_name = "test_loaded_" + relative_path.replace("\\", "_").replace("/", "_").replace(".", "_")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _write_dataset(self, root: Path) -> Path:
        dataset_dir = root / "data" / "nfcorpus"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "corpus.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                    json.dumps({"id": "d2", "text": "banana potassium benefits"}),
                    json.dumps({"id": "d3", "text": "fruit health evidence"}),
                ]
            ),
            encoding="utf-8",
        )
        (dataset_dir / "queries.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"id": "q1", "text": "apple facts"}),
                    json.dumps({"id": "q2", "text": "banana benefits"}),
                    json.dumps({"id": "q3", "text": "fruit evidence"}),
                ]
            ),
            encoding="utf-8",
        )
        (dataset_dir / "qrels.tsv").write_text(
            "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\nq3\td3\t1\n",
            encoding="utf-8",
        )
        return dataset_dir

    def _run_example(self, script_path: str, *, include_hyde: bool = False) -> None:
        module = self._load_script(script_path)
        ragaroo_module = getattr(module, "roo", getattr(module, "rr", None))
        if ragaroo_module is None:
            raise AttributeError(f"{script_path} must import ragaroo as roo")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_dataset(root)
            env = {
                "OPENROUTER_API_KEY": "test-key" if include_hyde else "",
                "OPENROUTER_MODEL": "test-model",
            }
            with patch.dict(os.environ, env, clear=False), patch.object(
                ragaroo_module, "SentenceTransformerEmbedder", FakeDenseEmbedder
            ), patch.object(
                ragaroo_module, "CrossEncoderReranker", FakeCrossEncoderReranker
            ), patch.object(
                ragaroo_module, "OpenRouterProvider", FakeOpenRouterProvider
            ):
                original_cwd = Path.cwd()
                try:
                    os.chdir(root)
                    module.main()
                finally:
                    os.chdir(original_cwd)

    def test_example_scripts_run_with_lightweight_fakes(self):
        for module_name in [
            "examples/compare_retrievers.py",
            "examples/compare_topk.py",
            "examples/compare_models.py",
            "examples/compare_rerank.py",
            "examples/compare_multiple_datasets.py",
        ]:
            with self.subTest(script=module_name):
                self._run_example(module_name)

    def test_hyde_example_runs_with_lightweight_fakes(self):
        self._run_example("examples/compare_hyde.py", include_hyde=True)
