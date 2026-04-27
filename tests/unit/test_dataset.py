import json
import tempfile
import unittest
from pathlib import Path

from ragaroo import Dataset
from ragaroo.retrieval.cache import corpus_hash as canonical_corpus_hash


class TestDataset(unittest.TestCase):
    def _write_dataset(self, root: Path) -> Path:
        dataset_dir = root / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "corpus.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                    json.dumps({"id": "d2", "text": "banana potassium benefits"}),
                ]
            ),
            encoding="utf-8",
        )
        (dataset_dir / "queries.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"id": "q1", "text": "apple facts"}),
                    json.dumps({"id": "q2", "text": "banana benefits"}),
                ]
            ),
            encoding="utf-8",
        )
        (dataset_dir / "qrels.tsv").write_text(
            "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n",
            encoding="utf-8",
        )
        return dataset_dir

    def test_load_dataset_components(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = self._write_dataset(Path(tmpdir))
            dataset = Dataset(dataset_dir, max_queries=1).load()

            self.assertTrue(dataset.loaded)
            self.assertGreater(len(dataset.corpus), 0)
            self.assertEqual(len(dataset.queries), 1)
            self.assertGreater(len(dataset.qrels), 0)
            self.assertIsNotNone(dataset.stats)

    def test_dataset_limits_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "max_corpus must be > 0"):
            Dataset(max_corpus=0)

        with self.assertRaisesRegex(ValueError, "max_queries must be > 0"):
            Dataset(max_queries=-1)

    def test_summary_contains_hashes_and_core_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = self._write_dataset(Path(tmpdir))
            dataset = Dataset.from_folder(dataset_dir)
            summary = dataset.summary()

            self.assertEqual(summary["dataset_name"], "dataset")
            self.assertIn("corpus_size", summary)
            self.assertIn("query_count", summary)
            self.assertIn("qrel_count", summary)
            self.assertIn("avg_chunk_length", summary)
            self.assertIn("dataset_hash", summary)
            self.assertIn("corpus_hash", summary)
            self.assertIn("queries_hash", summary)
            self.assertIn("qrels_hash", summary)
            self.assertEqual(summary["corpus_hash"], canonical_corpus_hash(dataset.corpus))

    def test_empty_corpus_and_query_rows_are_dropped_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "corpus.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                        json.dumps({"id": "d2", "text": "   "}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "q1", "text": "apple facts"}),
                        json.dumps({"id": "q2", "text": " "}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset.from_folder(dataset_dir)

            self.assertEqual(set(dataset.corpus), {"d1"})
            self.assertEqual(set(dataset.queries), {"q1"})
            self.assertEqual(set(dataset.qrels), {"q1"})
            self.assertEqual(dataset.summary()["dropped_empty_corpus_count"], 1)
            self.assertEqual(dataset.summary()["dropped_empty_query_count"], 1)
            self.assertIn("Dropped 1 corpus rows with empty text", dataset.validation_report.warnings)
            self.assertIn("Dropped 1 query rows with empty text", dataset.validation_report.warnings)

    def test_reload_clears_previous_dropped_empty_row_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "corpus.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                        json.dumps({"id": "d2", "text": "   "}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "q1", "text": "apple facts"}),
                        json.dumps({"id": "q2", "text": " "}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n",
                encoding="utf-8",
            )

            dataset = Dataset(dataset_dir).load()
            self.assertEqual(dataset.summary()["dropped_empty_corpus_count"], 1)
            self.assertEqual(dataset.summary()["dropped_empty_query_count"], 1)

            (dataset_dir / "corpus.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "d1", "text": "apple nutrition facts"}),
                        json.dumps({"id": "d2", "text": "banana potassium benefits"}),
                    ]
                ),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "q1", "text": "apple facts"}),
                        json.dumps({"id": "q2", "text": "banana benefits"}),
                    ]
                ),
                encoding="utf-8",
            )

            dataset.load()

            self.assertEqual(dataset.summary()["dropped_empty_corpus_count"], 0)
            self.assertEqual(dataset.summary()["dropped_empty_query_count"], 0)

    def test_strict_mode_still_fails_on_empty_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "corpus.jsonl").write_text(
                json.dumps({"id": "d1", "text": ""}),
                encoding="utf-8",
            )
            (dataset_dir / "queries.jsonl").write_text(
                json.dumps({"id": "q1", "text": "apple facts"}),
                encoding="utf-8",
            )
            (dataset_dir / "qrels.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                Dataset.from_folder(dataset_dir, drop_empty_text=False)


if __name__ == "__main__":
    unittest.main()
