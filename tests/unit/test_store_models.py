import os
import unittest
from pathlib import Path
from unittest.mock import patch

import ragaroo as rr


class TestStoreModels(unittest.TestCase):
    def setUp(self) -> None:
        self.original = os.environ.get("SENTENCE_TRANSFORMERS_HOME")

    def tearDown(self) -> None:
        if self.original is None:
            os.environ.pop("SENTENCE_TRANSFORMERS_HOME", None)
        else:
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = self.original

    def test_store_models_sets_environment_variable(self):
        value = rr.store_models("./models")

        self.assertEqual(value, str(Path("./models")))
        self.assertEqual(os.environ.get("SENTENCE_TRANSFORMERS_HOME"), str(Path("./models")))

    def test_store_models_none_restores_default_behavior(self):
        rr.store_models("./models")
        result = rr.store_models(None)

        self.assertIsNone(result)
        self.assertNotIn("SENTENCE_TRANSFORMERS_HOME", os.environ)

    def test_store_models_loads_dotenv(self):
        with patch("dotenv.load_dotenv") as mocked_load_dotenv:
            rr.store_models("./models")

        mocked_load_dotenv.assert_called_once()


if __name__ == "__main__":
    unittest.main()
