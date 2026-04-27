import unittest

from ragaroo.base import BaseRetriever
from ragaroo.pipeline.pipeline import Pipeline
from ragaroo.query_augmentation import (
    BaseLLMProvider,
    BaseQueryTransform,
    HyDE,
    IntentClarification,
    SequentialQueryTransform,
    SpellingCorrection,
)
from ragaroo.retrieval.types import RetrievedDocument


class PrefixTransform(BaseQueryTransform):
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def transform_one(self, query: str) -> str:
        return f"{self.prefix} {query}"

    def config_dict(self) -> dict[str, str]:
        return {"type": self.__class__.__name__, "prefix": self.prefix}


class RecordingRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.last_build_stats = {}
        self.last_query_stats = {}
        self.last_query = None

    def build_index(self, corpus):
        self.last_build_stats = {"total_build_time_s": 0.0}

    def retrieve(self, query, top_k=None):
        self.last_query = query
        self.last_query_stats = {"total_query_time_s": 0.0}
        return [RetrievedDocument("d1", 1.0, "doc")]

    def config_dict(self):
        return {"type": "RecordingRetriever"}


class FakeProvider(BaseLLMProvider):
    def __init__(self, outputs=None) -> None:
        self.outputs = list(outputs or [])
        self.prompts = []

    def generate(self, prompt, *, system_prompt=None, temperature=0.0, max_tokens=256):
        self.prompts.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.outputs:
            return self.outputs.pop(0)
        return "synthetic hyde passage"


class TestPipelineQueryAugmentation(unittest.TestCase):
    def test_pipeline_without_query_augmentation_reports_zero_augmentation_time(self):
        retriever = RecordingRetriever()
        pipeline = Pipeline(
            name="plain",
            retriever=retriever,
        )

        pipeline.retrieve("query")

        self.assertEqual(retriever.last_query, "query")
        self.assertEqual(pipeline.last_query_stats["query_augmentation_time_s"], 0.0)
        self.assertEqual(pipeline.last_query_stats["rerank_time_s"], 0.0)

    def test_pipeline_applies_sequential_query_transforms(self):
        retriever = RecordingRetriever()
        pipeline = Pipeline(
            name="augmented",
            retriever=retriever,
            query_augmentation=SequentialQueryTransform(
                [PrefixTransform("spelling"), PrefixTransform("hyde")]
            ),
        )

        pipeline.retrieve("query")

        self.assertEqual(retriever.last_query, "hyde spelling query")
        self.assertGreaterEqual(pipeline.last_query_stats["query_augmentation_time_s"], 0.0)

    def test_pipeline_hash_changes_with_query_transform_config(self):
        retriever = RecordingRetriever()
        pipeline_1 = Pipeline(
            name="augmented",
            retriever=retriever,
            query_augmentation=[PrefixTransform("a")],
        )
        pipeline_2 = Pipeline(
            name="augmented",
            retriever=retriever,
            query_augmentation=[PrefixTransform("b")],
        )

        self.assertNotEqual(pipeline_1.config_hash, pipeline_2.config_hash)

    def test_pipeline_resolves_query_transform_spec_with_llm_provider(self):
        retriever = RecordingRetriever()
        pipeline = Pipeline(
            name="hyde",
            retriever=retriever,
            query_augmentation=[HyDE(max_tokens=32)],
            llm_provider=FakeProvider(),
        )

        pipeline.retrieve("query")

        self.assertEqual(retriever.last_query, "synthetic hyde passage")
        self.assertGreaterEqual(pipeline.last_query_stats["query_augmentation_time_s"], 0.0)

    def test_pipeline_passes_custom_prompts_to_query_transform_spec(self):
        retriever = RecordingRetriever()
        provider = FakeProvider()
        pipeline = Pipeline(
            name="hyde",
            retriever=retriever,
            query_augmentation=[
                HyDE(
                    max_tokens=32,
                    user_prompt="Write this in product-support language.",
                    system_prompt="Custom HyDE system prompt.",
                    temperature=0.65,
                )
            ],
            llm_provider=provider,
        )

        pipeline.retrieve("reset password")

        self.assertEqual(retriever.last_query, "synthetic hyde passage")
        self.assertEqual(
            provider.prompts[0]["prompt"],
            "Write this in product-support language.\n\nreset password",
        )
        self.assertEqual(provider.prompts[0]["system_prompt"], "Custom HyDE system prompt.")
        self.assertEqual(provider.prompts[0]["temperature"], 0.65)
        self.assertEqual(provider.prompts[0]["max_tokens"], 32)

    def test_pipeline_passes_custom_prompts_to_sequential_specs(self):
        retriever = RecordingRetriever()
        provider = FakeProvider(["corrected query", "clarified query"])
        pipeline = Pipeline(
            name="custom-sequential",
            retriever=retriever,
            query_augmentation=[
                SpellingCorrection(
                    user_prompt="Correct spelling with acronym awareness.",
                    system_prompt="Custom spelling system prompt.",
                    temperature=0.1,
                ),
                IntentClarification(
                    user_prompt="Clarify the retrieval intent.",
                    system_prompt="Custom intent system prompt.",
                    temperature=0.2,
                ),
            ],
            llm_provider=provider,
        )

        pipeline.retrieve("retrival qury")

        self.assertEqual(retriever.last_query, "clarified query")
        self.assertEqual(
            provider.prompts[0]["prompt"],
            "Correct spelling with acronym awareness.\n\nretrival qury",
        )
        self.assertEqual(provider.prompts[0]["system_prompt"], "Custom spelling system prompt.")
        self.assertEqual(provider.prompts[0]["temperature"], 0.1)
        self.assertEqual(
            provider.prompts[1]["prompt"],
            "Clarify the retrieval intent.\n\ncorrected query",
        )
        self.assertEqual(provider.prompts[1]["system_prompt"], "Custom intent system prompt.")
        self.assertEqual(provider.prompts[1]["temperature"], 0.2)

    def test_pipeline_raises_when_spec_needs_provider_and_none_is_set(self):
        retriever = RecordingRetriever()
        pipeline = Pipeline(
            name="hyde",
            retriever=retriever,
            query_augmentation=[HyDE(max_tokens=32)],
        )

        with self.assertRaises(ValueError):
            pipeline.retrieve("query")
