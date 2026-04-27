import json
import unittest
from urllib import error
from unittest.mock import patch

from ragaroo.query_augmentation import (
    BaseLLMProvider,
    HyDEQueryTransform,
    IntentClarificationTransform,
    LLMSpellingCorrection,
    OpenRouterProvider,
    SequentialQueryTransform,
)
from ragaroo.base import BaseQueryTransform


class FakeProvider(BaseLLMProvider):
    def __init__(self, outputs):
        self.outputs = list(outputs)
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
        return self.outputs.pop(0)


class PassThroughTransform(BaseQueryTransform):
    def transform_one(self, query: str) -> str:
        return query


class TestQueryAugmentation(unittest.TestCase):
    def test_base_transform_can_keep_query_unchanged(self):
        transform = PassThroughTransform()

        self.assertEqual(transform.transform("alpha query"), "alpha query")
        self.assertEqual(transform.transform(["a", "b"]), ["a", "b"])

    def test_sequential_transform_applies_in_order(self):
        spelling_provider = FakeProvider(["apple benefits"])
        hyde_provider = FakeProvider(["apple benefits improve recovery and support nutrition."])
        transform = SequentialQueryTransform(
            [
                LLMSpellingCorrection(spelling_provider),
                HyDEQueryTransform(hyde_provider),
            ]
        )

        result = transform.transform("appel benfits")

        self.assertEqual(result, "apple benefits improve recovery and support nutrition.")

    def test_llm_transforms_use_default_prompts_when_overrides_are_none(self):
        cases = [
            (
                LLMSpellingCorrection,
                LLMSpellingCorrection.DEFAULT_USER_PROMPT,
                LLMSpellingCorrection.DEFAULT_SYSTEM_PROMPT,
                LLMSpellingCorrection.DEFAULT_TEMPERATURE,
            ),
            (
                IntentClarificationTransform,
                IntentClarificationTransform.DEFAULT_USER_PROMPT,
                IntentClarificationTransform.DEFAULT_SYSTEM_PROMPT,
                IntentClarificationTransform.DEFAULT_TEMPERATURE,
            ),
            (
                HyDEQueryTransform,
                HyDEQueryTransform.DEFAULT_USER_PROMPT,
                HyDEQueryTransform.DEFAULT_SYSTEM_PROMPT,
                HyDEQueryTransform.DEFAULT_TEMPERATURE,
            ),
        ]

        for transform_class, default_user_prompt, default_system_prompt, default_temperature in cases:
            with self.subTest(transform=transform_class.__name__):
                provider = FakeProvider(["augmented query"])
                transform = transform_class(provider)

                transform.transform("original query")

                self.assertEqual(
                    provider.prompts[0]["prompt"],
                    f"{default_user_prompt}\n\nQuery: original query",
                )
                self.assertEqual(provider.prompts[0]["system_prompt"], default_system_prompt)
                self.assertEqual(provider.prompts[0]["temperature"], default_temperature)

    def test_llm_transforms_use_custom_prompts_and_append_query(self):
        cases = [
            LLMSpellingCorrection,
            IntentClarificationTransform,
            HyDEQueryTransform,
        ]

        for transform_class in cases:
            with self.subTest(transform=transform_class.__name__):
                provider = FakeProvider(["augmented query"])
                transform = transform_class(
                    provider,
                    user_prompt="Use the domain-specific rewrite rules.",
                    system_prompt="Custom system instructions.",
                    temperature=0.7,
                )

                transform.transform("original query")

                self.assertEqual(
                    provider.prompts[0]["prompt"],
                    "Use the domain-specific rewrite rules.\n\noriginal query",
                )
                self.assertEqual(provider.prompts[0]["system_prompt"], "Custom system instructions.")
                self.assertEqual(provider.prompts[0]["temperature"], 0.7)

    def test_empty_prompt_overrides_are_preserved(self):
        provider = FakeProvider(["corrected query"])
        transform = LLMSpellingCorrection(provider, user_prompt="", system_prompt="")

        transform.transform("original query")

        self.assertEqual(provider.prompts[0]["prompt"], "\n\noriginal query")
        self.assertEqual(provider.prompts[0]["system_prompt"], "")

    def test_intent_clarification_returns_model_output(self):
        provider = FakeProvider(["benefits of apples for human nutrition"])
        transform = IntentClarificationTransform(provider)

        result = transform.transform("apple benefits")

        self.assertEqual(result, "benefits of apples for human nutrition")

    def test_openrouter_provider_parses_chat_completion_response(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "corrected query",
                    }
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with patch(
            "ragaroo.query_augmentation.llm_provider.request.urlopen",
            return_value=FakeResponse(),
        ) as mocked:
            provider = OpenRouterProvider(api_key="key", model="model")
            result = provider.generate("prompt")

        self.assertEqual(result, "corrected query")
        self.assertEqual(mocked.call_count, 1)

    def test_openrouter_provider_rejects_placeholder_api_key(self):
        with self.assertRaisesRegex(ValueError, "missing or invalid"):
            OpenRouterProvider(api_key="your_real_openrouter_key", model="model")

    def test_openrouter_provider_preserves_empty_system_prompt(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "corrected query",
                    }
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with patch("ragaroo.query_augmentation.llm_provider.request.urlopen", return_value=FakeResponse()) as mocked:
            provider = OpenRouterProvider(api_key="key", model="model")
            provider.generate("prompt", system_prompt="")

        request_arg = mocked.call_args.args[0]
        request_payload = json.loads(request_arg.data.decode("utf-8"))
        self.assertEqual(
            request_payload["messages"],
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": "prompt"},
            ],
        )

    def test_openrouter_provider_raises_clear_error_on_401(self):
        class FakeUnauthorizedResponse:
            def read(self):
                return b'{"error":"Unauthorized"}'

            def close(self):
                return None

        with patch(
            "ragaroo.query_augmentation.llm_provider.request.urlopen",
            side_effect=error.HTTPError(
                url="https://openrouter.ai/api/v1/chat/completions",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=FakeUnauthorizedResponse(),
            ),
        ):
            provider = OpenRouterProvider(api_key="bad-key", model="model")
            with self.assertRaisesRegex(RuntimeError, "401 Unauthorized"):
                provider.generate("prompt")
