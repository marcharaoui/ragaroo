import os
from pprint import pprint

from dotenv import load_dotenv

import ragaroo as roo


DATASET_PATH = os.path.join("data", "nfcorpus")
MODEL_CACHE = "./models"
EMBEDDER_MODEL = "intfloat/e5-small-v2"


def main() -> None:
    load_dotenv()
    roo.store_models(MODEL_CACHE)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("No OPENROUTER_API_KEY found in .env. Skipping HyDE example.")
        return

    dataset = roo.Dataset.from_folder(DATASET_PATH)

    print("Dataset summary")
    pprint(dataset.summary())
    print()

    embedder = roo.SentenceTransformerEmbedder(EMBEDDER_MODEL)
    provider = roo.OpenRouterProvider(
        api_key=api_key,
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    )

    pipelines = [
        roo.Pipeline(
            name="dense_baseline",
            retriever=roo.DenseRetriever(
                embedder=embedder,
                top_k=10,
                index_technique="flat",
                distance_metric="cosine",
            ),
        ),
        roo.Pipeline(
            name="dense_hyde",
            retriever=roo.DenseRetriever(
                embedder=embedder,
                top_k=10,
                index_technique="flat",
                distance_metric="cosine",
            ),
            query_augmentation=[
                roo.HyDEQueryTransform(
                    provider=provider,
                    concat_strategy="hyde",
                    max_tokens=220,
                )
            ],
        ),
    ]

    report = roo.Experiment(
        dataset=dataset,
        pipelines=pipelines,
        query_limit=10,
        warmup_queries=1,
        output_dir="results/compare_hyde",
        show_progress=True,
    ).run()

    report.summary()


if __name__ == "__main__":
    main()
