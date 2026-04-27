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
    dataset = roo.Dataset.from_folder(DATASET_PATH)

    print("Dataset summary")
    pprint(dataset.summary())
    print()

    embedder = roo.SentenceTransformerEmbedder(EMBEDDER_MODEL)
    pipelines = [
        roo.Pipeline(
            name="dense_topk_5",
            retriever=roo.DenseRetriever(
                embedder=embedder,
                top_k=5,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
        roo.Pipeline(
            name="dense_topk_10",
            retriever=roo.DenseRetriever(
                embedder=embedder,
                top_k=10,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
        roo.Pipeline(
            name="dense_topk_20",
            retriever=roo.DenseRetriever(
                embedder=embedder,
                top_k=20,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
    ]

    report = roo.Experiment(
        dataset=dataset,
        pipelines=pipelines,
        metrics=[
            "recall@5",
            "recall@10",
            "recall@20",
            "mrr@10",
            "latency_ms",
            "retrieval_latency_ms",
        ],
        query_limit=25,
        output_dir="results/compare_topk",
        show_progress=True,
    ).run()

    report.summary(sort_by="mrr@10")


if __name__ == "__main__":
    main()
