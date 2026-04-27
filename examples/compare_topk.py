import os
from pprint import pprint

from dotenv import load_dotenv

import ragaroo as rr


DATASET_PATH = os.path.join("data", "nfcorpus")
MODEL_CACHE = "./models"
EMBEDDER_MODEL = "intfloat/e5-small-v2"


def main() -> None:
    load_dotenv()
    rr.store_models(MODEL_CACHE)
    dataset = rr.Dataset.from_folder(DATASET_PATH)

    print("Dataset summary")
    pprint(dataset.summary())
    print()

    embedder = rr.SentenceTransformerEmbedder(EMBEDDER_MODEL)
    pipelines = [
        rr.Pipeline(
            name="dense_topk_5",
            retriever=rr.DenseRetriever(
                embedder=embedder,
                top_k=5,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
        rr.Pipeline(
            name="dense_topk_10",
            retriever=rr.DenseRetriever(
                embedder=embedder,
                top_k=10,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
        rr.Pipeline(
            name="dense_topk_20",
            retriever=rr.DenseRetriever(
                embedder=embedder,
                top_k=20,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
    ]

    report = rr.Experiment(
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
