import os
from pprint import pprint

from dotenv import load_dotenv

import ragaroo as rr


DATASET_PATH = os.path.join("data", "nfcorpus")
MODEL_CACHE = "./models"


def main() -> None:
    load_dotenv()
    rr.store_models(MODEL_CACHE)
    dataset = rr.Dataset.from_folder(DATASET_PATH)

    print("Dataset summary")
    pprint(dataset.summary())
    print()

    pipelines = [
        rr.Pipeline(
            name="dense_e5_small",
            retriever=rr.DenseRetriever(
                embedder=rr.SentenceTransformerEmbedder("intfloat/e5-small-v2"),
                top_k=10,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
        rr.Pipeline(
            name="dense_bge_small",
            retriever=rr.DenseRetriever(
                embedder=rr.SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"),
                top_k=10,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
    ]

    report = rr.Experiment(
        dataset=dataset,
        pipelines=pipelines,
        query_limit=25,
        output_dir="results/compare_models",
        show_progress=True,
    ).run()

    report.summary(sort_by="mrr@10")


if __name__ == "__main__":
    main()
