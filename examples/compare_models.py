import os
from pprint import pprint

from dotenv import load_dotenv

import ragaroo as roo


DATASET_PATH = os.path.join("data", "nfcorpus")
MODEL_CACHE = "./models"


def main() -> None:
    load_dotenv()
    roo.store_models(MODEL_CACHE)
    dataset = roo.Dataset.from_folder(DATASET_PATH)

    print("Dataset summary")
    pprint(dataset.summary())
    print()

    pipelines = [
        roo.Pipeline(
            name="dense_e5_small",
            retriever=roo.DenseRetriever(
                embedder=roo.SentenceTransformerEmbedder("intfloat/e5-small-v2"),
                top_k=10,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
        roo.Pipeline(
            name="dense_bge_small",
            retriever=roo.DenseRetriever(
                embedder=roo.SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"),
                top_k=10,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
    ]

    report = roo.Experiment(
        dataset=dataset,
        pipelines=pipelines,
        query_limit=25,
        output_dir="results/compare_models",
        show_progress=True,
    ).run()

    report.summary(sort_by="mrr@10")


if __name__ == "__main__":
    main()
