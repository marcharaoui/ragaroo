import os
from pprint import pprint

from dotenv import load_dotenv

import ragaroo as roo


DATASET_PATHS = [os.path.join("data", "nfcorpus")]
MODEL_CACHE = "./models"
EMBEDDER_MODEL = "intfloat/e5-small-v2"


def run_dataset(dataset_path: str) -> None:
    dataset = roo.Dataset.from_folder(dataset_path)
    print(f"\nDataset summary: {dataset_path}")
    pprint(dataset.summary())
    print()

    embedder = roo.SentenceTransformerEmbedder(EMBEDDER_MODEL)
    pipelines = [
        roo.Pipeline(name="bm25", retriever=roo.BM25Retriever(top_k=10)),
        roo.Pipeline(
            name="dense_hnsw",
            retriever=roo.DenseRetriever(
                embedder=embedder,
                top_k=10,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
    ]

    output_dir = os.path.join("results", "multi_dataset", os.path.basename(dataset_path))
    report = roo.Experiment(
        dataset=dataset,
        pipelines=pipelines,
        query_limit=25,
        output_dir=output_dir,
        show_progress=True,
    ).run()
    report.summary(sort_by="mrr@10")


def main() -> None:
    load_dotenv()
    roo.store_models(MODEL_CACHE)

    for dataset_path in DATASET_PATHS:
        run_dataset(dataset_path)


if __name__ == "__main__":
    main()
