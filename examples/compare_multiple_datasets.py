import os
from pprint import pprint

from dotenv import load_dotenv

import ragaroo as rr


DATASET_PATHS = [os.path.join("data", "nfcorpus")]
MODEL_CACHE = "./models"
EMBEDDER_MODEL = "intfloat/e5-small-v2"


def run_dataset(dataset_path: str) -> None:
    dataset = rr.Dataset.from_folder(dataset_path)
    print(f"\nDataset summary: {dataset_path}")
    pprint(dataset.summary())
    print()

    embedder = rr.SentenceTransformerEmbedder(EMBEDDER_MODEL)
    pipelines = [
        rr.Pipeline(name="bm25", retriever=rr.BM25Retriever(top_k=10)),
        rr.Pipeline(
            name="dense_hnsw",
            retriever=rr.DenseRetriever(
                embedder=embedder,
                top_k=10,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
    ]

    output_dir = os.path.join("results", "multi_dataset", os.path.basename(dataset_path))
    report = rr.Experiment(
        dataset=dataset,
        pipelines=pipelines,
        query_limit=25,
        output_dir=output_dir,
        show_progress=True,
    ).run()
    report.summary(sort_by="mrr@10")


def main() -> None:
    load_dotenv()
    rr.store_models(MODEL_CACHE)

    for dataset_path in DATASET_PATHS:
        run_dataset(dataset_path)


if __name__ == "__main__":
    main()
