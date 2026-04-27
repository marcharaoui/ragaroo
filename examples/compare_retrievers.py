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
            name="bm25",
            retriever=rr.BM25Retriever(top_k=10),
        ),
        rr.Pipeline(
            name="dense_hnsw",
            retriever=rr.DenseRetriever(
                embedder=embedder,
                top_k=10,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
        rr.Pipeline(
            name="hybrid_rrf",
            retriever=rr.HybridRetriever(
                retriever_1=rr.DenseRetriever(
                    embedder=embedder,
                    top_k=10,
                    index_technique="hnsw",
                    distance_metric="cosine",
                ),
                retriever_2=rr.BM25Retriever(top_k=10),
                top_k=10,
                fusion_technique="rrf",
            ),
        ),
    ]

    report = rr.Experiment(
        dataset=dataset,
        pipelines=pipelines,
        query_limit=25,
        output_dir="results/compare_retrievers",
        show_progress=True,
    ).run()

    report.summary()


if __name__ == "__main__":
    main()
