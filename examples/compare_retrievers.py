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
            name="bm25",
            retriever=roo.BM25Retriever(top_k=10),
        ),
        roo.Pipeline(
            name="dense_hnsw",
            retriever=roo.DenseRetriever(
                embedder=embedder,
                top_k=10,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
        roo.Pipeline(
            name="hybrid_rrf",
            retriever=roo.HybridRetriever(
                retriever_1=roo.DenseRetriever(
                    embedder=embedder,
                    top_k=10,
                    index_technique="hnsw",
                    distance_metric="cosine",
                ),
                retriever_2=roo.BM25Retriever(top_k=10),
                top_k=10,
                fusion_technique="rrf",
            ),
        ),
    ]

    report = roo.Experiment(
        dataset=dataset,
        pipelines=pipelines,
        query_limit=25,
        output_dir="results/compare_retrievers",
        show_progress=True,
    ).run()

    report.summary()


if __name__ == "__main__":
    main()
