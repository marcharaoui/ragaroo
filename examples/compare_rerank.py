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
            name="dense_only",
            retriever=roo.DenseRetriever(
                embedder=embedder,
                top_k=20,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
        roo.Pipeline(
            name="dense_rerank_top10",
            retriever=roo.DenseRetriever(
                embedder=embedder,
                top_k=50,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
            reranker=roo.CrossEncoderReranker(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_k=10,
            ),
        ),
        roo.Pipeline(
            name="bm25_dense_rerank_top10",
            retriever=roo.BM25Retriever(top_k=50),
            reranker=roo.DenseRetriever(
                embedder=embedder,
                top_k=10,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
    ]

    report = roo.Experiment(
        dataset=dataset,
        pipelines=pipelines,
        query_limit=10,
        warmup_queries=1,
        output_dir="results/compare_rerank",
        show_progress=True,
    ).run()

    report.summary(sort_by="mrr@10")


if __name__ == "__main__":
    main()
