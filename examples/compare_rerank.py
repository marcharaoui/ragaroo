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
            name="dense_only",
            retriever=rr.DenseRetriever(
                embedder=embedder,
                top_k=20,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
        rr.Pipeline(
            name="dense_rerank_top10",
            retriever=rr.DenseRetriever(
                embedder=embedder,
                top_k=50,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
            reranker=rr.CrossEncoderReranker(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_k=10,
            ),
        ),
        rr.Pipeline(
            name="bm25_dense_rerank_top10",
            retriever=rr.BM25Retriever(top_k=50),
            reranker=rr.DenseRetriever(
                embedder=embedder,
                top_k=10,
                index_technique="hnsw",
                distance_metric="cosine",
            ),
        ),
    ]

    report = rr.Experiment(
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
