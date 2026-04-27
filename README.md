# Ragaroo: A Lightweight Retrieval Benchmarking Library for Your Custom Datasets


<p align="center">
  <img src="images/ragaroo.png" alt="Ragaroo Logo" width="50%">
</p>

Ragaroo is a lightweight Python library for benchmarking retrieval pipelines on custom datasets. It is designed for practical RAG/retrieval experiments: load a dataset, define pipelines, run an experiment, compare quality and latency, and save reproducible artifacts.

## Features

- Strict dataset loading from `corpus.jsonl`, `queries.jsonl`, and `qrels.tsv`
- BM25 retrieval with `bm25s`
- Dense retrieval with Sentence Transformers and FAISS
- Sparse retrieval with Sentence Transformers sparse encoders
- Hybrid retrieval with reciprocal-rank fusion or average-score fusion
- Cross-encoder reranking and retriever-as-reranker workflows
- LLM-based query augmentation: HyDE, spelling correction, intent clarification
- Ranking metrics, latency metrics, CSV/JSON exports, plots, and manifests
- Local index caching for repeatable experiments

## Installation

For most users, installation should be simple:
```bash
pip install ragaroo
```
OR,

If you are working directly from the GitHub repository:
```bash
git clone https://github.com/marcharaoui/ragaroo.git
cd ragaroo
```

For development:
```bash
uv sync --extra dev
```

From a local checkout:

```bash
pip install -e .
```

Ragaroo requires Python 3.10+.

## Dataset Format

Each dataset folder must contain:

```text
my_dataset/
  corpus.jsonl
  queries.jsonl
  qrels.tsv
```

`corpus.jsonl`:

```json
{"id":"d1","text":"Paris is the capital of France.","metadata":{"source":"wiki"}}
{"id":"d2","text":"Berlin is the capital of Germany."}
```

`queries.jsonl`:

```json
{"id":"q1","text":"capital of France"}
{"id":"q2","text":"capital of Germany"}
```

`qrels.tsv`:

```tsv
query-id	corpus-id	score
q1	d1	1
q2	d2	1
```

`id` and `_id` are both accepted. Ragaroo validates missing files, malformed JSON, duplicate ids, empty text, and qrels that reference unknown queries or documents.

## Quickstart

```python
import ragaroo as roo

dataset = roo.Dataset.from_folder("data/nfcorpus")
embedder = roo.SentenceTransformerEmbedder("intfloat/e5-small-v2")

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
            retriever_1=roo.DenseRetriever(embedder=embedder, top_k=10),
            retriever_2=roo.BM25Retriever(top_k=10),
            top_k=10,
        ),
    ),
]

report = roo.Experiment(
    dataset=dataset,
    pipelines=pipelines,
    show_progress=True,
).run()

report.summary(sort_by="mrr@10")
```

## Common Patterns

Reranking with a cross-encoder:

```python
roo.Pipeline(
    name="dense_rerank",
    retriever=roo.DenseRetriever(embedder=embedder, top_k=50),
    reranker=roo.CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k=10,
    ),
)
```

Using a retriever as a reranker:

```python
roo.Pipeline(
    name="bm25_then_dense",
    retriever=roo.BM25Retriever(top_k=50),
    reranker=roo.DenseRetriever(embedder=embedder, top_k=10),
)
```

HyDE query augmentation:

```python
import os

provider = roo.OpenRouterProvider(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
)

roo.Pipeline(
    name="dense_hyde",
    retriever=roo.DenseRetriever(embedder=embedder, top_k=10),
    query_augmentation=[
        roo.HyDE(
            user_prompt="Write a concise support-style passage for this query.",
            system_prompt="Return only the passage.",
            temperature=0.3,
            max_tokens=220,
        )
    ],
    llm_provider=provider,
)
```

If `user_prompt`, `system_prompt`, or `temperature` is omitted, the transform uses its default. Custom `user_prompt` values are prepended to the dataset query.

By default, experiments evaluate all queries. While prototyping, pass `query_limit=50` or any other integer directly to `Experiment`.

## Metrics

Supported quality metrics:

- `recall@k`
- `precision@k`
- `mrr@k`
- `map@k`
- `hit_rate@k`
- `ndcg@k`

Supported latency metrics:

- `latency_ms`
- `query_augmentation_latency_ms`
- `retrieval_latency_ms`
- `rerank_latency_ms`
- `p50_latency_ms`
- `p95_latency_ms`
- `total_time_s`

`recall`, `precision`, `mrr`, `map`, and `hit_rate` treat qrel scores greater than zero as relevant. `ndcg` uses graded qrel scores.

## Experiment Outputs

Each experiment saves:

- `report.json`
- `report.csv`
- `manifest.json`
- `config.json`
- plots under `plots/`

With `store_query_results=True`, Ragaroo also saves per-query metrics and retrieved ids.

The manifest records dataset hashes, pipeline configs, pipeline hashes, dependency versions, platform metadata, git metadata when available, notes, tags, and random seed.

## Examples

See `examples/README.md`.

Included examples:

- `examples/compare_retrievers.py`
- `examples/compare_topk.py`
- `examples/compare_models.py`
- `examples/compare_rerank.py`
- `examples/compare_hyde.py`
- `examples/compare_multiple_datasets.py`

Most examples default to `data/nfcorpus`. Change the constants at the top of each script to use another dataset, model, cache folder, or query limit.

## Model Access

Use `HF_TOKEN` for gated Hugging Face models or higher Hub rate limits. Use `OPENROUTER_API_KEY` only for LLM-based query augmentation. Dataset paths, model choices, model cache folders, and query limits are regular Python arguments in scripts and examples, not environment variables.

To keep model downloads in a project-local folder:

```python
import ragaroo as roo

roo.store_models("./models")
```

## Limitations

- Ragaroo evaluates retrieval, not answer generation.
- It assumes the benchmark target is retrieving the right passages for each query.
- It currently loads dataset files into memory.
- Generated query augmentation can affect reproducibility unless the provider/model is controlled.

## Development Note

Ragaroo was built with human engineering work and assistance from LLM technologies. The code is tested, but LLM-assisted projects can still contain mistakes. Please validate benchmark setup, metrics, and outputs before relying on them for important decisions.

## Author

Ragaroo is created and maintained by **Marc Haraoui**.

## Citation

```bibtex
@software{haraoui_ragaroo,
  author = {Marc Haraoui},
  title = {Ragaroo: A Lightweight Retrieval Benchmarking Library for Custom Datasets},
  year = {2026},
  url = {https://github.com/marcharaoui/ragaroo}
}
```
