# Examples

These scripts are intended to show the common exploration patterns users care about.

## Available Scripts

- `compare_retrievers.py`
  - BM25 vs dense vs hybrid on one dataset.
- `compare_topk.py`
  - Compare the same retriever family with different `top_k` values.
- `compare_models.py`
  - Compare different embedding models under the same retrieval setup.
- `compare_rerank.py`
  - Compare dense retrieval with and without reranking.
- `compare_hyde.py`
  - Compare dense retrieval with and without HyDE query augmentation.
  - Requires `OPENROUTER_API_KEY` in `.env`.
- `compare_multiple_datasets.py`
  - Run the same benchmark recipe across more than one dataset folder.

## Setup

Most scripts assume:

```bash
uv sync --extra dev
```

and a local dataset folder such as:

```text
data/
  nfcorpus/
    corpus.jsonl
    queries.jsonl
    qrels.tsv
```

To use gated/private Hugging Face models, add `HF_TOKEN` to `.env`.
To use HyDE, also add `OPENROUTER_API_KEY`.

Most scripts default to `data/nfcorpus`. Change the constants at the top of each script to use another dataset, model, cache folder, or query limit.

## How To Use These Examples

Keep the scripts simple and edit them directly:

- change the dataset path
- change the embedding model
- compare different `top_k` values
- swap `flat`, `hnsw`, and `ivf`
- add or remove reranking
- add or remove query augmentation
- duplicate pipelines to run ablations

The intended workflow is:

1. Copy the closest example.
2. Change only the dataset path and the pipelines you care about.
3. Run the experiment.
4. Inspect `report.csv`, `manifest.json`, plots, and optional per-query outputs.
