# CartIQ

Grocery quick-commerce search and ranking engine: query understanding for short, misspelled
and Hinglish queries, hybrid BM25 + dense retrieval, store-aware inventory filtering,
personalized ranking, substitutions, and offline evaluation (NDCG@10, MRR, zero-result rate).

Status: **Phase 3** — BM25F lexical retrieval, measured with the Phase 2 evaluation harness
on top of Phase 1's catalog ingestion, seeded simulation, storage layer and snapshots.

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs and manages Python 3.12 for this project)

## Quick start

```bash
uv sync                 # create .venv with pinned dependencies from uv.lock
make check              # ruff format check + ruff lint + mypy --strict + pytest
uv run cartiq serve     # http://127.0.0.1:8000/health, docs at /docs
```

## Data snapshots

```bash
uv run cartiq build-snapshot                       # synthetic catalog (default, no download)
uv run cartiq build-snapshot --source bigbasket    # reads ingestion.bigbasket_csv
uv run cartiq verify-snapshot snap-<id>            # recompute and compare the content hash
```

A snapshot (`artifacts/snapshots/snap-<hash>/`) holds `cartiq.sqlite` (catalog, stores,
inventory, users, orders) and `manifest.json` (source hash, simulation settings, ingestion
report, counts). The id is a hash of the content, so the same inputs always give the same id.

**BigBasket data** comes from Kaggle's
[BigBasket Entire Product List](https://www.kaggle.com/datasets/surajjha101/bigbasket-entire-product-list-28k-datapoints),
licensed **CC BY-NC-SA 4.0** (non-commercial, attribution, share-alike). It is not
redistributed here: download `BigBasket Products.csv` to `data/raw/bigbasket_products.csv`.
Snapshots built from it are adaptations under the same license; `data/` and `artifacts/`
are gitignored.

## Evaluation

```bash
uv run cartiq build-snapshot                         # prints snap-<id>
uv run cartiq evaluate snap-<id>                     # token-overlap baseline
uv run cartiq evaluate snap-<id> --ranker bm25       # BM25F retrieval
uv run cartiq compare-reports BEFORE.json AFTER.json # overall, segment and per-query deltas
uv run cartiq bench-bm25 snap-<id>                   # index build time, query p50/p95
```

Judgments are hand-labeled TOML files under `eval/judgments/`, bound to one catalog version
and fingerprint; the grading rubric and metric definitions are in
[`eval/README.md`](eval/README.md). Reports are written to
`artifacts/eval/<snapshot>/<pipeline>-<judgments hash>.json` and contain no timestamps, so
the same inputs always give the same report. The BM25 pipeline name includes a hash of its
configuration and tokenizer. `compare-reports` refuses reports made under different
conditions (snapshot, catalog, judgments, cut-offs or stores). The token-overlap report is
pinned byte for byte by `tests/integration/test_baseline_regression.py`.

BM25F (formula in `src/cartiq/retrieval/bm25f.py`) weights product name, product type,
brand, sub-category and description separately; field weights live in `[retrieval.bm25]`
and were fixed before evaluation, not tuned on the judgments. The classic `(k1 + 1)` factor
is omitted: absolute scores are scaled down by that constant, rankings are unchanged.

Future optimizations, recorded rather than done: cache the per-store eligibility mask used
by store-filtered BM25F search (it dominates that path's ~1 ms latency on BigBasket), and
persist the index to disk if build time (~1.2 s for 27k products) starts to matter.

## Configuration

Settings load from `config/default.toml`, then `CARTIQ_*` environment variables override them
(nested keys use `__`, e.g. `CARTIQ_API__PORT=9000`). Set `CARTIQ_CONFIG_FILE` to use a
different TOML file. Inspect the effective configuration with `uv run cartiq show-config`.

## Layout

```
config/            default configuration
eval/judgments/    hand-labeled relevance judgments (rubric in eval/README.md)
src/cartiq/
  domain/          entities and value objects (no dependencies on other cartiq packages)
  parsing/         text cleaning, tokenization, quantity extraction ("6 x 200 ml")
  ingestion/       source adapters (synthetic, BigBasket) and catalog normalization
  simulation/      seeded stores, inventory, users and orders
  storage/         repository interfaces; in-memory and SQLite implementations
  retrieval/       BM25F index over product fields (in memory)
  evaluation/      judgments, metrics, runner, pipelines (baseline, BM25), report comparison
  snapshots.py     build and verify content-addressed snapshots
  api/             FastAPI app factory, routes, response schemas
  config.py        typed settings
  clock.py         injectable clock (FixedClock in tests)
  container.py     composition root that wires dependencies
  cli.py           `cartiq` command
tests/unit, tests/contract (run against every storage backend), tests/integration
```

Further packages (query, inventory, ranking, personalization, substitutions) are added in the phase that introduces them.
