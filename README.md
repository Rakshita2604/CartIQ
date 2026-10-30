# CartIQ

Grocery quick-commerce search and ranking engine: query understanding for short, misspelled
and Hinglish queries, hybrid BM25 + dense retrieval, store-aware inventory filtering,
personalized ranking, substitutions, and offline evaluation (NDCG@10, MRR, zero-result rate).

Status: **Phase 1** — catalog ingestion, seeded simulation, storage layer, versioned snapshots.

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

## Configuration

Settings load from `config/default.toml`, then `CARTIQ_*` environment variables override them
(nested keys use `__`, e.g. `CARTIQ_API__PORT=9000`). Set `CARTIQ_CONFIG_FILE` to use a
different TOML file. Inspect the effective configuration with `uv run cartiq show-config`.

## Layout

```
config/            default configuration
src/cartiq/
  domain/          entities and value objects (no dependencies on other cartiq packages)
  parsing/         text cleaning and quantity extraction ("6 x 200 ml")
  ingestion/       source adapters (synthetic, BigBasket) and catalog normalization
  simulation/      seeded stores, inventory, users and orders
  storage/         repository interfaces; in-memory and SQLite implementations
  snapshots.py     build and verify content-addressed snapshots
  api/             FastAPI app factory, routes, response schemas
  config.py        typed settings
  clock.py         injectable clock (FixedClock in tests)
  container.py     composition root that wires dependencies
  cli.py           `cartiq` command
tests/unit, tests/contract (run against every storage backend), tests/integration
```

Further packages (query, retrieval, inventory, ranking, personalization, substitutions,
evaluation) are added in the phase that introduces them.
