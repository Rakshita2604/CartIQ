# CartIQ

Grocery quick-commerce search and ranking engine: query understanding for short, misspelled
and Hinglish queries, hybrid BM25 + dense retrieval, store-aware inventory filtering,
personalized ranking, substitutions, and offline evaluation (NDCG@10, MRR, zero-result rate).

Status: **Phase 0** — scaffolding, configuration, domain model, health endpoint.

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs and manages Python 3.12 for this project)

## Quick start

```bash
uv sync                 # create .venv with pinned dependencies from uv.lock
make check              # ruff format check + ruff lint + mypy --strict + pytest
uv run cartiq serve     # http://127.0.0.1:8000/health, docs at /docs
```

## Configuration

Settings load from `config/default.toml`, then `CARTIQ_*` environment variables override them
(nested keys use `__`, e.g. `CARTIQ_API__PORT=9000`). Set `CARTIQ_CONFIG_FILE` to use a
different TOML file. Inspect the effective configuration with `uv run cartiq show-config`.

## Layout

```
config/            default configuration
src/cartiq/
  domain/          entities and value objects (no dependencies on other cartiq packages)
  api/             FastAPI app factory, routes, response schemas
  config.py        typed settings
  clock.py         injectable clock (FixedClock in tests)
  container.py     composition root that wires dependencies
  cli.py           `cartiq` command
tests/unit, tests/integration
```

Further packages (ingestion, query, retrieval, inventory, ranking, personalization,
substitutions, evaluation) are added in the phase that introduces them.
