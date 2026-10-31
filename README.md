# CartIQ

Grocery quick-commerce search and ranking engine: query understanding for short, misspelled
and Hinglish queries, hybrid BM25 + dense retrieval, store-aware inventory filtering,
personalized ranking, substitutions, and offline evaluation (NDCG@10, MRR, zero-result rate).

Status: **Phase 4** — query understanding (quantities, Hinglish lexicon, spelling, plurals,
brands and aliases) and constraint-first ordering over BM25F retrieval, measured stage by
stage with the Phase 2 evaluation harness on a development set and a held-out set.

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
uv run cartiq evaluate snap-<id> --ranker bm25       # BM25F retrieval (stage S0)
uv run cartiq evaluate snap-<id> --ranker qu --stage s6   # + query understanding (s1..s6)
uv run cartiq evaluate-qu snap-<id> --stage s5       # quantity / brand / spelling vs expectations
uv run cartiq evaluate-tiers snap-<id>               # EXACT tier vs grade-3 labels (stage s6)
uv run cartiq evaluate snap-<id> --judgments eval/judgments/synthetic-v1-heldout.toml ...
uv run cartiq compare-reports BEFORE.json AFTER.json # overall, segment and per-query deltas
uv run cartiq bench-bm25 snap-<id>                   # index build time, query p50/p95
uv run cartiq bench-query snap-<id> --stage s4       # parser build/memory, parse latency
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

Stage s6 orders each store's 100 best eligible BM25F candidates EXACT first (brand, size or
pack, and every product concept satisfied), then RELAXED, both in BM25 order; scores and ranks
are unchanged. A product contains a concept when every word of one of the concept's
alternative phrases appears in its name, brand or product type (`ranking.coverage_fields`);
shelf text and descriptions still feed retrieval but never make a match exact. Its reports define available zero-result as "no in-stock EXACT product" (v2),
labelled in the report; earlier stages use v1 (no in-stock match) and are unchanged.

Future optimizations, recorded rather than done: cache the per-store eligibility mask used
by store-filtered BM25F search (it dominates that path's ~1 ms latency on BigBasket), and
persist the index to disk if build time (~1.2 s for 27k products) starts to matter.

## Query understanding and exact matches (Phase 4)

Stages are cumulative and evaluated separately (`--ranker qu --stage sN`; S0 is `--ranker bm25`):

| Stage | Adds |
|---|---|
| s1 | quantity extraction (size, pack, dozen, Hindi number words, ambiguous trailing count) |
| s2 | Hinglish lexicon expansion (frozen `hinglish-v1`) |
| s3 | spelling correction over name, brand and product-type words |
| s4 | singular/plural variants |
| s5 | brand and alias recognition (frozen `brand-aliases-v1`), brand protection, brand-first then Soundex tie-break, brand-share guard |
| s6 | constraint evaluation and EXACT-first ordering |

**What EXACT means.** A result is EXACT when it satisfies every constraint the parser found
in the query: the named brand, the stated size or pack count, and every product concept
(each concept's words, or one of its alternatives such as a Hinglish translation, spelling
correction or plural, found in the product's name, brand or product type). It does **not**
mean the product is the dominant interpretation of the query: Parachute coconut oil is an
EXACT match for "oil", Maggi ketchup for "maggi" and Everest Chicken Masala for "chicken",
because each contains exactly what was typed. RELAXED results list the constraints they
break (`brand`, `quantity`, `product`). Ordering puts EXACT before RELAXED and otherwise keeps
the BM25F order; it never changes BM25F scores or ranks.

**Frozen Phase 4b configuration** (stage s6 fingerprint `qu-s6-a9d842bd342f`, pinned by
`tests/unit/evaluation/test_pipelines.py`; any change to these settings changes it):

| Setting | Value |
|---|---|
| BM25F (`[retrieval.bm25]`) | k1 1.2; name 3.0/b 0.75, product type 2.0/0, brand 1.0/0, sub-category 0.5/0, description 0.3/0.75, category 0 (config id `7986afe23085`) |
| Lexicon / aliases | `hinglish-v1` (sha256 `6dbc8552…`), `brand-aliases-v1` (`34657058…`) |
| Spelling (`[query]`) | 1 edit from 4 characters, 2 from 8; dominance 20x and df >= 5; vocabulary from name, brand, product type |
| Brands (`[query]`) | one-word brand recognised when >= 50% of products containing the word have it in a brand; ties: brand words, then Soundex, then document frequency, then alphabetical |
| Quantity (`[query]`) | trailing bare count 2..50 |
| Ordering (`[ranking]`) | `candidate_depth` 100 store-eligible candidates; `coverage_fields` name, brand, product type |

Zero-result rates: *catalog* (retrieval finds nothing), *available* and *final*. For s6,
available means "no in-stock EXACT product" (v2, labelled in its reports); RELAXED results can
still be shown, which *final* counts. Earlier stages use v1 (no in-stock match).

### Known limitations (for semantic retrieval and ranking, Phases 5 and 7)

- **Words in names that mean a different product.** Coverage is lexical, so a product whose
  name or type contains the query's word is EXACT even when it is a different kind of
  product: tomato *ketchup* for "tomatto", *Aloo* Bhujia for "aloo", *Masala* noodles and chips
  for "masala", *Lemon*-flavoured tea, *Lime* soda and *Lemon* dishwash for "nimbu", and garlic
  for "adrak" because both share the product type "Ginger & Garlic". On the synthetic
  development set these are 14 of the 21 EXACT results not labelled grade 3; the other 7 are
  relevant but not the dominant interpretation (see above).
- **Names written only in Hindi.** On BigBasket, coriander powders sold as "Dhania" or
  "Dhaniya" say "coriander" only in their description, so for "coriander" (or a misspelling)
  they are RELAXED, not EXACT. Descriptions are deliberately not used for exactness: on the
  real catalog they mostly add noise (frying pans for "eggs", pickles for "dahi").
- **Words the catalog never uses are ignored for exactness.** A query word that no product's
  name, brand or product type contains (an unknown brand such as "Dabur", or "multigrain",
  "mozzarella", "strawberry" in a catalog without them) is not required, so the remaining
  words decide: "Dabur honey" makes a honey-flavoured tea EXACT. This keeps an uncorrectable
  typo from emptying the EXACT section, at the cost of over-generous EXACT results for
  out-of-catalog modifiers.
- **Descriptive words are required like any other.** "fresh paneer" and "brown bread" need
  "fresh" and "brown" in the product's name, brand or type, so Amul Malai Paneer and whole
  wheat bread are RELAXED for them even where they are the intended products.
- **Unknown sizes.** About 98.6% of BigBasket products have no parseable size, so size queries
  there usually have no EXACT result and fall back to RELAXED ones.
- **Totals.** A size is compared per unit, never as a pack total (6 x 200 ml is not 1.2 l).

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
  query/           query understanding; frozen lexicon and brand aliases in query/lexicon/
  retrieval/       BM25F index over product fields (in memory), concept-aware scoring
  ranking/         constraint evaluation and exact-first ordering (stage s6)
  evaluation/      judgments, metrics, runner, pipelines (baseline, BM25), report comparison
  snapshots.py     build and verify content-addressed snapshots
  api/             FastAPI app factory, routes, response schemas
  config.py        typed settings
  clock.py         injectable clock (FixedClock in tests)
  container.py     composition root that wires dependencies
  cli.py           `cartiq` command
tests/unit, tests/contract (run against every storage backend), tests/integration
```

Further packages (inventory, personalization, substitutions) are added in the phase that introduces them.
