# Relevance judgments

Hand-labeled query sets used by `cartiq evaluate`. The format is documented in
`src/cartiq/evaluation/judgments.py` (schema version 2).

Labels come from reading the catalog. They are never taken from, or adjusted to, the output
of a ranker under evaluation: a label changes only when the rubric or the catalog says it was
wrong, and such a change is its own commit.

## Catalog identity

Each file names its catalog (`catalog = "synthetic-v1"`) and records its fingerprint
(`catalog_sha256`): a SHA-256 over every product's relevance-bearing fields (id, name, brand,
category path, quantity, pack count, tags, description) in a fixed canonical form; see
`src/cartiq/catalog_fingerprint.py`. Price and MRP are excluded, so price changes do not
invalidate labels. `cartiq evaluate` refuses to run when either the name or the fingerprint
differs from the snapshot's catalog, and prints the actual fingerprint. After the catalog
changes, re-check the labels against it before updating `catalog_sha256`.

## Grades

A query's *stated constraints* are the brand, size or pack, and variant words it names
("toned", "powder", "brown"). Product type is what the shopper is buying (milk, atta, soap).

| Grade | List | Meaning |
|---|---|---|
| 3 | `exact` | Right product type, and every stated constraint holds. |
| 2 | `relevant` | A good alternative: right product type with exactly one stated constraint different (other size of the named brand, or the named size from another brand), a very close variant of the type (cookies for "biscuit"), or a product serving a minority interpretation. |
| 1 | `related` | An acceptable fallback a shopper might take: a neighbouring product type (pav for "bread", jaggery for "sugar"), two constraints off, or a fringe interpretation. |
| 0 | `irrelevant` | Explicitly judged not useful, including the same brand in a different product type (Amul butter for "amul milk") and lexical traps (Aloo Bhujia for "aloo"). |

Rules that follow from the "never silently relax" requirement:

- A product whose size is unknown never gets grade 3 on a query that states a size.
- A query naming a brand grades other brands at most 2, however close the product.
- A query with no grade 3 product must carry the `no_exact_match` attribute (and one with
  the attribute must have none), so an empty label is never an accident.

## Reasons

`[query.reasons]` maps a product id to a short explanation of why it got grade 0, 1 or 2.
Reasons are optional, never change a grade, and are rejected for grade 3 products and for
products that are not graded.

## Coverage: judged versus unjudged

- `coverage = "exhaustive"`: the labeler considered **every product in the exact catalog the
  fingerprint identifies**, independently of any retrieval or ranking system. Unlisted
  products are then judged 0. Use it only when that is literally true.
- `coverage = "partial"` (the default): unlisted products are *unjudged*. Only the
  `irrelevant` list holds confirmed negatives.

Unjudged products contribute zero gain, but metrics over partial queries are
**incomplete-judgment estimates**, not bounds: the ideal ranking may be missing unjudged
relevant products as well. The report marks such rows with `~`, shows Judged@10, and lists
unjudged products that were shown under `to_label` for the next labeling pass.

## Ambiguous queries

When several interpretations are plausible, list them instead of assuming one intent. Each
`[[query.interpretation]]` has a `name`, a `share` and the products that serve it:

| Share | Meaning | Highest grade |
|---|---|---|
| `dominant` | what most shoppers mean | 3 |
| `minority` | a real but smaller group of shoppers | 2 |
| `fringe` | possible, rarely intended | 1 |

Several interpretations may be dominant when intent is genuinely split. An interpretation
may have no products: "chicken" lists *chicken meat* (dominant, not in the catalog), which is
why it has no grade 3 product. Every product graded 1 or higher must belong to an
interpretation whose share allows its grade. Interpretations require the `multi_intent`
attribute, which the `ambiguous` segment implies, and there must be at least two.

## Segments and attributes

Each query has one primary segment (`head`, `misspelled`, `hinglish`, `brand`, `quantity`,
`ambiguous`), which partitions the set. Attributes are extra tags, and a query can have
several: `brand`, `quantity`, `transliteration`, `misspelling`, `multi_intent`,
`no_exact_match`. The primary segment implies its own attribute (for example, `hinglish`
implies `transliteration`). Metrics are reported per segment and per attribute; attribute
groups overlap.

## Metrics

A *hit* is a product graded at least `hit_min_grade` (default 2). Every query is evaluated
against every store in the snapshot. Relevance never depends on stock; stock only decides
what each store can show.

Retrieval stage (whole catalog, ignoring stock):

- **Recall@K** (K = 10, 100): hits in the top K candidates divided by the query's judged
  hits. Undefined when the query has no judged hits.
- **Catalog zero-result rate**: share of queries whose retrieval returns nothing.

Final stage (per store, then averaged per query):

- **NDCG@10**: DCG with gain `2^grade - 1` and discount `log2(rank + 1)`, divided by the DCG
  of the best possible top 10 from judged products in stock at that store. Undefined when no
  product graded 1 or higher is in stock there.
- **MRR@10**: 1 / rank of the first hit in the top 10; 0 when there is none. Undefined when
  no hit is in stock at that store.
- **Judged@10**: share of the top 10 results that carry a judgment. Undefined for an empty
  result list.
- **Available zero-result rate**: share of (query, store) pairs with no in-stock match
  before any relaxation or substitution.
- **Final zero-result rate**: share of pairs that show nothing after relaxation and
  substitution.

When retrieval returns nothing but the denominator is non-zero, NDCG, MRR and Recall are 0.
Undefined values are excluded from averages and counted in the report's `undef` column
(NDCG / MRR / Recall), never scored as 0 or 1. Per-store values are averaged per query; groups
average over queries, so every query weighs the same.

The expected orderings apply to the zero-result **rates**, not to result-list sizes: the final
rate should not exceed the available rate (fallbacks and substitutions only add results), and
the catalog rate should not exceed the available rate (stock only removes them). The report
warns when either is violated.

## Adding queries

1. Pick a query real shoppers would type and its primary segment; add attributes.
2. Grade products into `exact`, `relevant`, `related` and `irrelevant`, with reasons for
   grades 0–2 where helpful. List interpretations for multi-intent queries.
3. Mark the query `exhaustive` only if you considered every product in the catalog.
4. Run `make check`: the loader rejects malformed or inconsistent labels, and
   `cartiq evaluate` rejects a catalog mismatch.
