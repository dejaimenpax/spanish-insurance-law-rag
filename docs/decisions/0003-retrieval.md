# 0003 — Retrieval: hybrid search, literal references and reranking

**Status:** accepted · **Date:** 2026-10-02

## Decision

1. **Dense + BM25 fused with RRF in Qdrant.** Dense vectors come from `BAAI/bge-m3` (local,
   multilingual, 8k context, MIT). BM25 is stored as a sparse vector with Qdrant's IDF modifier,
   using a Spanish Snowball stemmer and a tokenizer that keeps numbers and references
   ("20/2015", "bis"). One database, no extra search service.
2. **Literal reference lookup.** Queries that cite a provision ("artículo 23 de la Ley 50/1980",
   "DT 4.ª del RDL 3/2020") are parsed and the provision is fetched by its coordinates and placed
   first. Neither embeddings nor BM25 handle these reliably: "23" appears in hundreds of
   cross-references.
3. **Cross-encoder reranking** (`BAAI/bge-reranker-v2-m3`) of the top 10 fused candidates,
   truncated to 512 tokens. On by default; set `RERANKER_MODEL=` to disable it.

## Evidence

Evaluation set v1 (157 answerable questions; see `eval/README.md`), provision-level metrics:

| Config | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| dense | 0.621 | 0.803 | 0.838 | 0.892 | 0.755 |
| sparse (BM25) | 0.462 | 0.627 | 0.701 | 0.818 | 0.589 |
| hybrid | 0.627 | 0.793 | 0.873 | 0.920 | 0.762 |
| hybrid + references | 0.697 | 0.860 | 0.943 | 0.978 | 0.831 |
| hybrid + references + rerank | 0.799 | 0.943 | 0.968 | 0.978 | 0.910 |

* Literal questions: MRR 0.25 with hybrid search alone, 1.00 with reference lookup.
* Cross-norm questions (law + regulation): R@5 0.80 → 0.90 with reranking.
* Reranker size: 10, 20 and 50 candidates gave the same MRR (0.910, 0.910, 0.902 at 1,024
  tokens), so the smallest pool was kept.

Reranking latency per query (10 candidates, 512 tokens): ~0.7 s on Apple MPS, ~2.5 s on CPU
(the Docker image). Dense query embedding is ~40 ms; Qdrant search ~10 ms. The reranker is worth
its cost here because answer generation takes several seconds anyway and the first retrieved
provision is the one the answer leans on.

Runs on Apple MPS vary by about ±0.01 in hybrid metrics because of floating-point
non-determinism in embeddings.
