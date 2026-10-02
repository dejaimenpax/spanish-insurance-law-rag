# Evaluation

`questions.yaml` holds 177 Spanish questions over the phase 1 corpus:

| Category | Count | What it tests |
|---|---|---|
| `single` | 122 | Answer contained in one provision; every norm has 7+ questions |
| `literal` | 15 | The question cites the provision ("¿Qué dice el artículo 23 de la Ley 50/1980?") |
| `cross` | 20 | Needs two provisions, usually a law and its regulation |
| `out_of_scope` | 20 | Not covered by the indexed norms (other areas of law, non-existent provisions, EU law before phase 2): the assistant must abstain |

**Provenance and review.** Questions were drafted with an LLM from sampled provisions, and each
gold provision was checked to exist in the index (`insurance-rag eval validate`). Reference
answers were checked against the gold text for the doubtful cases. A full legal review is
pending, which is why the file is marked `review_status: draft`. Run
`insurance-rag eval validate --show` to print every reference answer next to its gold text.

**Gold matching** is at provision level: a retrieved chunk counts if it belongs to the gold
article or provision (any apartado).

## Running

```bash
docker compose up -d qdrant
uv run insurance-rag ingest                 # incremental; first run downloads and embeds
uv run insurance-rag eval validate
uv run insurance-rag eval retrieval --configs dense,sparse,hybrid,hybrid+refs,hybrid+refs+rerank
```

Reports are written to `eval/results/` with the git revision, models and dataset version.
