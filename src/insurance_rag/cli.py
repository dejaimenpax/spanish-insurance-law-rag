"""Command-line entry point."""

from datetime import date
from typing import TYPE_CHECKING, Annotated, cast

import typer

if TYPE_CHECKING:
    from insurance_rag.domain.models import Chunk, NormSpec
    from insurance_rag.index.qdrant_store import SearchMode

app = typer.Typer(help="Spanish insurance law RAG.", no_args_is_help=True)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("insurance_rag.api.app:create_app", factory=True, host=host, port=port)


@app.command()
def ingest(
    norm: Annotated[
        list[str] | None, typer.Option(help="Norm id (BOE-A-…) to ingest; repeatable.")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="Parse and chunk from the raw cache, print statistics, index nothing."),
    ] = False,
    force: Annotated[
        bool, typer.Option(help="Re-process norms even if the BOE index is unchanged.")
    ] = False,
    recreate: Annotated[bool, typer.Option(help="Drop and rebuild the vector collection.")] = False,
) -> None:
    """Download, parse, chunk and index the catalog norms (incremental by default)."""
    from insurance_rag.config import get_settings
    from insurance_rag.corpus.catalog import load_catalog
    from insurance_rag.observability.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    catalog = load_catalog()
    specs = [s for s in catalog.norms if not norm or s.id in norm]
    if dry_run:
        _ingest_dry_run(specs)
        return

    from insurance_rag.embeddings.sparse import Bm25Encoder
    from insurance_rag.ingestion.boe.client import BoeClient
    from insurance_rag.ingestion.indexer import Indexer
    from insurance_rag.ingestion.pipeline import RawStore
    from insurance_rag.ingestion.state import IngestionState
    from insurance_rag.wiring import build_embedder, build_store

    with BoeClient() as client, IngestionState(settings.state_path) as state:
        indexer = Indexer(
            client=client,
            raw_store=RawStore(settings.raw_dir),
            state=state,
            store=build_store(settings),
            embedder=build_embedder(settings),
            bm25=Bm25Encoder(),
            max_chars=settings.chunk_max_chars,
        )
        indexer.prepare(recreate=recreate)
        reports = indexer.sync(
            specs,
            as_of=date.today(),
            force=force,
            catalog_ids=[s.id for s in catalog.norms],
        )

    typer.echo(
        f"{'norm':<11}{'status':<9}{'blocks':>7}{'removed':>9}{'chunks':>8}{'embedded':>10}  reason"
    )
    for r in reports:
        typer.echo(
            f"{r.short_name:<11}{r.status:<9}{r.blocks_changed:>7}{r.blocks_removed:>9}"
            f"{r.chunks_indexed:>8}{r.embeddings_computed:>10}  {r.reason}"
        )


def _ingest_dry_run(specs: list["NormSpec"]) -> None:
    from insurance_rag.config import get_settings
    from insurance_rag.ingestion.boe.client import BoeClient
    from insurance_rag.ingestion.pipeline import RawStore, fetch_snapshot, process_norm

    settings = get_settings()
    store = RawStore(settings.raw_dir)
    header = f"{'norm':<11}{'arts':>6}{'DA':>5}{'DT':>5}{'DD':>5}{'DF':>5}{'annex':>7}"
    typer.echo(f"{header}{'repealed':>10}{'chunks':>8}{'max_chars':>11}  consolidated")
    with BoeClient() as client:
        for spec in specs:
            snapshot = fetch_snapshot(client, store, spec.id)
            result = process_norm(
                spec, snapshot, as_of=date.today(), max_chars=settings.chunk_max_chars
            )
            st = result.stats
            typer.echo(
                f"{spec.short_name:<11}{st['articulo']:>6}{st['disposicion_adicional']:>5}"
                f"{st['disposicion_transitoria']:>5}{st['disposicion_derogatoria']:>5}"
                f"{st['disposicion_final']:>5}{st['anexo']:>7}{st['repealed']:>10}"
                f"{st['chunks']:>8}{st['max_chunk_chars']:>11}  {result.norm.consolidated_as_of}"
            )


@app.command()
def search(
    query: str,
    k: Annotated[int, typer.Option(help="Number of results.")] = 8,
    mode: Annotated[str, typer.Option(help="dense, sparse or hybrid.")] = "hybrid",
    norm: Annotated[list[str] | None, typer.Option(help="Restrict to these norm ids.")] = None,
) -> None:
    """Run retrieval only and print the ranked chunks."""
    from insurance_rag.config import get_settings
    from insurance_rag.index.qdrant_store import ChunkFilter
    from insurance_rag.observability.logging import configure_logging
    from insurance_rag.wiring import build_retriever

    settings = get_settings()
    configure_logging("WARNING", json=False)
    retriever = build_retriever(settings)
    if mode not in ("dense", "sparse", "hybrid"):
        raise typer.BadParameter("mode must be dense, sparse or hybrid")
    result = retriever.retrieve(
        query,
        k=k,
        mode=cast("SearchMode", mode),
        chunk_filter=ChunkFilter(norm_ids=tuple(norm or ())),
    )
    for rank, hit in enumerate(result.hits, start=1):
        c = hit.chunk
        typer.echo(
            f"{rank:>2}. [{hit.score:.3f} {'+'.join(hit.sources)}] {c.citation_label}"
            f" — {c.heading or ''}\n    {c.text[:160]!r}"
        )
    typer.echo(f"timings (ms): {result.timings_ms}")


@app.command()
def ask(
    question: str,
    norm: Annotated[list[str] | None, typer.Option(help="Restrict to these norm ids.")] = None,
) -> None:
    """Answer a question in the terminal, streaming the text and listing the cited sources."""
    import asyncio

    from insurance_rag.config import get_settings
    from insurance_rag.generation.answer import (
        CitationEvent,
        DeltaEvent,
        DoneEvent,
        SourcesEvent,
        WarningEvent,
    )
    from insurance_rag.index.qdrant_store import ChunkFilter
    from insurance_rag.observability.logging import configure_logging
    from insurance_rag.wiring import build_answer_service, build_retriever

    settings = get_settings()
    configure_logging("WARNING", json=False)
    service = build_answer_service(settings, build_retriever(settings))

    async def run() -> None:
        sources = []
        async for event in service.stream(
            question, chunk_filter=ChunkFilter(norm_ids=tuple(norm or ()))
        ):
            if isinstance(event, SourcesEvent):
                sources = event.sources
            elif isinstance(event, DeltaEvent):
                typer.echo(event.text, nl=False)
            elif isinstance(event, CitationEvent):
                typer.echo(f" [{event.citation.source_index + 1}]", nl=False)
            elif isinstance(event, WarningEvent):
                typer.secho(f"\n⚠ {event.message}", fg="yellow")
            elif isinstance(event, DoneEvent):
                typer.echo("\n")
                for i in event.cited_sources:
                    s = sources[i]
                    typer.echo(
                        f"[{i + 1}] {s.citation} (consolidado a {s.consolidated_as_of}) {s.url}"
                    )
                typer.secho(
                    f"\n{event.model} · {event.usage} · ${event.cost_usd} · "
                    f"{event.timings_ms.get('total')} ms",
                    dim=True,
                )

    asyncio.run(run())


eval_app = typer.Typer(help="Evaluation commands.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
DEFAULT_DATASET = "eval/questions.yaml"


@eval_app.command("validate")
def eval_validate(
    dataset: Annotated[str, typer.Option(help="Path to the questions file.")] = DEFAULT_DATASET,
    show: Annotated[
        bool, typer.Option(help="Print gold text next to each reference answer.")
    ] = False,
) -> None:
    """Check that every gold provision exists in the index."""
    from pathlib import Path

    from insurance_rag.config import get_settings
    from insurance_rag.corpus.catalog import load_catalog
    from insurance_rag.evaluation.dataset import load_dataset
    from insurance_rag.observability.logging import configure_logging
    from insurance_rag.wiring import build_store

    configure_logging("WARNING", json=False)
    catalog = load_catalog()
    data = load_dataset(Path(dataset), catalog)
    store = build_store(get_settings())
    missing = 0
    for q in data.questions:
        for key in q.gold:
            chunks = (
                [
                    c
                    for c in store.find_provision(
                        kind=key.kind, number=key.number, norm_ids=[key.norm_id], limit=50
                    )
                ]
                if key.number
                else store.find_by_block(key.norm_id, key.block_id or "")
            )
            if not chunks:
                missing += 1
                typer.echo(f"MISSING {q.id}: {key.label(catalog)}")
            elif show:
                typer.echo(f"\n[{q.id}] {q.question}\n  expected: {q.answer}")
                typer.echo(f"  gold {key.label(catalog)}: {' '.join(c.text for c in chunks)[:700]}")
    typer.echo(f"{len(data.questions)} questions, {missing} missing gold provisions")
    if missing:
        raise typer.Exit(1)


@eval_app.command("retrieval")
def eval_retrieval(
    dataset: Annotated[str, typer.Option(help="Path to the questions file.")] = DEFAULT_DATASET,
    configs: Annotated[
        str,
        typer.Option(
            help="Comma-separated: dense, sparse, hybrid, hybrid+refs, hybrid+refs+rerank."
        ),
    ] = "dense,sparse,hybrid,hybrid+refs",
    out_dir: Annotated[str, typer.Option(help="Where to write the JSON report.")] = "eval/results",
) -> None:
    """Measure recall@k and MRR of each retrieval configuration."""
    from pathlib import Path

    from insurance_rag.config import get_settings
    from insurance_rag.corpus.catalog import load_catalog
    from insurance_rag.evaluation.dataset import load_dataset
    from insurance_rag.evaluation.runner import (
        SearchFn,
        evaluate_retrieval,
        markdown_table,
        write_report,
    )
    from insurance_rag.observability.logging import configure_logging
    from insurance_rag.wiring import build_reranker, build_retriever

    configure_logging("WARNING", json=False)
    settings = get_settings()
    data = load_dataset(Path(dataset), load_catalog())
    retriever = build_retriever(settings, with_reranker=False)

    def make(mode: str, refs: bool, rerank: bool) -> SearchFn:
        def search(query: str) -> list["Chunk"]:
            retriever.reranker = reranker if rerank else None
            result = retriever.retrieve(
                query, k=10, mode=cast("SearchMode", mode), use_references=refs
            )
            return [h.chunk for h in result.hits]

        return search

    names = [c.strip() for c in configs.split(",") if c.strip()]
    reranker = build_reranker(settings) if any("rerank" in n for n in names) else None
    if any("rerank" in n for n in names) and reranker is None:
        raise typer.BadParameter("rerank configs need RERANKER_MODEL to be set")
    fns = {}
    for name in names:
        parts = name.split("+")
        fns[name] = make(parts[0], "refs" in parts, "rerank" in parts)
    results = evaluate_retrieval(data, fns)
    path = write_report(
        results,
        Path(out_dir),
        prefix="retrieval",
        metadata={
            "dataset_version": data.version,
            "review_status": data.review_status,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model
            or ("BAAI/bge-reranker-v2-m3" if reranker else None),
            "questions": len(data.questions),
        },
    )
    typer.echo(markdown_table(results))
    for r in results:
        typer.echo(
            f"\n{r.name}: "
            + ", ".join(
                f"{c}: MRR {m['mrr']:.3f} R@5 {m['recall@5']:.3f} (n={m['n']})"
                for c, m in r.by_category.items()
            )
        )
    typer.echo(f"\nreport: {path}")


@app.command()
def version() -> None:
    """Print the package version."""
    from insurance_rag import __version__

    typer.echo(__version__)


def main() -> None:
    app()
