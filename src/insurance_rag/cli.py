"""Command-line entry point."""

from datetime import date
from typing import TYPE_CHECKING, Annotated, cast

import typer

if TYPE_CHECKING:
    from insurance_rag.domain.models import NormSpec
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
def version() -> None:
    """Print the package version."""
    from insurance_rag import __version__

    typer.echo(__version__)


def main() -> None:
    app()
