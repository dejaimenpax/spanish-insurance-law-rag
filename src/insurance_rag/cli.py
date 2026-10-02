"""Command-line entry point."""

from datetime import date
from typing import Annotated

import typer

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
        bool, typer.Option(help="Parse and chunk, print statistics, write nothing.")
    ] = False,
    refresh_raw: Annotated[
        bool, typer.Option(help="Re-download texts even if they are cached.")
    ] = False,
) -> None:
    """Download, parse and chunk the norms in the catalog."""
    from insurance_rag.config import get_settings
    from insurance_rag.corpus.catalog import load_catalog
    from insurance_rag.ingestion.boe.client import BoeClient
    from insurance_rag.ingestion.pipeline import (
        RawStore,
        fetch_snapshot,
        process_norm,
        write_chunks,
    )
    from insurance_rag.observability.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level, json=False)
    specs = [s for s in load_catalog().norms if not norm or s.id in norm]
    store = RawStore(settings.data_dir / "raw")
    today = date.today()

    header = f"{'norm':<11}{'arts':>6}{'DA':>5}{'DT':>5}{'DD':>5}{'DF':>5}{'annex':>7}"
    typer.echo(f"{header}{'repealed':>10}{'chunks':>8}{'max_chars':>11}  consolidated")
    with BoeClient() as client:
        for spec in specs:
            snapshot = fetch_snapshot(client, store, spec.id, refresh=refresh_raw)
            result = process_norm(spec, snapshot, as_of=today)
            st = result.stats
            typer.echo(
                f"{spec.short_name:<11}{st['articulo']:>6}{st['disposicion_adicional']:>5}"
                f"{st['disposicion_transitoria']:>5}{st['disposicion_derogatoria']:>5}"
                f"{st['disposicion_final']:>5}{st['anexo']:>7}{st['repealed']:>10}"
                f"{st['chunks']:>8}{st['max_chunk_chars']:>11}  {result.norm.consolidated_as_of}"
            )
            if not dry_run:
                write_chunks(result, settings.data_dir / "chunks")


@app.command()
def version() -> None:
    """Print the package version."""
    from insurance_rag import __version__

    typer.echo(__version__)


def main() -> None:
    app()
