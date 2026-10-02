"""Command-line entry point."""

import typer

app = typer.Typer(help="Spanish insurance law RAG.", no_args_is_help=True)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("insurance_rag.api.app:create_app", factory=True, host=host, port=port)


@app.command()
def version() -> None:
    """Print the package version."""
    from insurance_rag import __version__

    typer.echo(__version__)


def main() -> None:
    app()
