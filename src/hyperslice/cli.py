"""HyperSlice command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import panel as pn
import typer

from hyperslice.exceptions import HyperSliceError
from hyperslice.explorer import Explorer

app = typer.Typer(help="Explore rectilinear N-dimensional xarray datasets.")


@app.command()
def main(
    dataset: Annotated[Path, typer.Argument(help="NetCDF file or Zarr store")],
    variable: Annotated[str | None, typer.Option("--variable")] = None,
    x_dim: Annotated[str | None, typer.Option("--x")] = None,
    y_dim: Annotated[str | None, typer.Option("--y")] = None,
    status_variable: Annotated[str | None, typer.Option("--status-variable")] = None,
    port: Annotated[int, typer.Option("--port")] = 0,
    address: Annotated[str, typer.Option("--address")] = "localhost",
    show: Annotated[bool, typer.Option("--show/--no-browser")] = True,
) -> None:
    """Launch a local HyperSlice server."""
    logging.basicConfig(level=logging.INFO)
    try:
        explorer = Explorer(
            dataset,
            default_variable=variable,
            default_x=x_dim,
            default_y=y_dim,
            status_variable=status_variable,
        )
    except HyperSliceError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Serving HyperSlice for {dataset}")
    pn.serve(
        {"/": explorer.view},
        port=port,
        address=address,
        show=show,
        title="HyperSlice",
    )


if __name__ == "__main__":
    app()
