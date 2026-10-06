"""HyperSlice command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import panel as pn
import typer

from hyperslice.exceptions import HyperSliceError
from hyperslice.explorer import Explorer, scattered_notice
from hyperslice.schema import SAMPLE_DIM, cardinality_table

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
    show_cardinality: Annotated[
        bool,
        typer.Option(
            "--show-cardinality",
            help="Print each output's cardinality (distinct values) before serving.",
        ),
    ] = False,
    points: Annotated[
        bool,
        typer.Option(
            "--points",
            help="Load design-point JSON as scattered samples (Filter tab only), even "
            "when it would fill a grid. Mostly-empty grids load this way already.",
        ),
    ] = False,
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
            layout="points" if points else "auto",
        )
    except HyperSliceError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    if explorer.schema.scattered:
        notice = scattered_notice(explorer.schema, int(explorer.dataset.sizes[SAMPLE_DIM]))
        typer.secho(f"Warning: {notice.replace('**', '')}", fg=typer.colors.YELLOW, err=True)
    if show_cardinality:
        typer.echo(cardinality_table(explorer.schema))
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
