"""MPAS current vs future zonal-mean LWA-budget Hovmoller, WP storms only
(paper Fig. 13)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

import downstream_et_lwa.plotting.mpas_zonal_budget as mpas_zonal_budget


def main(
        mpas_composites_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_recurvature_mpas_<scenario>_"
                 "WP.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option()] = (
            "fig13_mpas_lwa_budget_zonal_wp_only_recurvature.png"),
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    out = mpas_zonal_budget.plot_mpas_fig10_zonal_budget(
        composite_dir=mpas_composites_directory,
        out_path=Path(output_directory) / figure_name,
        basin="WP",
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    typer.run(main)
