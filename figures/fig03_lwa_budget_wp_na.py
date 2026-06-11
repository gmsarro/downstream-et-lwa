"""Paper Fig. 3: ERA5 LWA-budget Hovmoller composite, WP | NA in one 3x4
grid with minimap row, shared budget colorbar, and a vertical separator."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from typing_extensions import Annotated

import downstream_et_lwa.plotting.budget_hovmoller as budget_hovmoller

_LOG = logging.getLogger(__name__)


def _default_worker_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 8)


def main(
        tracks_csv: Annotated[Path, typer.Option(
            help="recurving_nh_tracks.csv from the track database")],
        budget_strip_directory: Annotated[Path, typer.Option(
            help="Directory with budget_strips_era5_YYYY_MM.nc and "
                 "lwa_budget_climatology_era5.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {reference} is substituted")]
        = "fig3_lwa_budget_era5_wp_na_{reference}.png",
        tracks_directory: Annotated[Optional[Path], typer.Option(
            help="Track database directory for the mean-track overlay "
                 "(default: parent of --tracks-csv)")] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature, et, or both")] = "recurvature",
        n_mc: Annotated[int, typer.Option()] = 300,
        n_workers: Annotated[Optional[int], typer.Option()] = None,
        year_start: Annotated[int, typer.Option()] = 2000,
        year_end: Annotated[int, typer.Option()] = 2022,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    workers = n_workers if n_workers is not None else _default_worker_count()

    storms_df = pd.read_csv(
        tracks_csv,
        parse_dates=["recurv_time", "et_time"],
        keep_default_na=False,
        na_values=[""],
    )
    refs = (["recurvature", "et"] if reference == "both" else [reference])
    fig_dir = Path(output_directory)
    fig_dir.mkdir(parents=True, exist_ok=True)
    td = (tracks_directory if tracks_directory is not None
          else Path(tracks_csv).resolve().parent)

    for ref in refs:
        out = budget_hovmoller.plot_budget_wp_na_combined(
            storms_df=storms_df,
            strip_dir=budget_strip_directory,
            source="era5",
            reference=ref,
            n_mc=n_mc,
            n_workers=workers,
            year_start=year_start,
            year_end=year_end,
            tracks_dir=td,
            out_path=fig_dir / figure_name.format(reference=ref),
        )
        print(f"Paper copy: {out}")


if __name__ == "__main__":
    typer.run(main)
