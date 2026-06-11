"""Paper Fig. 4: WP | NA Hovmoller composite for the non-conservative row,
with the ERA5 non-QG source plus the MERRA-2 diabatic source decomposition."""

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
            help="Directory with budget_strips_<source>_YYYY_MM.nc and "
                 "lwa_budget_climatology_<source>.nc for era5 and merra2")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {reference} is substituted")]
        = "fig4_lwa_merra2_wp_na_{reference}.png",
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
        out = budget_hovmoller.plot_fig4_alt_nonqg_wp_na(
            storms_df=storms_df,
            strip_dir=budget_strip_directory,
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
