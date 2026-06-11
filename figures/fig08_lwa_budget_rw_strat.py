"""Paper Fig. 8 (RW stratification): ERA5 LWA-budget Hovmoller, WP+NA
pooled, RW case vs no-RW case (Quinting & Jones 2016 quintiles)."""

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
        classification_file: Annotated[Path, typer.Option(
            help="RW classification CSV (storm_id, reference, wb_group "
                 "with rwcase/norwcase)")],
        budget_strip_directory: Annotated[Path, typer.Option(
            help="Directory with budget_strips_era5_YYYY_MM.nc and "
                 "lwa_budget_climatology_era5.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {reference} is substituted")]
        = "fig08_lwa_budget_era5_wp_na_rw_strat_{reference}.png",
        tracks_directory: Annotated[Optional[Path], typer.Option(
            help="Track database directory for the mean-track overlay "
                 "(default: parent of --tracks-csv)")] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature, et, or both")] = "recurvature",
        n_mc: Annotated[int, typer.Option()] = 400,
        n_workers: Annotated[Optional[int], typer.Option()] = None,
        year_start: Annotated[int, typer.Option()] = 2000,
        year_end: Annotated[int, typer.Option()] = 2022,
        sigma_lag: Annotated[float, typer.Option(
            help="Gaussian sigma along lag (6h grid)")] = 1.0,
        sigma_lon: Annotated[float, typer.Option(
            help="Gaussian sigma along longitude (1 deg/cell)")] = 2.5,
        n_bootstrap: Annotated[int, typer.Option(
            help="Bootstrap-of-differences resamples for high-vs-low "
                 "hatching; 0 falls back to per-group MC envelope")] = 300,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    workers = n_workers if n_workers is not None else _default_worker_count()

    cls = Path(classification_file)
    if not cls.is_file():
        raise SystemExit(f"Classification CSV not found: {cls}")

    storms_df = pd.read_csv(
        tracks_csv,
        parse_dates=["recurv_time", "et_time"],
        keep_default_na=False, na_values=[""],
    )
    td = (tracks_directory if tracks_directory is not None
          else Path(tracks_csv).resolve().parent)

    refs = (["recurvature", "et"] if reference == "both" else [reference])
    fig_dir = Path(output_directory)
    fig_dir.mkdir(parents=True, exist_ok=True)

    smooth_sigma = (float(sigma_lag), float(sigma_lon))

    for ref in refs:
        out = budget_hovmoller.plot_budget_wp_na_wb_strat_high_low(
            storms_df=storms_df,
            strip_dir=budget_strip_directory,
            classification_csv=cls,
            reference=ref,
            n_mc=n_mc,
            n_workers=workers,
            smooth_sigma=smooth_sigma,
            year_start=year_start,
            year_end=year_end,
            tracks_dir=td,
            n_bootstrap_diff=int(n_bootstrap),
            group_high="rwcase",
            group_low="norwcase",
            title_high=("RW case (top quintile by Q&J 2016 score) "
                        "\N{EM DASH} WP+NA pooled"),
            title_low=("no-RW case (bottom quintile) "
                       "\N{EM DASH} WP+NA pooled"),
            footer_label_high="RW case",
            footer_label_low="no-RW case",
            out_path=fig_dir / figure_name.format(reference=ref),
        )
        print(f"Paper copy: {out}")


if __name__ == "__main__":
    typer.run(main)
