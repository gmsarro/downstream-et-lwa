"""Paper Fig. 7: WB-stratified version of Fig. 2; the 3-column Hovmoller
row repeated for each basin x RWB-quintile (WP/NA, high/low RWB)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.cm
import matplotlib.colors
import pandas as pd
import typer
from typing_extensions import Annotated

import downstream_et_lwa.plotting.budget_hovmoller as budget_hovmoller
import downstream_et_lwa.plotting.lwa_hovmoller as lwa_hovmoller
import downstream_et_lwa.plotting.qj_hovmoller as qj_hovmoller
import downstream_et_lwa.plotting.rwp_lwa_rows as rwp_lwa_rows
import downstream_et_lwa.tracks as tracks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)


def _default_worker_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 8)


def main(
        tracks_directory: Annotated[Path, typer.Option(
            help="Track database directory with recurving_nh_tracks.csv "
                 "and individual/")],
        classification_file: Annotated[Path, typer.Option(
            help="WB classification CSV (storm_id, reference, wb_group)")],
        rwp_directory: Annotated[Path, typer.Option(
            help="Directory with rwp_envelope_YYYY_MM.nc")],
        lwa_directory: Annotated[Path, typer.Option(
            help="Directory with lwa_filtered_YYYY_MM.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {reference} is substituted")]
        = "fig07_rwp_lwa_wp_na_wb_strat_{reference}.png",
        rwp_climatology_path: Annotated[Optional[Path], typer.Option(
            help="rwp_climatology.nc (default: "
                 "<rwp-directory>/rwp_climatology.nc)")] = None,
        lwa_climatology_path: Annotated[Optional[Path], typer.Option(
            help="lwa_climatology.nc (default: "
                 "<lwa-directory>/lwa_climatology.nc)")] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature or et")] = "recurvature",
        n_mc: Annotated[int, typer.Option()] = 400,
        n_workers: Annotated[Optional[int], typer.Option()] = None,
        rwp_strip_year_start: Annotated[int, typer.Option()] = 2000,
        rwp_strip_year_end: Annotated[int, typer.Option()] = 2022,
        lwa_year_start: Annotated[int, typer.Option()] = 2000,
        lwa_year_end: Annotated[int, typer.Option()] = 2022,
        tau_star: Annotated[float, typer.Option()] = 1.9,
        mc_smoothed_envelope: Annotated[bool, typer.Option()] = False,
        mc_legacy_envelope: Annotated[bool, typer.Option(
            hidden=True)] = False,
        sigma_lag: Annotated[Optional[float], typer.Option()] = None,
        sigma_lon: Annotated[Optional[float], typer.Option()] = None,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if reference not in ("recurvature", "et"):
        raise typer.BadParameter("reference must be recurvature or et")
    workers = n_workers if n_workers is not None else _default_worker_count()

    cls = Path(classification_file)
    if not cls.is_file():
        raise SystemExit(f"Classification CSV not found: {cls}")

    storms_path = Path(tracks_directory) / "recurving_nh_tracks.csv"
    storms_df = pd.read_csv(
        storms_path,
        parse_dates=["recurv_time", "et_time"],
        keep_default_na=False,
        na_values=[""],
    )
    _, full_tracks = tracks.load_track_database(
        tracks_directory=tracks_directory)

    ids_hi = budget_hovmoller.wb_strat_storm_ids(
        classification_csv=cls, reference=reference, wb_group="highwb")
    ids_lo = budget_hovmoller.wb_strat_storm_ids(
        classification_csv=cls, reference=reference, wb_group="lowwb")
    storms_hi = storms_df[storms_df["storm_id"].isin(ids_hi)].copy()
    storms_lo = storms_df[storms_df["storm_id"].isin(ids_lo)].copy()

    clim_rwp = (rwp_climatology_path if rwp_climatology_path is not None
                else Path(rwp_directory) / "rwp_climatology.nc")
    clim_lwa = (lwa_climatology_path if lwa_climatology_path is not None
                else Path(lwa_directory) / "lwa_climatology.nc")

    qj_hovmoller.load_all_strips(
        rwp_dir=rwp_directory,
        year_start=rwp_strip_year_start,
        year_end=rwp_strip_year_end,
    )
    lwa_hovmoller.load_all_strips(
        lwa_dir=lwa_directory,
        year_start=lwa_year_start,
        year_end=lwa_year_end,
        tau_star=tau_star,
    )

    fig = plt.figure(figsize=(18.5, 24.5))
    gs = fig.add_gridspec(
        9, 3,
        height_ratios=[
            0.45, 1.45,
            0.45, 1.45,
            0.45, 1.45,
            0.45, 1.45,
            0.22,
        ],
        hspace=0.55,
        wspace=0.26,
        left=0.055,
        right=0.99,
        top=0.965,
        bottom=0.04,
    )

    smooth_mc = mc_smoothed_envelope and not mc_legacy_envelope
    smooth_sigma = (
        sigma_lag if sigma_lag is not None
        else qj_hovmoller.HOVMOLLER_GAUSSIAN_SIGMA[0],
        sigma_lon if sigma_lon is not None
        else qj_hovmoller.HOVMOLLER_GAUSSIAN_SIGMA[1],
    )

    rows: list = []
    ims: list = []

    blocks = [
        (storms_hi, "WP", ("a", "b", "c"), 0, 1,
         "High downstream RWB (upper quintile) \N{EM DASH} WP"),
        (storms_hi, "NA", ("d", "e", "f"), 2, 3,
         "High downstream RWB (upper quintile) \N{EM DASH} NA"),
        (storms_lo, "WP", ("g", "h", "i"), 4, 5,
         "Low downstream RWB (lower quintile) \N{EM DASH} WP"),
        (storms_lo, "NA", ("j", "k", "l"), 6, 7,
         "Low downstream RWB (lower quintile) \N{EM DASH} NA"),
    ]

    n_counts: dict[str, int] = {}
    for st, basin, letters, mrow, hrow, btxt in blocks:
        labels = (
            f"({letters[0]}) {basin} \N{EM DASH} RWP frequency",
            f"({letters[1]}) {basin} \N{EM DASH} RWP amplitude",
            f"({letters[2]}) {basin} \N{EM DASH} Raw LWA",
        )
        these_ims: list = []
        before = list(fig.axes)
        n = rwp_lwa_rows.basin_plots_fig2_rwp_lwa(
            storms_df=st, basin=basin, reference=reference,
            n_mc=n_mc,
            n_workers=workers,
            smooth_sigma=smooth_sigma,
            smooth_mc_anomalies=smooth_mc,
            tau_star=tau_star,
            rwp_strip_y0=rwp_strip_year_start,
            rwp_strip_y1=rwp_strip_year_end,
            lwa_y0=lwa_year_start,
            lwa_y1=lwa_year_end,
            rwp_dir=rwp_directory,
            clim_path_rwp=clim_rwp,
            clim_path_lwa=clim_lwa,
            full_tracks=full_tracks,
            map_row=mrow,
            hov_row=hrow,
            labels=labels,
            fig=fig,
            gs=gs,
            hov_axes=rows,
            ims=these_ims,
        )
        if not ims:
            ims = these_ims
        n_counts[f"{basin}-{'high' if mrow < 4 else 'low'}"] = n
        new_axes = [a for a in fig.axes if a not in before]
        if new_axes:
            leftmost_minimap = new_axes[0]
            leftmost_minimap.set_title(
                f"  {btxt}  (N={n})", fontsize=13, fontweight="semibold",
                loc="left", pad=4)

    gs_cb = gs[8, :].subgridspec(1, 3, wspace=0.45)
    for j in range(3):
        _im, levels, label = ims[j]
        cax = fig.add_subplot(gs_cb[0, j])
        norm = matplotlib.colors.BoundaryNorm(
            levels, ncolors=qj_hovmoller._BWOR_8.N)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=qj_hovmoller._BWOR_8)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal",
                          ticks=levels, extend="both")
        cb.ax.tick_params(labelsize=12)
        cb.set_label(label, fontsize=13)

    ref_lbl = {"recurvature": "Recurvature-relative",
               "et": "ET-relative"}[reference]
    n_summary = "  ".join(f"{k} N={v}" for k, v in n_counts.items())
    fig.suptitle(
        f"{ref_lbl} \N{EM DASH} Fig. 7 (WB stratified, WP/NA split): "
        f"{n_summary}. {qj_hovmoller.BASIN_SEASON['WP']} climatology",
        fontsize=14,
        y=0.995,
    )

    out = Path(output_directory) / figure_name.format(reference=reference)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    typer.run(main)
