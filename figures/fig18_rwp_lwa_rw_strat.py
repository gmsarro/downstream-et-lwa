"""Paper Fig. 18: WP-only RW-case and no-RW-case Hovmoller rows (RWP
frequency, RWP amplitude, raw LWA), absolute longitude as in Fig. 2."""

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
            help="WP-only RW classification CSV (storm_id, reference, "
                 "wb_group with rwcase/norwcase)")],
        rwp_directory: Annotated[Path, typer.Option(
            help="Directory with rwp_envelope_YYYY_MM.nc")],
        lwa_directory: Annotated[Path, typer.Option(
            help="Directory with lwa_filtered_YYYY_MM.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {reference} is substituted")]
        = "fig18_rwp_lwa_wp_na_rw_strat_{reference}.png",
        rwp_climatology_path: Annotated[Optional[Path], typer.Option(
            help="rwp_climatology.nc (default: "
                 "<rwp-directory>/rwp_climatology.nc)")] = None,
        lwa_climatology_path: Annotated[Optional[Path], typer.Option(
            help="lwa_climatology.nc (default: "
                 "<lwa-directory>/lwa_climatology.nc)")] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature or et")] = "recurvature",
        n_mc: Annotated[int, typer.Option()] = 300,
        n_workers: Annotated[Optional[int], typer.Option()] = None,
        rwp_strip_year_start: Annotated[int, typer.Option()] = 2000,
        rwp_strip_year_end: Annotated[int, typer.Option()] = 2022,
        lwa_year_start: Annotated[int, typer.Option()] = 2000,
        lwa_year_end: Annotated[int, typer.Option()] = 2022,
        tau_star: Annotated[float, typer.Option()] = 1.9,
        mc_smoothed_envelope: Annotated[bool, typer.Option(
            "--mc-smoothed-envelope/--mc-raw-envelope",
            help="Smoothed-vs-smoothed MC null (default) or legacy "
                 "raw-composite-vs-raw-null test")] = True,
        sigma_lag: Annotated[Optional[float], typer.Option()] = None,
        sigma_lon: Annotated[Optional[float], typer.Option()] = None,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if reference not in ("recurvature", "et"):
        raise typer.BadParameter("reference must be recurvature or et")
    workers = n_workers if n_workers is not None else _default_worker_count()

    storms_path = Path(tracks_directory) / "recurving_nh_tracks.csv"
    storms_df = pd.read_csv(
        storms_path,
        parse_dates=["recurv_time", "et_time"],
        keep_default_na=False, na_values=[""],
    )
    _, full_tracks = tracks.load_track_database(
        tracks_directory=tracks_directory)

    cls_path = Path(classification_file)
    if not cls_path.is_file():
        raise SystemExit(f"Classification CSV not found: {cls_path}")
    cls = pd.read_csv(cls_path)
    cls_ref = cls[cls["reference"] == reference]
    rw_ids = set(cls_ref[cls_ref["wb_group"] == "rwcase"]["storm_id"])
    no_ids = set(cls_ref[cls_ref["wb_group"] == "norwcase"]["storm_id"])
    print(f"RW={len(rw_ids)}, no-RW={len(no_ids)}", flush=True)
    if not rw_ids or not no_ids:
        raise SystemExit("Empty RW or no-RW set; rerun classifier.")

    storms_rw = storms_df[
        (storms_df["storm_id"].isin(rw_ids)) & (storms_df["basin"] == "WP")
    ].copy()
    storms_no = storms_df[
        (storms_df["storm_id"].isin(no_ids)) & (storms_df["basin"] == "WP")
    ].copy()

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

    smooth_mc = bool(mc_smoothed_envelope)
    smooth_sigma = (
        sigma_lag if sigma_lag is not None
        else qj_hovmoller.HOVMOLLER_GAUSSIAN_SIGMA[0],
        sigma_lon if sigma_lon is not None
        else qj_hovmoller.HOVMOLLER_GAUSSIAN_SIGMA[1],
    )

    fig = plt.figure(figsize=(23.5, 12.8))
    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[0.5, 1.55, 1.55, 0.22],
        hspace=0.38, wspace=0.26,
        left=0.055, right=0.99, top=0.965, bottom=0.05,
    )

    print(f"Building RW row (WP, N={len(storms_rw)})...", flush=True)
    ims_rw: list = []
    rwp_lwa_rows.basin_plots_fig2_rwp_lwa(
        storms_df=storms_rw, basin="WP", reference=reference,
        map_row=0, hov_row=1,
        labels=("(a) RW case (WP) \N{EM DASH} RWP frequency",
                "(b) RW case (WP) \N{EM DASH} RWP amplitude",
                "(c) RW case (WP) \N{EM DASH} raw LWA"),
        hov_axes=[], ims=ims_rw,
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
        fig=fig,
        gs=gs,
        title_fontsize=16,
        tick_labelsize=15,
        show_xlabel=False,
    )
    print(f"Building no-RW row (WP, N={len(storms_no)})...", flush=True)
    ims_no: list = []
    rwp_lwa_rows.basin_plots_fig2_rwp_lwa(
        storms_df=storms_no, basin="WP", reference=reference,
        map_row=-1, hov_row=2,
        labels=("(d) no-RW case (WP) \N{EM DASH} RWP frequency",
                "(e) no-RW case (WP) \N{EM DASH} RWP amplitude",
                "(f) no-RW case (WP) \N{EM DASH} raw LWA"),
        hov_axes=[], ims=ims_no,
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
        fig=fig,
        gs=gs,
        show_minimap=False,
        title_fontsize=16,
        tick_labelsize=15,
    )

    gs_cb = gs[3, :].subgridspec(1, 3, wspace=0.45)
    for j in range(3):
        _im, levels, label = ims_rw[j]
        cax = fig.add_subplot(gs_cb[0, j])
        norm = matplotlib.colors.BoundaryNorm(
            levels, ncolors=qj_hovmoller._BWOR_8.N)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=qj_hovmoller._BWOR_8)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal",
                          ticks=levels, extend="both")
        cb.ax.tick_params(labelsize=18)
        cb.set_label(label, fontsize=19)

    out = Path(output_directory) / figure_name.format(reference=reference)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    typer.run(main)
