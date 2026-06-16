"""Paper Fig. 2: 2x3 Hovmoller grid (WP row, NA row) of RWP frequency,
RWP amplitude, and raw LWA anomalies, with shared per-column colorbars."""

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

basin_plots_fig2_rwp_lwa = rwp_lwa_rows.basin_plots_fig2_rwp_lwa


def _default_worker_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 8)


def main(
        tracks_directory: Annotated[Path, typer.Option(
            help="Track database directory with recurving_nh_tracks.csv "
                 "and individual/")],
        rwp_directory: Annotated[Path, typer.Option(
            help="Directory with rwp_envelope_YYYY_MM.nc")],
        lwa_directory: Annotated[Path, typer.Option(
            help="Directory with lwa_filtered_YYYY_MM.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {reference} is substituted")]
        = "fig2_rwp_lwa_wp_na_{reference}.png",
        rwp_climatology_path: Annotated[Optional[Path], typer.Option(
            help="rwp_climatology.nc (default: "
                 "<rwp-directory>/rwp_climatology.nc)")] = None,
        lwa_climatology_path: Annotated[Optional[Path], typer.Option(
            help="lwa_climatology.nc (default: "
                 "<lwa-directory>/lwa_climatology.nc)")] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature or et")] = "recurvature",
        n_mc: Annotated[int, typer.Option(
            help="Monte Carlo iterations per panel source")] = 400,
        n_workers: Annotated[Optional[int], typer.Option(
            help="Parallel MC workers (default: CPUs available)")] = None,
        rwp_strip_year_start: Annotated[int, typer.Option()] = 2000,
        rwp_strip_year_end: Annotated[int, typer.Option()] = 2022,
        lwa_year_start: Annotated[int, typer.Option()] = 2000,
        lwa_year_end: Annotated[int, typer.Option()] = 2022,
        tau_star: Annotated[float, typer.Option()] = 1.9,
        mc_smoothed_envelope: Annotated[bool, typer.Option(
            help="Build MC null from Gaussian-smoothed anomalies (same "
                 "sigma as the map). Default is raw composite vs raw "
                 "null.")] = False,
        mc_legacy_envelope: Annotated[bool, typer.Option(
            hidden=True)] = False,
        sigma_lag: Annotated[Optional[float], typer.Option(
            help="Gaussian sigma (lag grid)")] = None,
        sigma_lon: Annotated[Optional[float], typer.Option(
            help="Gaussian sigma along longitude (1 deg/cell)")] = None,
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
        keep_default_na=False,
        na_values=[""],
    )
    _, full_tracks = tracks.load_track_database(
        tracks_directory=tracks_directory)

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

    fig = plt.figure(figsize=(19.5, 10.8))
    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[0.5, 1.55, 1.55, 0.22],
        hspace=0.38,
        wspace=0.26,
        left=0.055,
        right=0.99,
        top=0.965,
        bottom=0.05,
    )

    hov_axes_wp: list = []
    hov_axes_na: list = []
    ims_wp: list = []
    ims_na: list = []

    smooth_mc = mc_smoothed_envelope and not mc_legacy_envelope
    smooth_sigma = (
        sigma_lag if sigma_lag is not None
        else qj_hovmoller.HOVMOLLER_GAUSSIAN_SIGMA[0],
        sigma_lon if sigma_lon is not None
        else qj_hovmoller.HOVMOLLER_GAUSSIAN_SIGMA[1],
    )

    _n_wp = rwp_lwa_rows.basin_plots_fig2_rwp_lwa(
        storms_df=storms_df, basin="WP", reference=reference,
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
        map_row=0,
        hov_row=1,
        labels=("(a) WP \N{EM DASH} RWP frequency",
                "(b) WP \N{EM DASH} RWP amplitude",
                "(c) WP \N{EM DASH} raw LWA"),
        fig=fig,
        gs=gs,
        hov_axes=hov_axes_wp,
        ims=ims_wp,
        show_xlabel=False,
    )
    _n_na = rwp_lwa_rows.basin_plots_fig2_rwp_lwa(
        storms_df=storms_df, basin="NA", reference=reference,
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
        map_row=-1,
        hov_row=2,
        labels=("(d) NA \N{EM DASH} RWP frequency",
                "(e) NA \N{EM DASH} RWP amplitude",
                "(f) NA \N{EM DASH} raw LWA"),
        fig=fig,
        gs=gs,
        hov_axes=hov_axes_na,
        ims=ims_na,
        show_minimap=False,
    )

    gs_cb = gs[3, :].subgridspec(1, 3, wspace=0.45)
    for j in range(3):
        _im_wp, levels, label = ims_wp[j]
        cax = fig.add_subplot(gs_cb[0, j])
        norm = matplotlib.colors.BoundaryNorm(
            levels, ncolors=qj_hovmoller._BWOR_8.N)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=qj_hovmoller._BWOR_8)
        sm.set_array([])
        cb = fig.colorbar(
            sm, cax=cax, orientation="horizontal",
            ticks=levels, extend="both",
        )
        cb.ax.tick_params(labelsize=15)
        cb.set_label(label, fontsize=16)

    out = Path(output_directory) / figure_name.format(reference=reference)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    typer.run(main)
