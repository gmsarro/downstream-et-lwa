"""MPAS current vs future RWP frequency, RWP amplitude, and barotropic LWA
Hovmoller composites restricted to NA storms only (paper Fig. 15)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.cm
import matplotlib.colors
import typer
from typing_extensions import Annotated

import downstream_et_lwa.composites.mpas_composites as mpas_composites
import downstream_et_lwa.plotting.mpas_strips as mpas_strips
import downstream_et_lwa.plotting.qj_hovmoller as qj3
import downstream_et_lwa.tracks as tracks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)

_BASIN = "NA"
_BASIN_LABEL = "NA only"


def _default_worker_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 8)


def main(
        tracks_root: Annotated[Path, typer.Option(
            help="Directory containing current/ and future/ track DBs "
                 "(recurving_nh_tracks.csv + individual/)")],
        rwp_root: Annotated[Path, typer.Option(
            help="Directory containing current/ and future/ RWP envelope "
                 "files and rwp_climatology.nc")],
        mpas_budget_root: Annotated[Path, typer.Option(
            help="MPAS budget root with <scenario>/*/BARO_N monthly "
                 "LWAb_N files")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option()] = (
            "fig15_mpas_rwp_lwa_na_only_recurvature.png"),
        reference: Annotated[str, typer.Option(
            help="recurvature or et")] = "recurvature",
        n_mc: Annotated[int, typer.Option()] = 400,
        n_workers: Annotated[int, typer.Option()] = _default_worker_count(),
        year_strip_start: Annotated[int, typer.Option()] = 1988,
        year_strip_end: Annotated[int, typer.Option()] = 2016,
        mc_year_start: Annotated[Optional[int], typer.Option(
            help="Defaults to --year-strip-start")] = None,
        mc_year_end: Annotated[Optional[int], typer.Option(
            help="Defaults to --year-strip-end")] = None,
        lwab_year_start: Annotated[Optional[int], typer.Option()] = None,
        lwab_year_end: Annotated[Optional[int], typer.Option()] = None,
        sigma_lag: Annotated[Optional[float], typer.Option()] = None,
        sigma_lon: Annotated[Optional[float], typer.Option()] = None,
        mc_smoothed_envelope: Annotated[bool, typer.Option()] = False,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if reference not in ("recurvature", "et"):
        raise typer.BadParameter("reference must be recurvature or et")

    mc_lo = mc_year_start or year_strip_start
    mc_hi = mc_year_end or year_strip_end
    lw_lo = lwab_year_start or year_strip_start
    lw_hi = lwab_year_end or year_strip_end

    smooth_sigma = (
        sigma_lag if sigma_lag is not None
        else qj3.HOVMOLLER_GAUSSIAN_SIGMA[0],
        sigma_lon if sigma_lon is not None
        else qj3.HOVMOLLER_GAUSSIAN_SIGMA[1],
    )

    fig = plt.figure(figsize=(18.5, 13.0))
    gs = fig.add_gridspec(
        5, 3,
        height_ratios=[0.5, 1.55, 0.5, 1.55, 0.22],
        hspace=0.50, wspace=0.26,
        left=0.055, right=0.99, top=0.91, bottom=0.045,
    )
    hov_axes: list = []
    ims_curr: list = []
    ims_fut: list = []

    storms_curr, ft_curr = tracks.load_track_database(
        tracks_directory=tracks_root / "current")
    storms_fut, ft_fut = tracks.load_track_database(
        tracks_directory=tracks_root / "future")
    storms_curr = mpas_composites.filter_mpas_storms_within_data(
        storms_df=storms_curr, ref_col="recurv_time")
    storms_fut = mpas_composites.filter_mpas_storms_within_data(
        storms_df=storms_fut, ref_col="recurv_time")

    storms_curr_b = storms_curr[storms_curr["basin"] == _BASIN].copy()
    storms_fut_b = storms_fut[storms_fut["basin"] == _BASIN].copy()

    common: dict[str, Any] = dict(
        n_mc=n_mc,
        n_workers=n_workers,
        smooth_sigma=smooth_sigma,
        smooth_mc_anomalies=mc_smoothed_envelope,
        tau_star=3.2,
        year_strip_lo=year_strip_start,
        year_strip_hi=year_strip_end,
        mc_year_lo=mc_lo,
        mc_year_hi=mc_hi,
        lwab_year_lo=lw_lo,
        lwab_year_hi=lw_hi,
        mc_years_pool=tuple(mpas_composites.MPAS_VALID_YEARS),
        mpas_budget_root=mpas_budget_root,
        fig=fig,
        gs=gs,
        hov_axes=hov_axes,
        basin=_BASIN,
        title_fontsize=15,
        tick_labelsize=14,
    )

    n_db_c = int(len(storms_curr_b))
    n_db_f = int(len(storms_fut_b))

    fig.text(
        0.5, 0.935,
        f"MPAS current \N{EM DASH} {_BASIN_LABEL} (storm DB N={n_db_c})",
        fontsize=13, fontweight="semibold", ha="center",
        transform=fig.transFigure)
    n_cur = mpas_strips.scenario_rwp_lwa_row(
        storms_df=storms_curr_b, scenario="current", reference=reference,
        rwp_dir=rwp_root / "current",
        clim_rwp=rwp_root / "current" / qj3.RWP_CLIMATOLOGY_FILENAME,
        map_row=0, hov_row=1,
        labels=("(a) RWP frequency", "(b) RWP amplitude",
                "(c) Barotropic LWA (MPAS)"),
        ims=ims_curr,
        full_tracks=ft_curr,
        show_xlabel=False,
        **common,
    )

    fig.text(
        0.5, 0.510,
        f"MPAS future \N{EM DASH} {_BASIN_LABEL} (storm DB N={n_db_f})",
        fontsize=13, fontweight="semibold", ha="center",
        transform=fig.transFigure)
    n_fut = mpas_strips.scenario_rwp_lwa_row(
        storms_df=storms_fut_b, scenario="future", reference=reference,
        rwp_dir=rwp_root / "future",
        clim_rwp=rwp_root / "future" / qj3.RWP_CLIMATOLOGY_FILENAME,
        map_row=2, hov_row=3,
        labels=("(d) RWP frequency", "(e) RWP amplitude",
                "(f) Barotropic LWA (MPAS)"),
        ims=ims_fut,
        full_tracks=ft_fut,
        **common,
    )

    gs_cb = gs[4, :].subgridspec(1, 3, wspace=0.45)
    for j in range(3):
        _im, levels, label = ims_curr[j]
        cax = fig.add_subplot(gs_cb[0, j])
        norm = matplotlib.colors.BoundaryNorm(levels, ncolors=qj3._BWOR_8.N)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=qj3._BWOR_8)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal",
                          ticks=levels, extend="both")
        cb.ax.tick_params(labelsize=13)
        cb.set_label(label, fontsize=14)

    ref_lbl = {"recurvature": "Recurvature-relative",
               "et": "ET-relative"}[reference]
    fig.suptitle(
        f"{ref_lbl} \N{EM DASH} Fig.~15 MPAS current vs future, "
        f"{_BASIN_LABEL} (composite N_cur={n_cur}, N_fut={n_fut}; "
        f"storm DB N_cur={n_db_c}, N_fut={n_db_f})",
        fontsize=13, y=0.985,
    )

    out = Path(output_directory) / figure_name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    typer.run(main)
