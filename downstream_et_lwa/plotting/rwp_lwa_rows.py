"""Shared Fig.-2-style Hovmoller row builder: RWP frequency, RWP amplitude,
and raw LWA anomaly panels for one basin (used by paper Figs. 2, 7, 18)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.axes
import matplotlib.collections
import matplotlib.figure
import matplotlib.gridspec
import numpy as np
import pandas as pd
import scipy.ndimage

import downstream_et_lwa.plotting.lwa_hovmoller as lwa_hovmoller
import downstream_et_lwa.plotting.qj_hovmoller as qj_hovmoller

_LOG = logging.getLogger(__name__)

FREQ_LEVELS = np.array([-24, -18, -12, -6, 0, 6, 12, 18, 24], dtype=float)
AMPL_LEVELS = np.array([-2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4])
RAW_LEVELS = np.array([-12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0])


def _add_minimap(*, fig: matplotlib.figure.Figure,
                 gs: matplotlib.gridspec.GridSpec,
                 row: int, col: int) -> matplotlib.axes.Axes:
    ax_m = fig.add_subplot(gs[row, col])
    qj_hovmoller._draw_minimap(ax=ax_m)
    ax_m.set_title("", fontsize=9)
    return ax_m


def basin_plots_fig2_rwp_lwa(
        *,
        storms_df: pd.DataFrame,
        basin: str,
        reference: str,
        n_mc: int,
        n_workers: int,
        smooth_sigma: tuple[float, float],
        smooth_mc_anomalies: bool,
        tau_star: float,
        rwp_strip_y0: int,
        rwp_strip_y1: int,
        lwa_y0: int,
        lwa_y1: int,
        rwp_dir: Path | None,
        clim_path_rwp: Path,
        clim_path_lwa: Path,
        full_tracks: dict[str, pd.DataFrame] | None,
        map_row: int,
        hov_row: int,
        labels: tuple[str, str, str],
        fig: matplotlib.figure.Figure,
        gs: matplotlib.gridspec.GridSpec,
        hov_axes: list[matplotlib.axes.Axes],
        ims: list[tuple[matplotlib.collections.QuadMesh, np.ndarray, str]],
) -> int:
    season = qj_hovmoller.BASIN_SEASON.get(basin, "JJASON")

    comp = qj_hovmoller.build_composite(
        storms_df=storms_df, reference=reference, basin=basin)
    if len(comp["ref_times"]) == 0:
        raise RuntimeError(f"No storms for basin {basin}")

    F_clim, E_clim = qj_hovmoller.load_clim(
        season=season, clim_path=clim_path_rwp)
    F_anom = comp["F"] - F_clim[None, :]
    E_anom = comp["E"] - E_clim[None, :]

    f_lo, f_hi, e_lo, e_hi = qj_hovmoller.monte_carlo_sig(
        ref_times=comp["ref_times"],
        n_iter=n_mc,
        year_start=rwp_strip_y0,
        year_end=rwp_strip_y1,
        n_workers=n_workers,
        F_clim=F_clim,
        E_clim=E_clim,
        smooth_sigma=smooth_sigma,
        smooth_mc_anomalies=smooth_mc_anomalies,
    )
    if smooth_mc_anomalies:
        f_lo_anom, f_hi_anom, e_lo_anom, e_hi_anom = f_lo, f_hi, e_lo, e_hi
    else:
        f_lo_anom = f_lo - F_clim[None, :]
        f_hi_anom = f_hi - F_clim[None, :]
        e_lo_anom = e_lo - E_clim[None, :]
        e_hi_anom = e_hi - E_clim[None, :]

    F_plot = scipy.ndimage.gaussian_filter(F_anom, sigma=smooth_sigma) * 100.0
    E_plot = scipy.ndimage.gaussian_filter(E_anom, sigma=smooth_sigma)
    f_lo_p = f_lo_anom * 100.0
    f_hi_p = f_hi_anom * 100.0
    e_lo_p = e_lo_anom
    e_hi_p = e_hi_anom

    comp_lwa = lwa_hovmoller.build_composite(
        storms_df=storms_df, reference=reference, basin=basin)
    Fc, Ec, R_clim = lwa_hovmoller.load_clim(
        season=season, clim_path=clim_path_lwa)
    R_anom = comp_lwa["R"] - R_clim[None, :]
    f_lo, f_hi, e_lo, e_hi, r_lo, r_hi = lwa_hovmoller.monte_carlo_sig(
        ref_times=comp_lwa["ref_times"],
        n_iter=n_mc,
        year_start=lwa_y0,
        year_end=lwa_y1,
        n_workers=n_workers,
        F_clim=Fc,
        E_clim=Ec,
        R_clim=R_clim,
        smooth_sigma=smooth_sigma,
        smooth_mc_anomalies=smooth_mc_anomalies,
    )
    if smooth_mc_anomalies:
        r_lo_anom, r_hi_anom = r_lo, r_hi
    else:
        r_lo_anom = r_lo - R_clim[None, :]
        r_hi_anom = r_hi - R_clim[None, :]
    R_plot = scipy.ndimage.gaussian_filter(R_anom, sigma=smooth_sigma)

    if str(basin).upper() == "WPNA":
        st = storms_df.copy()
    else:
        st = storms_df[storms_df["basin"] == basin].copy()
    ref_col = "recurv_lon" if reference == "recurvature" else "et_lon"
    if ref_col in st.columns and len(st) > 0:
        lon_vals = st[ref_col].to_numpy() % 360
        lon_range: tuple[float, float] | None = (
            float(np.nanmin(lon_vals)), float(np.nanmax(lon_vals)))
        recurv_lon_mean: float | None = float(np.nanmean(lon_vals))
    else:
        lon_range, recurv_lon_mean = None, None
    mean_track_lag, mean_track_lon = qj_hovmoller._mean_track(
        storms_df=storms_df, basin=basin, reference=reference,
        full_tracks=full_tracks)
    track_kw: dict[str, Any] = dict(
        lon_range=lon_range,
        mean_track_lag=mean_track_lag,
        mean_track_lon=mean_track_lon,
        recurv_lon_mean=recurv_lon_mean,
    )

    for c in range(3):
        _add_minimap(fig=fig, gs=gs, row=map_row, col=c)

    kw: dict[str, Any] = dict(
        tick_labelsize=11,
        title_fontsize=12,
        with_colorbar=False,
        show_lon_extent_hline=(str(basin).upper() != "WPNA"),
        **track_kw,
    )
    sig_f = None if smooth_mc_anomalies else F_anom * 100.0
    sig_e = None if smooth_mc_anomalies else E_anom
    sig_r = None if smooth_mc_anomalies else R_anom

    ax0 = fig.add_subplot(gs[hov_row, 0])
    im0 = qj_hovmoller._hovmoller_panel(
        ax=ax0, data=F_plot, mask_lo=f_lo_p, mask_hi=f_hi_p,
        levels=FREQ_LEVELS, cmap=qj_hovmoller._BWOR_8,
        title=labels[0],
        cbar_label="RWP frequency anomaly (%)",
        sig_field=sig_f,
        **kw,
    )
    ax1 = fig.add_subplot(gs[hov_row, 1])
    im1 = qj_hovmoller._hovmoller_panel(
        ax=ax1, data=E_plot, mask_lo=e_lo_p, mask_hi=e_hi_p,
        levels=AMPL_LEVELS, cmap=qj_hovmoller._BWOR_8,
        title=labels[1],
        cbar_label=r"RWP amplitude anomaly (m s$^{-1}$)",
        sig_field=sig_e,
        **kw,
    )
    ax2 = fig.add_subplot(gs[hov_row, 2])
    im2 = qj_hovmoller._hovmoller_panel(
        ax=ax2, data=R_plot, mask_lo=r_lo_anom, mask_hi=r_hi_anom,
        levels=RAW_LEVELS, cmap=qj_hovmoller._BWOR_8,
        title=labels[2],
        cbar_label=r"Raw LWA anomaly (m s$^{-1}$)",
        sig_field=sig_r,
        **kw,
    )
    hov_axes.extend([ax0, ax1, ax2])
    ims.extend([(im0, FREQ_LEVELS, "RWP frequency anomaly (%)"),
                (im1, AMPL_LEVELS, r"RWP amplitude anomaly (m s$^{-1}$)"),
                (im2, RAW_LEVELS, r"Raw LWA anomaly (m s$^{-1}$)")])

    return len(comp["ref_times"])
