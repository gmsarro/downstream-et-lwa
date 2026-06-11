"""Population composites of the LH / R+ removal experiment in the Fig. 2 plotting style."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.cm
import matplotlib.colors
import netCDF4 as nc
import numpy as np
import pandas as pd
import scipy.ndimage
import typer
from typing_extensions import Annotated

import downstream_et_lwa.lh_removal._composite_helpers as composite_helpers
import downstream_et_lwa.lh_removal.advection_kernel as advection_kernel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)

HOURS_BEFORE = 24
HOURS_AFTER = 24 * 8
DT_LAG_HOURS = 6
LAGS_H = np.arange(-HOURS_BEFORE, HOURS_AFTER + DT_LAG_HOURS, DT_LAG_HOURS)
LAGS_D = LAGS_H / 24.0

REL_HALF = composite_helpers.REL_LON_HALF

SMOOTH_SIGMA = composite_helpers.HOVMOLLER_GAUSSIAN_SIGMA


def _interp_to_lag(*, times_s: np.ndarray, t0_window: datetime.datetime,
                   t_recurv: datetime.datetime,
                   strip: np.ndarray) -> np.ndarray:
    rel_s = times_s + (t0_window - t_recurv).total_seconds()
    target_s = LAGS_H * 3600.0
    out = np.full((len(LAGS_H), advection_kernel.NLON), np.nan,
                  dtype=np.float64)
    in_range = (target_s >= rel_s[0]) & (target_s <= rel_s[-1])
    if not in_range.any():
        return out
    idx = np.where(in_range)[0]
    for k in range(advection_kernel.NLON):
        out[idx, k] = np.interp(target_s[in_range], rel_s, strip[:, k])
    return out


def _rotate_storm_relative(*, arr_lag_lon: np.ndarray,
                           recurv_lon: float) -> np.ndarray:
    rl_int = int(round(float(recurv_lon) % 360.0)) % advection_kernel.NLON
    return np.roll(arr_lag_lon, REL_HALF - rl_int, axis=1)


def load_lwa_climatology(*, climatology_file: Path) -> np.ndarray:
    if not climatology_file.exists():
        raise FileNotFoundError(
            f"{climatology_file} not found. Build the JJASON LWA climatology "
            "first (lwa_climatology_jjason).")
    with nc.Dataset(climatology_file, "r") as d:
        return np.array(d["lwa_clim"][:], dtype=np.float64)


def _load_strip(*, storm_id: str, var: str,
                strip_directory: Path) -> np.ndarray:
    p = strip_directory / f"strip_{storm_id}.nc"
    with nc.Dataset(p, "r") as d:
        return np.array(d[var][:], dtype=np.float64)


def _strip_meta(*, storm_id: str, strip_directory: Path) -> dict[str, Any]:
    p = strip_directory / f"strip_{storm_id}.nc"
    with nc.Dataset(p, "r") as d:
        return dict(
            times_s=np.array(d["time"][:], dtype=np.float64),
            window_start=d.window_start,
            recurv_time=d.recurv_time,
        )


def composite_basin(*, df_basin: pd.DataFrame, var: str,
                    strip_directory: Path,
                    storm_relative: bool = False
                    ) -> tuple[np.ndarray, int]:
    acc = np.zeros((len(LAGS_H), advection_kernel.NLON), dtype=np.float64)
    cnt = np.zeros((len(LAGS_H), advection_kernel.NLON), dtype=np.float64)
    used = 0
    for _, r in df_basin.iterrows():
        sid = r["storm_id"]
        p = strip_directory / f"strip_{sid}.nc"
        if not p.exists():
            continue
        try:
            strip = _load_strip(storm_id=sid, var=var,
                                strip_directory=strip_directory)
            meta = _strip_meta(storm_id=sid, strip_directory=strip_directory)
            t0w = pd.Timestamp(meta["window_start"]).to_pydatetime()
            tr = pd.Timestamp(meta["recurv_time"]).to_pydatetime()
            on_lag = _interp_to_lag(times_s=meta["times_s"], t0_window=t0w,
                                    t_recurv=tr, strip=strip)
            if storm_relative:
                on_lag = _rotate_storm_relative(arr_lag_lon=on_lag,
                                                recurv_lon=r["recurv_lon"])
            mask = np.isfinite(on_lag)
            acc[mask] += on_lag[mask]
            cnt[mask] += 1.0
            used += 1
        except Exception:
            _LOG.exception("Strip %s failed", sid)
    out = np.full_like(acc, np.nan)
    np.divide(acc, cnt, out=out, where=cnt > 0)
    return out, used


def _safe_smooth(*, field: np.ndarray) -> np.ndarray:
    f = np.where(np.isfinite(field), field, 0.0)
    w = (np.isfinite(field)).astype(np.float64)
    fs = scipy.ndimage.gaussian_filter(f, sigma=SMOOTH_SIGMA)
    ws = scipy.ndimage.gaussian_filter(w, sigma=SMOOTH_SIGMA)
    out = np.full_like(field, np.nan, dtype=np.float64)
    np.divide(fs, ws, out=out, where=ws > 0.05)
    return out


def make_composite_figure(*,
                          df_pop: pd.DataFrame,
                          output_path: Path,
                          strip_directory: Path,
                          climatology_file: Path,
                          tracks_file: Path,
                          individual_tracks_directory: Path,
                          tc_radius_deg: float = 10.0,
                          cbar_abs_max: float = 5.0,
                          suptitle: bool = True) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_wp = df_pop[df_pop["basin"] == "WP"].copy()
    df_na = df_pop[df_pop["basin"] == "NA"].copy()

    _LOG.info("Loading JJASON LWA climatology (cos-lat 20-80N)")
    lwa_clim_1d = load_lwa_climatology(climatology_file=climatology_file)

    _LOG.info("Compositing CTRL (top row + delta-A base)")
    ctrl_wp, n_wp = composite_basin(df_basin=df_wp, var="A_CTRL",
                                    strip_directory=strip_directory)
    ctrl_na, n_na = composite_basin(df_basin=df_na, var="A_CTRL",
                                    strip_directory=strip_directory)
    _LOG.info("CTRL WP n=%d, NA n=%d", n_wp, n_na)

    _LOG.info("Compositing NoR+")
    nrp_wp, _ = composite_basin(df_basin=df_wp, var="A_NoRPos",
                                strip_directory=strip_directory)
    nrp_na, _ = composite_basin(df_basin=df_na, var="A_NoRPos",
                                strip_directory=strip_directory)

    _LOG.info("Compositing NoLH")
    nlh_wp, _ = composite_basin(df_basin=df_wp, var="A_NoLH",
                                strip_directory=strip_directory)
    nlh_na, _ = composite_basin(df_basin=df_na, var="A_NoLH",
                                strip_directory=strip_directory)

    ctrl_wp_anom = ctrl_wp - lwa_clim_1d[None, :]
    ctrl_na_anom = ctrl_na - lwa_clim_1d[None, :]
    dRp_wp = ctrl_wp - nrp_wp
    dRp_na = ctrl_na - nrp_na
    dLH_wp = ctrl_wp - nlh_wp
    dLH_na = ctrl_na - nlh_na

    ctrl_wp_s = _safe_smooth(field=ctrl_wp_anom)
    ctrl_na_s = _safe_smooth(field=ctrl_na_anom)
    dRp_wp_s = _safe_smooth(field=dRp_wp)
    dRp_na_s = _safe_smooth(field=dRp_na)
    dLH_wp_s = _safe_smooth(field=dLH_wp)
    dLH_na_s = _safe_smooth(field=dLH_na)

    _LOG.info("Loading full track database for mean-track overlays")
    full_tracks: dict[str, pd.DataFrame] | None
    try:
        _, full_tracks = composite_helpers.load_track_database(
            tracks_file=tracks_file,
            individual_tracks_directory=individual_tracks_directory)
    except Exception:
        _LOG.exception("Could not load track database; track skipped")
        full_tracks = None

    track_lag_wp, track_lon_wp = composite_helpers._mean_track(
        storms_df=df_pop, basin="WP", reference="recurvature",
        full_tracks=full_tracks)
    track_lag_na, track_lon_na = composite_helpers._mean_track(
        storms_df=df_pop, basin="NA", reference="recurvature",
        full_tracks=full_tracks)

    def _recurv_stats(*, df_b: pd.DataFrame
                      ) -> tuple[tuple[float, float] | None, float | None]:
        if "recurv_lon" not in df_b.columns or len(df_b) == 0:
            return None, None
        lon_vals = df_b["recurv_lon"].to_numpy() % 360.0
        return ((float(np.nanmin(lon_vals)),
                 float(np.nanmax(lon_vals))),
                float(np.nanmean(lon_vals)))

    lon_range_wp, recurv_mean_wp = _recurv_stats(df_b=df_wp)
    lon_range_na, recurv_mean_na = _recurv_stats(df_b=df_na)

    fig = plt.figure(figsize=(14.0, 18.5))
    gs = fig.add_gridspec(
        nrows=7, ncols=2,
        height_ratios=[0.40, 2.20, 0.40, 2.20, 0.40, 2.20, 0.42],
        hspace=0.18, wspace=0.20,
        left=0.07, right=0.965, top=0.935, bottom=0.06,
    )

    shared_levels = np.linspace(-float(cbar_abs_max), float(cbar_abs_max),
                                9, dtype=float)

    common_kw: dict[str, Any] = dict(
        y_days=LAGS_D,
        y_lim=(-1.5, 8.5),
        show_lon_extent_hline=True,
        significance=False,
        with_colorbar=False,
        tick_labelsize=11,
        title_fontsize=12,
    )

    def _block(*, map_row: int, hov_row: int,
               panels: tuple[np.ndarray, np.ndarray],
               titles: tuple[str, str]) -> None:
        for c in range(2):
            ax_m = fig.add_subplot(gs[map_row, c])
            composite_helpers._draw_minimap(ax=ax_m)
            if map_row == 0:
                ax_m.set_title(
                    "WP recurving TCs" if c == 0 else "NA recurving TCs",
                    fontsize=11, loc="center", pad=2.0,
                )

        ax_left = fig.add_subplot(gs[hov_row, 0])
        composite_helpers._hovmoller_panel(
            ax=ax_left, data=panels[0],
            mask_lo=np.zeros_like(panels[0]),
            mask_hi=np.zeros_like(panels[0]),
            levels=shared_levels, cmap=composite_helpers._BWOR_8,
            title=titles[0], cbar_label="",
            mean_track_lag=track_lag_wp, mean_track_lon=track_lon_wp,
            lon_range=lon_range_wp,
            recurv_lon_mean=recurv_mean_wp,
            show_xlabel=(hov_row == 5), show_ylabel=True,
            **common_kw,
        )
        ax_right = fig.add_subplot(gs[hov_row, 1])
        composite_helpers._hovmoller_panel(
            ax=ax_right, data=panels[1],
            mask_lo=np.zeros_like(panels[1]),
            mask_hi=np.zeros_like(panels[1]),
            levels=shared_levels, cmap=composite_helpers._BWOR_8,
            title=titles[1], cbar_label="",
            mean_track_lag=track_lag_na, mean_track_lon=track_lon_na,
            lon_range=lon_range_na,
            recurv_lon_mean=recurv_mean_na,
            show_xlabel=(hov_row == 5), show_ylabel=False,
            **common_kw,
        )

    _block(
        map_row=0, hov_row=1,
        panels=(ctrl_wp_s, ctrl_na_s),
        titles=(f"(a) WP - CTRL reconstruction LWA anomaly  (N={n_wp})",
                f"(b) NA - CTRL reconstruction LWA anomaly  (N={n_na})"),
    )
    _block(
        map_row=2, hov_row=3,
        panels=(dRp_wp_s, dRp_na_s),
        titles=(f"(c) WP - $\\Delta$A = CTRL $-$ NoR$^+$   "
                f"(local TC R$^+$, R={tc_radius_deg:g}\N{DEGREE SIGN})",
                f"(d) NA - $\\Delta$A = CTRL $-$ NoR$^+$   "
                f"(local TC R$^+$, R={tc_radius_deg:g}\N{DEGREE SIGN})"),
    )
    _block(
        map_row=4, hov_row=5,
        panels=(dLH_wp_s, dLH_na_s),
        titles=(f"(e) WP - $\\Delta$A = CTRL $-$ NoLH   "
                f"(local TC LH, R={tc_radius_deg:g}\N{DEGREE SIGN})",
                f"(f) NA - $\\Delta$A = CTRL $-$ NoLH   "
                f"(local TC LH, R={tc_radius_deg:g}\N{DEGREE SIGN})"),
    )

    cb_outer = fig.add_subplot(gs[6, :])
    cb_outer.set_axis_off()
    bb = cb_outer.get_position()
    cax = fig.add_axes((
        bb.x0 + 0.12 * bb.width,
        bb.y0 + 0.15 * bb.height,
        0.76 * bb.width,
        0.55 * bb.height,
    ))
    norm = matplotlib.colors.BoundaryNorm(shared_levels,
                                          ncolors=composite_helpers._BWOR_8.N)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=composite_helpers._BWOR_8)
    sm.set_array([])
    cb = fig.colorbar(
        sm, cax=cax, orientation="horizontal",
        ticks=shared_levels, extend="both",
    )
    cb.set_label(
        r"LWA anomaly / $\Delta\langle$LWA$\rangle$  (20-80N, m s$^{-1}$); "
        f"shared scale \N{PLUS-MINUS SIGN}{cbar_abs_max:g}",
        fontsize=11,
    )
    cb.ax.tick_params(labelsize=9)

    if suptitle:
        yr_lo = int(df_pop["recurv_time"].dt.year.min())
        yr_hi = int(df_pop["recurv_time"].dt.year.max())
        fig.suptitle(
            "Recurvature-relative composites of the 2-D LWA reconstruction "
            f"- TC-local diabatic source removal (R = {tc_radius_deg:g}\N{DEGREE SIGN})\n"
            f"WP and NA recurving TCs, {yr_lo}-{yr_hi}  "
            f"(absolute longitude; BARO_N ERA5; LN24 + NHN22; "
            f"Gaussian sigma=(6 h, 2.5\N{DEGREE SIGN}))\n"
            "Top row: CTRL forward-integrated LWA anomaly "
            "(diverges from observed at long lead times)",
            fontsize=11, y=0.985,
        )

    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main(
    tracks_file: Annotated[Path, typer.Option(help="Recurving NH tracks CSV.")],
    individual_tracks_directory: Annotated[Path, typer.Option(
        help="Directory of per-storm track CSVs (<storm_id>.csv).")],
    strip_directory: Annotated[Path, typer.Option(
        help="Directory with strip_<storm_id>.nc files.")],
    climatology_file: Annotated[Path, typer.Option(
        help="JJASON cos-lat 20-80N LWA climatology NetCDF.")],
    output_directory: Annotated[Path, typer.Option(
        help="Directory for the composite figure.")],
    figure_name: Annotated[str, typer.Option()] =
        "fig5_lh_Rpos_removal_composites_WP_NA.png",
    year_start: Annotated[int, typer.Option()] = 2000,
    year_end: Annotated[int, typer.Option()] = 2022,
    basins: Annotated[list[str], typer.Option()] = ["WP", "NA"],
    tc_radius_deg: Annotated[float, typer.Option(
        help="Figure title only; the actual radius is whatever was used "
             "when building the catalogs.")] = 10.0,
    cbar_max: Annotated[float, typer.Option(
        help="Half-range of the shared diverging colour scale (m/s).")] = 5.0,
    suptitle: Annotated[bool, typer.Option(
        help="Draw the figure suptitle (disable for manuscript output).")] = True,
) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    df = pd.read_csv(tracks_file,
                     parse_dates=["recurv_time", "et_time"],
                     keep_default_na=False, na_values=[""])
    yr = df["recurv_time"].dt.year
    sel = ((yr >= year_start) & (yr <= year_end)
           & df["basin"].isin(basins))
    df = df[sel].copy()
    have_strip = df["storm_id"].apply(
        lambda sid: (strip_directory / f"strip_{sid}.nc").exists()
    )
    df = df[have_strip].copy()
    print(f"[info] {len(df)} storms with strips in basins {basins} "
          f"({year_start}-{year_end})")

    out = make_composite_figure(
        df_pop=df,
        output_path=output_directory / figure_name,
        strip_directory=strip_directory,
        climatology_file=climatology_file,
        tracks_file=tracks_file,
        individual_tracks_directory=individual_tracks_directory,
        tc_radius_deg=tc_radius_deg,
        cbar_abs_max=cbar_max,
        suptitle=suptitle,
    )
    print(out)


if __name__ == "__main__":
    typer.run(main)
