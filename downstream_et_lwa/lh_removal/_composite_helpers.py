"""Minimal Hovmoller/minimap/track helpers inlined from the internal plot_qj_fig3 module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import cartopy.io.shapereader
import matplotlib
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors
import matplotlib.ticker
import numpy as np
import pandas as pd

import downstream_et_lwa.lh_removal.advection_kernel as advection_kernel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)

_BWOR_8 = matplotlib.colors.ListedColormap([
    "#2B3A8A",
    "#4C7CC6",
    "#90BADC",
    "#FFFFFF",
    "#FFFFFF",
    "#F5A85A",
    "#D9642C",
    "#8F2014",
], name="bwor8")
_BWOR_8.set_under("#1A2560")
_BWOR_8.set_over("#4A0A04")

HOVMOLLER_GAUSSIAN_SIGMA = (1.0, 2.5)

REL_LON_HALF = advection_kernel.NLON // 2

_TRACK_HOURS_BEFORE = 48
_TRACK_HOURS_AFTER = 168
_TRACK_DT_HOURS = 6
N_LAGS = (_TRACK_HOURS_BEFORE + _TRACK_HOURS_AFTER) // _TRACK_DT_HOURS + 1
LAG_HOURS = np.arange(-_TRACK_HOURS_BEFORE,
                      _TRACK_HOURS_AFTER + _TRACK_DT_HOURS, _TRACK_DT_HOURS)


def _fmt_lon(x: float, pos: int | None = None) -> str:
    x = ((x + 180) % 360) - 180
    if np.isclose(x, 0):
        return "0\N{DEGREE SIGN}"
    if np.isclose(x, 180) or np.isclose(x, -180):
        return "180\N{DEGREE SIGN}"
    if x > 0:
        return f"{int(round(x))}\N{DEGREE SIGN}E"
    return f"{int(round(-x))}\N{DEGREE SIGN}W"


def _fmt_lag(y: float, pos: int | None = None) -> str:
    sign = "+" if y >= 0 else "-"
    return f"T{sign}{abs(int(round(y)))}d"


def load_track_database(*, tracks_file: Path,
                        individual_tracks_directory: Path
                        ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    storms_df = pd.read_csv(tracks_file,
                            parse_dates=["recurv_time", "et_time"],
                            keep_default_na=False, na_values=[""])
    full_tracks = {}
    for sid in storms_df["storm_id"]:
        safe_sid = sid.replace("/", "_").replace(" ", "_")
        path = individual_tracks_directory / f"{safe_sid}.csv"
        if path.exists():
            full_tracks[sid] = pd.read_csv(path, parse_dates=["time"])
    return storms_df, full_tracks


def interpolate_track_to_lags(*, track_df: pd.DataFrame,
                              reference: str = "recurvature"
                              ) -> tuple[np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray]:
    if reference == "recurvature":
        hours_col = "hours_from_recurv"
    else:
        hours_col = "hours_from_et"

    out_lat = np.full(N_LAGS, np.nan)
    out_lon = np.full(N_LAGS, np.nan)
    out_wind = np.full(N_LAGS, np.nan)
    out_pres = np.full(N_LAGS, np.nan)

    t_track = track_df[hours_col].values
    sort_idx = np.argsort(t_track)
    t_sorted = t_track[sort_idx]

    for i, lag_h in enumerate(LAG_HOURS):
        diffs = np.abs(t_sorted - lag_h)
        best = np.argmin(diffs)
        if diffs[best] <= 3.0:
            j = sort_idx[best]
            out_lat[i] = track_df["lat"].iloc[j]
            out_lon[i] = track_df["lon"].iloc[j]
            out_wind[i] = track_df["wind"].iloc[j]
            out_pres[i] = track_df["pres"].iloc[j]
        elif len(t_sorted) >= 2:
            if t_sorted[0] <= lag_h <= t_sorted[-1]:
                out_lat[i] = np.interp(lag_h, t_sorted,
                                       track_df["lat"].values[sort_idx])
                out_lon[i] = np.interp(lag_h, t_sorted,
                                       track_df["lon"].values[sort_idx])

    return out_lat, out_lon, out_wind, out_pres


def _mean_track(*, storms_df: pd.DataFrame, basin: str,
                reference: str = "recurvature",
                full_tracks: dict[str, pd.DataFrame] | None
                ) -> tuple[np.ndarray | None, np.ndarray | None]:
    if full_tracks is None:
        _LOG.info("No track database available; mean track skipped")
        return None, None

    b = str(basin).upper()
    if b == "WPNA":
        sub = storms_df[storms_df["basin"].isin(("WP", "NA"))]
    else:
        sub = storms_df[storms_df["basin"] == basin]
    lag_sel = (LAG_HOURS >= -48) & (LAG_HOURS <= 96)
    lon_acc = []
    for _, st in sub.iterrows():
        sid = st["storm_id"]
        if sid not in full_tracks:
            continue
        try:
            _, lon_full, _, _ = interpolate_track_to_lags(
                track_df=full_tracks[sid], reference=reference)
        except Exception:
            _LOG.exception("Track interpolation failed for %s", sid)
            continue
        lon_acc.append(lon_full[lag_sel] % 360)

    if not lon_acc:
        return None, None
    lon_arr = np.stack(lon_acc, axis=0)
    ang = np.deg2rad(lon_arr)
    with np.errstate(invalid="ignore"):
        s = np.nanmean(np.sin(ang), axis=0)
        c = np.nanmean(np.cos(ang), axis=0)
    mean_lon = (np.rad2deg(np.arctan2(s, c)) + 360) % 360
    return LAG_HOURS[lag_sel] / 24.0, mean_lon


def _hovmoller_panel(*, ax: matplotlib.axes.Axes, data: np.ndarray,
                     mask_lo: np.ndarray, mask_hi: np.ndarray,
                     levels: np.ndarray, cmap: matplotlib.colors.Colormap,
                     title: str, cbar_label: str,
                     y_days: np.ndarray,
                     lon_range: tuple[float, float] | None = None,
                     mean_track_lag: Any = None,
                     mean_track_lon: Any = None,
                     recurv_lon_mean: Any = None,
                     y_lim: tuple[float, float] = (-2, 7),
                     with_colorbar: bool = True,
                     tick_labelsize: int = 10,
                     title_fontsize: int = 11,
                     title_pad: float | None = None,
                     show_xlabel: bool = True,
                     show_ylabel: bool = True,
                     sig_field: np.ndarray | None = None,
                     sig_mask: np.ndarray | None = None,
                     show_lon_extent_hline: bool = True,
                     x_lon: np.ndarray | None = None,
                     significance: bool = True
                     ) -> matplotlib.collections.QuadMesh:
    X = advection_kernel.LONS if x_lon is None else np.asarray(x_lon, dtype=float)
    Y = np.asarray(y_days, dtype=float)
    norm = matplotlib.colors.BoundaryNorm(levels, ncolors=cmap.N)
    im = ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm,
                       shading="nearest", rasterized=True)
    neg_levels = [lv for lv in levels if lv < 0 and lv != levels[0]]
    pos_levels = [lv for lv in levels if lv > 0 and lv != levels[-1]]
    if neg_levels:
        ax.contour(X, Y, data, levels=neg_levels,
                   colors="black", linewidths=0.4, alpha=0.7)
    if pos_levels:
        ax.contour(X, Y, data, levels=pos_levels,
                   colors="black", linewidths=0.4, alpha=0.7)

    if significance:
        if sig_mask is not None:
            sig = np.asarray(sig_mask, dtype=bool)
        else:
            sig_src = data if sig_field is None else sig_field
            sig = (sig_src < mask_lo) | (sig_src > mask_hi)
        ax.contourf(X, Y, sig.astype(int),
                    levels=[0.5, 1.5], colors="none",
                    hatches=["////"])

    ax.axhline(0, color="k", lw=0.6, ls=":", alpha=0.5)

    if lon_range is not None and show_lon_extent_hline:
        ax.hlines(0, lon_range[0], lon_range[1],
                  colors="black", linewidth=5, zorder=5)

    def _is_listlike_of_arrays(*, obj: Any) -> bool:
        if obj is None:
            return False
        if isinstance(obj, (list, tuple)):
            return any(isinstance(x, np.ndarray) for x in obj)
        return False

    if mean_track_lag is not None and mean_track_lon is not None:
        if _is_listlike_of_arrays(obj=mean_track_lon):
            lags_seq = (mean_track_lag
                        if _is_listlike_of_arrays(obj=mean_track_lag)
                        else [mean_track_lag] * len(mean_track_lon))
            for tlag, tlon in zip(lags_seq, mean_track_lon):
                if tlag is None or tlon is None:
                    continue
                ax.plot(tlon, tlag,
                        color="black", linewidth=2.4, zorder=6,
                        solid_capstyle="round")
        else:
            ax.plot(mean_track_lon, mean_track_lag,
                    color="black", linewidth=2.4, zorder=6,
                    solid_capstyle="round")
    if recurv_lon_mean is not None:
        if isinstance(recurv_lon_mean, (list, tuple, np.ndarray)):
            stars = list(recurv_lon_mean)
        else:
            stars = [recurv_lon_mean]
        for s in stars:
            if s is None or not np.isfinite(s):
                continue
            ax.plot(s, 0, marker="*", markersize=16,
                    markerfacecolor="white", markeredgecolor="black",
                    markeredgewidth=1.0, zorder=7)

    if show_ylabel:
        ax.set_ylabel("time", fontsize=title_fontsize)
    else:
        ax.set_ylabel("")
    is_storm_relative = (
        x_lon is not None and (
            float(np.nanmin(X)) < 0.0 or float(np.nanmax(X)) <= 180.0)
    )
    if show_xlabel:
        ax.set_xlabel(
            "lon rel. to center (\u00b0)" if is_storm_relative else "longitude",
            fontsize=title_fontsize,
        )
    else:
        ax.set_xlabel("")
    t_kw: dict[str, Any] = dict(fontsize=title_fontsize, loc="left")
    if title_pad is not None:
        t_kw["pad"] = title_pad
    ax.set_title(title, **t_kw)
    if is_storm_relative:
        x0, x1 = float(np.nanmin(X)), float(np.nanmax(X))
        ax.set_xlim(x0, x1)
        span = x1 - x0
        step = 50.0 if span > 180.0 else 25.0
        first = step * np.ceil(x0 / step)
        ticks = np.arange(first, x1 + 1e-6, step)
        ax.set_xticks(ticks)
    else:
        ax.set_xlim(0, 360)
        ax.xaxis.set_major_locator(
            matplotlib.ticker.FixedLocator([0, 60, 120, 180, 240, 300]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_fmt_lon))
    ax.set_ylim(*y_lim)
    y_ticks = np.arange(y_lim[0], y_lim[1] + 1, 2)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_fmt_lag))
    ax.tick_params(axis="both", labelsize=tick_labelsize)

    if with_colorbar:
        cb = plt.colorbar(im, ax=ax, orientation="horizontal",
                          pad=0.08, shrink=0.95, ticks=levels,
                          extend="both")
        cb.set_label(cbar_label, fontsize=title_fontsize)
        cb.ax.tick_params(labelsize=tick_labelsize - 1)
    return im


def _lon_segments_for_lon360_plot(*, lon: np.ndarray, lat: np.ndarray
                                  ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.size == 0:
        return
    start = 0
    for i in range(1, lon.size):
        if abs(lon[i] - lon[i - 1]) > 180.0:
            sl = lon[start:i]
            sa = lat[start:i]
            yield (sl + 360.0) % 360.0, sa
            start = i
    sl = lon[start:]
    sa = lat[start:]
    if sl.size:
        yield (sl + 360.0) % 360.0, sa


def _break_lon_wrap_artifacts(*, xs: np.ndarray, ys: np.ndarray,
                              deg_thresh: float = 179.0
                              ) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return xs, ys
    out_x, out_y = [xs[0]], [ys[0]]
    for i in range(1, xs.size):
        if abs(xs[i] - xs[i - 1]) > deg_thresh:
            out_x.append(np.nan)
            out_y.append(np.nan)
        out_x.append(xs[i])
        out_y.append(ys[i])
    return np.asarray(out_x), np.asarray(out_y)


def _natural_earth_coastlines_lon360(*, ax: matplotlib.axes.Axes,
                                     resolution: str = "110m") -> None:
    path = cartopy.io.shapereader.natural_earth(
        resolution=resolution, category="physical", name="coastline")
    reader = cartopy.io.shapereader.Reader(path)
    for geom in reader.geometries():
        if geom.geom_type == "LineString":
            lines = (geom,)
        elif geom.geom_type == "MultiLineString":
            lines = geom.geoms
        else:
            continue
        for line in lines:
            x, y = np.array(line.xy[0]), np.array(line.xy[1])
            for xs, ys in _lon_segments_for_lon360_plot(lon=x, lat=y):
                if xs.size < 2:
                    continue
                xs2, ys2 = _break_lon_wrap_artifacts(xs=xs, ys=ys)
                ax.plot(
                    xs2, ys2, color="black", linewidth=0.5,
                    solid_capstyle="round", zorder=2,
                )


def _draw_minimap(*, ax: matplotlib.axes.Axes) -> None:
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 90)
    ax.set_facecolor("white")
    ax.set_aspect("auto")
    for xv in (0, 60, 120, 180, 240, 300):
        ax.axvline(xv, color="gray", linestyle=":", linewidth=0.4,
                   alpha=0.75, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([30, 60])
    ax.set_yticklabels(["30\N{DEGREE SIGN}N", "60\N{DEGREE SIGN}N"], fontsize=10)
    _natural_earth_coastlines_lon360(ax=ax, resolution="110m")
