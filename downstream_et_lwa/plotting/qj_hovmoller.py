"""Quinting & Jones (2016) Fig. 3 RWP frequency/amplitude Hovmoller composites:
1-D strip loading, composites, Monte Carlo significance, panels, and minimaps."""

from __future__ import annotations

import concurrent.futures
import datetime
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import cartopy.io.shapereader
import matplotlib
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors
import matplotlib.ticker
import netCDF4 as nc
import numpy as np
import pandas as pd
import scipy.ndimage
import typer
from typing_extensions import Annotated

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.tracks as tracks

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

NLAT = 91
NLON = 360
LATS = np.linspace(0, 90, NLAT)
LONS = np.linspace(0, 359, NLON)
COSPHI = np.cos(np.deg2rad(LATS))

HOURS_BEFORE = 24 * 7
HOURS_AFTER = 24 * 10
DT_HOURS = 6
LAGS = np.arange(-HOURS_BEFORE, HOURS_AFTER + DT_HOURS, DT_HOURS)
N_LAGS = len(LAGS)

LAT_MIN = 20.0
LAT_MAX = 80.0

BASIN_SEASON = {
    "WP": "JJASON", "EP": "JJASON", "NA": "JJASON", "NI": "JJASON",
    "SI": "DJFMA", "SP": "DJFMA", "AU": "DJFMA",
    "WPNA": "JJASON",
}

HOVMOLLER_GAUSSIAN_SIGMA = (1.0, 2.5)

RWP_ENVELOPE_FILENAME = "rwp_envelope_{year}_{month:02d}.nc"
RWP_CLIMATOLOGY_FILENAME = "rwp_climatology.nc"

REL_LON_HALF = NLON // 2
REL_LONS = np.arange(NLON) - REL_LON_HALF

_STRIP_CACHE_KEY: tuple | None = None
_STRIP_TIMES: np.ndarray | None = None
_STRIP_E: np.ndarray | None = None
_STRIP_M: np.ndarray | None = None
_STRIP_INDEX: dict | None = None


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


def _merid_avg_2d(*, arr_tlatlon: np.ndarray,
                  lat_min: float = LAT_MIN,
                  lat_max: float = LAT_MAX) -> np.ndarray:
    sel = (LATS >= lat_min) & (LATS <= lat_max)
    w = COSPHI[sel]
    sub = arr_tlatlon[..., sel, :]
    return (sub * w[:, None]).sum(axis=-2) / w.sum()


def load_all_strips(*, rwp_dir: Path,
                    year_start: int = 2000, year_end: int = 2023,
                    lat_min: float = LAT_MIN, lat_max: float = LAT_MAX) -> None:
    global _STRIP_TIMES, _STRIP_E, _STRIP_M, _STRIP_INDEX, _STRIP_CACHE_KEY
    root = Path(rwp_dir)
    key = (str(root.resolve()), year_start, year_end, lat_min, lat_max)
    if _STRIP_TIMES is not None and _STRIP_CACHE_KEY == key:
        return

    _STRIP_TIMES = _STRIP_E = _STRIP_M = _STRIP_INDEX = None
    _STRIP_CACHE_KEY = key

    times_list, e_list, m_list = [], [], []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            path = root / RWP_ENVELOPE_FILENAME.format(year=year, month=month)
            if not path.exists():
                continue
            with nc.Dataset(path, "r") as d:
                ts_any: Any = nc.num2date(
                    d["time"][:], d["time"].units,
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True,
                )
                ts = np.atleast_1d(ts_any)
                env_thr = d["envelope_thr"][:]
            e1d = _merid_avg_2d(arr_tlatlon=env_thr, lat_min=lat_min,
                                lat_max=lat_max).astype(np.float32)
            m1d = (e1d > 0).astype(np.float32)
            times_list.append(np.array([np.datetime64(t, "s") for t in ts]))
            e_list.append(e1d)
            m_list.append(m1d)

    if not times_list:
        raise FileNotFoundError(
            f"No rwp_envelope_*.nc under {root} for years {year_start}-{year_end}")

    _STRIP_TIMES = np.concatenate(times_list)
    _STRIP_E = np.concatenate(e_list, axis=0)
    _STRIP_M = np.concatenate(m_list, axis=0)

    order = np.argsort(_STRIP_TIMES)
    _STRIP_TIMES = _STRIP_TIMES[order]
    _STRIP_E = _STRIP_E[order]
    _STRIP_M = _STRIP_M[order]
    _STRIP_INDEX = {t: i for i, t in enumerate(_STRIP_TIMES)}

    _LOG.info("Loaded %d 1-D strips covering %s .. %s (mem ~%.2f GB)",
              len(_STRIP_TIMES), _STRIP_TIMES[0], _STRIP_TIMES[-1],
              (_STRIP_E.nbytes + _STRIP_M.nbytes) / 1e9)


def _strip_at(*, target_dt: datetime.datetime
              ) -> tuple[np.ndarray | None, np.ndarray | None]:
    if (_STRIP_TIMES is None or _STRIP_E is None or _STRIP_M is None
            or _STRIP_INDEX is None):
        return None, None
    t64 = np.datetime64(target_dt.replace(tzinfo=None), "s")
    j = _STRIP_INDEX.get(t64)
    if j is not None:
        return _STRIP_E[j], _STRIP_M[j]
    idx = int(np.searchsorted(_STRIP_TIMES, t64))
    best = None
    for cand in (idx - 1, idx):
        if 0 <= cand < len(_STRIP_TIMES):
            dt = abs((_STRIP_TIMES[cand] - t64).astype("timedelta64[s]").astype(int))
            if dt <= 3 * 3600 and (best is None or dt < best[0]):
                best = (dt, cand)
    if best is None:
        return None, None
    return _STRIP_E[best[1]], _STRIP_M[best[1]]


def _rotate_strip_relative(*, arr1d: np.ndarray, rl_int: int) -> np.ndarray:
    return np.roll(arr1d, REL_LON_HALF - int(rl_int))


def _resolve_full_tracks(
        *,
        full_tracks: dict[str, pd.DataFrame] | None,
        tracks_directory: Path | None,
) -> dict[str, pd.DataFrame] | None:
    if full_tracks is not None:
        return full_tracks
    if tracks_directory is None:
        return None
    try:
        _, loaded = tracks.load_track_database(tracks_directory=tracks_directory)
    except Exception:
        _LOG.exception("Could not load track database from %s", tracks_directory)
        return None
    return loaded


def _mean_track_relative(*, storms_df: pd.DataFrame, basin: str,
                         reference: str = "recurvature",
                         full_tracks: dict[str, pd.DataFrame] | None = None,
                         tracks_directory: Path | None = None
                         ) -> tuple[np.ndarray | None, np.ndarray | None]:
    full_tracks = _resolve_full_tracks(
        full_tracks=full_tracks, tracks_directory=tracks_directory)
    if full_tracks is None:
        _LOG.info("[mean_track_relative] no track database, skipping")
        return None, None

    b = str(basin).upper()
    if b == "WPNA":
        sub = storms_df[storms_df["basin"].isin(("WP", "NA"))]
    else:
        sub = storms_df[storms_df["basin"] == basin]
    lon_col = "recurv_lon" if reference == "recurvature" else "et_lon"
    all_lag_hours = composite_config.LAG_HOURS
    lag_sel = (all_lag_hours >= -48) & (all_lag_hours <= 96)
    rel_acc = []
    for _, st in sub.iterrows():
        sid = st["storm_id"]
        if sid not in full_tracks:
            continue
        rl = st.get(lon_col)
        if rl is None or pd.isna(rl):
            continue
        try:
            _, lon_full, _, _ = tracks.interpolate_track_to_lags(
                track_df=full_tracks[sid], reference=reference)
        except Exception:
            _LOG.exception("Track interpolation failed for %s", sid)
            continue
        rel = ((np.asarray(lon_full[lag_sel]) - float(rl) + 180.0) % 360.0) - 180.0
        rel_acc.append(rel)

    if not rel_acc:
        return None, None
    rel_arr = np.stack(rel_acc, axis=0)
    with np.errstate(invalid="ignore"):
        mean_rel = np.nanmean(rel_arr, axis=0)
    return all_lag_hours[lag_sel] / 24.0, mean_rel


def _relative_clim_1d(*, clim_1d: np.ndarray,
                      recurv_lon_ints: Sequence[int]) -> np.ndarray:
    if not recurv_lon_ints:
        return np.zeros_like(clim_1d, dtype=np.float32)
    acc = np.zeros_like(clim_1d, dtype=np.float64)
    n = 0
    for rl in recurv_lon_ints:
        if rl is None:
            continue
        acc += np.roll(clim_1d.astype(np.float64), REL_LON_HALF - int(rl))
        n += 1
    if n == 0:
        return np.zeros_like(clim_1d, dtype=np.float32)
    return (acc / n).astype(np.float32)


def _round_to_6h(*, dt: datetime.datetime) -> datetime.datetime:
    hr = int(round(dt.hour / 6.0)) * 6
    if hr == 24:
        dt = dt + datetime.timedelta(days=1)
        hr = 0
    return dt.replace(hour=hr, minute=0, second=0, microsecond=0)


def build_composite(*, storms_df: pd.DataFrame, reference: str = "recurvature",
                    basin: str | None = None,
                    storm_relative: bool = False) -> dict[str, Any]:
    if basin is not None:
        b = str(basin).upper()
        if b == "WPNA":
            storms_df = storms_df[storms_df["basin"].isin(("WP", "NA"))].copy()
        else:
            storms_df = storms_df[storms_df["basin"] == basin].copy()

    ref_col = "recurv_time" if reference == "recurvature" else "et_time"
    lon_col = "recurv_lon" if reference == "recurvature" else "et_lon"

    e_sum = np.zeros((N_LAGS, NLON), dtype=np.float64)
    m_sum = np.zeros_like(e_sum)
    n_cases = np.zeros_like(e_sum, dtype=np.int32)

    ref_times = []
    rl_ints: list[int] = []
    for _, st in storms_df.iterrows():
        ref_time = pd.to_datetime(st[ref_col])
        if pd.isna(ref_time):
            continue
        if storm_relative:
            rl = st.get(lon_col)
            if rl is None or pd.isna(rl):
                continue
            rl_int = int(round(float(rl) % 360.0)) % NLON
        else:
            rl_int = -1
        ref_dt = _round_to_6h(dt=ref_time.to_pydatetime())
        ref_times.append(ref_dt)
        rl_ints.append(rl_int)
        for li, lag_h in enumerate(LAGS):
            target = ref_dt + datetime.timedelta(hours=int(lag_h))
            e1d, m1d = _strip_at(target_dt=target)
            if e1d is None or m1d is None:
                continue
            if storm_relative:
                e1d = _rotate_strip_relative(arr1d=e1d, rl_int=rl_int)
                m1d = _rotate_strip_relative(arr1d=m1d, rl_int=rl_int)
            e_sum[li] += e1d
            m_sum[li] += m1d
            n_cases[li] += 1

    with np.errstate(invalid="ignore"):
        F = m_sum / np.maximum(n_cases, 1)
        E = np.where(m_sum > 0, e_sum / np.maximum(m_sum, 1), 0.0)

    out: dict[str, Any] = dict(F=F.astype(np.float32), E=E.astype(np.float32),
                               n_cases=n_cases, ref_times=ref_times)
    if storm_relative:
        out["recurv_lon_ints"] = rl_ints
    return out


def _mc_one(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    seed, ref_times_utc, year_start, year_end, years_pool, recurv_lon_ints = args
    rng = np.random.default_rng(seed)
    e_sum = np.zeros((N_LAGS, NLON), dtype=np.float64)
    m_sum = np.zeros_like(e_sum)
    n_cases = np.zeros_like(e_sum, dtype=np.int32)
    pool_arr = (
        np.asarray(years_pool, dtype=np.int64) if years_pool else None)
    rl_seq = recurv_lon_ints
    for k, ref in enumerate(ref_times_utc):
        if pool_arr is not None and pool_arr.size:
            year_r = int(pool_arr[rng.integers(pool_arr.size)])
        else:
            year_r = int(rng.integers(year_start, year_end + 1))
        doy_shift = int(rng.integers(-7, 8))
        try:
            new_ref = ref.replace(year=year_r) + datetime.timedelta(days=doy_shift)
        except ValueError:
            new_ref = (datetime.datetime(year_r, ref.month, min(ref.day, 28))
                       + datetime.timedelta(days=doy_shift))
        new_ref = _round_to_6h(dt=new_ref)
        rl_int = (int(rl_seq[k]) if rl_seq is not None else -1)
        for li, lag_h in enumerate(LAGS):
            target = new_ref + datetime.timedelta(hours=int(lag_h))
            e1d, m1d = _strip_at(target_dt=target)
            if e1d is None or m1d is None:
                continue
            if rl_seq is not None:
                e1d = _rotate_strip_relative(arr1d=e1d, rl_int=rl_int)
                m1d = _rotate_strip_relative(arr1d=m1d, rl_int=rl_int)
            e_sum[li] += e1d
            m_sum[li] += m1d
            n_cases[li] += 1
    with np.errstate(invalid="ignore"):
        F = m_sum / np.maximum(n_cases, 1)
        E = np.where(m_sum > 0, e_sum / np.maximum(m_sum, 1), 0.0)
    return F.astype(np.float32), E.astype(np.float32)


def monte_carlo_sig(
        *,
        ref_times: Sequence[datetime.datetime],
        n_iter: int = 1000,
        year_start: int = 2000,
        year_end: int = 2023,
        n_workers: int = 8,
        seed_base: int = 12345,
        F_clim: np.ndarray | None = None,
        E_clim: np.ndarray | None = None,
        smooth_sigma: tuple[float, float] | None = None,
        smooth_mc_anomalies: bool = True,
        years_pool: Sequence[int] | None = None,
        recurv_lon_ints: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seeds = [seed_base + i for i in range(n_iter)]
    pool_tuple = tuple(int(y) for y in years_pool) if years_pool else None
    rl_tuple = (tuple(int(r) for r in recurv_lon_ints)
                if recurv_lon_ints is not None else None)
    args = [(s, ref_times, year_start, year_end, pool_tuple, rl_tuple)
            for s in seeds]
    F_draws = np.zeros((n_iter, N_LAGS, NLON), dtype=np.float32)
    E_draws = np.zeros_like(F_draws)
    if pool_tuple:
        _LOG.info("Running %d Monte Carlo draws with %d workers... pool=%d years",
                  n_iter, n_workers, len(pool_tuple))
    else:
        _LOG.info("Running %d Monte Carlo draws with %d workers... range=%d..%d",
                  n_iter, n_workers, year_start, year_end)
    ctx = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx) as ex:
        futs = {ex.submit(_mc_one, a): i for i, a in enumerate(args)}
        for done, fut in enumerate(concurrent.futures.as_completed(futs)):
            i = futs[fut]
            F_draws[i], E_draws[i] = fut.result()
            if (done + 1) % 50 == 0 or done == n_iter - 1:
                _LOG.info("  %d/%d MC draws done", done + 1, n_iter)

    if (
        smooth_mc_anomalies
        and smooth_sigma is not None
        and F_clim is not None
        and E_clim is not None
    ):
        for i in range(n_iter):
            fa = scipy.ndimage.gaussian_filter(
                F_draws[i].astype(np.float64) - F_clim[None, :],
                sigma=smooth_sigma,
            ).astype(np.float32)
            ea = scipy.ndimage.gaussian_filter(
                E_draws[i].astype(np.float64) - E_clim[None, :],
                sigma=smooth_sigma,
            ).astype(np.float32)
            F_draws[i] = fa
            E_draws[i] = ea

    return (
        np.percentile(F_draws, 2.5, axis=0),
        np.percentile(F_draws, 97.5, axis=0),
        np.percentile(E_draws, 2.5, axis=0),
        np.percentile(E_draws, 97.5, axis=0),
    )


def _mean_track(*, storms_df: pd.DataFrame, basin: str,
                reference: str = "recurvature",
                full_tracks: dict[str, pd.DataFrame] | None = None,
                tracks_directory: Path | None = None
                ) -> tuple[np.ndarray | None, np.ndarray | None]:
    full_tracks = _resolve_full_tracks(
        full_tracks=full_tracks, tracks_directory=tracks_directory)
    if full_tracks is None:
        _LOG.info("[mean_track] no track database, skipping")
        return None, None

    b = str(basin).upper()
    if b == "WPNA":
        sub = storms_df[storms_df["basin"].isin(("WP", "NA"))]
    else:
        sub = storms_df[storms_df["basin"] == basin]
    all_lag_hours = composite_config.LAG_HOURS
    lag_sel = (all_lag_hours >= -48) & (all_lag_hours <= 96)
    lon_acc = []
    for _, st in sub.iterrows():
        sid = st["storm_id"]
        if sid not in full_tracks:
            continue
        try:
            _, lon_full, _, _ = tracks.interpolate_track_to_lags(
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
    return all_lag_hours[lag_sel] / 24.0, mean_lon


def load_clim(*, season: str, clim_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with nc.Dataset(Path(clim_path), "r") as d:
        grp = d[season]
        F = grp["freq_1d"][:].astype(np.float32)
        E = grp["ampl_1d"][:].astype(np.float32)
    return F, E


def _hovmoller_panel(*, ax: matplotlib.axes.Axes, data: np.ndarray,
                     mask_lo: np.ndarray, mask_hi: np.ndarray,
                     levels: np.ndarray, cmap: matplotlib.colors.Colormap,
                     title: str, cbar_label: str,
                     lon_range: tuple[float, float] | None = None,
                     mean_track_lag: Any = None,
                     mean_track_lon: Any = None,
                     recurv_lon_mean: Any = None,
                     y_lim: tuple[float, float] = (-2, 7),
                     tick_pad: float = 0.1,
                     with_colorbar: bool = True,
                     tick_labelsize: int = 12,
                     title_fontsize: int = 13,
                     title_pad: float | None = None,
                     show_xlabel: bool = True,
                     show_ylabel: bool = True,
                     sig_field: np.ndarray | None = None,
                     sig_mask: np.ndarray | None = None,
                     show_lon_extent_hline: bool = True,
                     x_lon: np.ndarray | None = None,
                     significance: bool = True,
                     y_days: np.ndarray | None = None
                     ) -> matplotlib.collections.QuadMesh:
    X = LONS if x_lon is None else np.asarray(x_lon, dtype=float)
    Y = np.asarray(y_days, dtype=float) if y_days is not None else (LAGS / 24.0)
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

    def _is_listlike_of_arrays(obj: Any) -> bool:
        if obj is None:
            return False
        if isinstance(obj, (list, tuple)):
            return any(isinstance(x, np.ndarray) for x in obj)
        return False

    if mean_track_lag is not None and mean_track_lon is not None:
        if _is_listlike_of_arrays(mean_track_lon):
            lags_seq = (mean_track_lag if _is_listlike_of_arrays(mean_track_lag)
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


def plot_fig3(*, storms_df: pd.DataFrame, basin: str,
              rwp_dir: Path,
              out_dir: Path,
              clim_path: Path | None = None,
              reference: str = "recurvature",
              n_mc: int = 500, n_workers: int = 8,
              smooth_sigma: tuple[float, float] = HOVMOLLER_GAUSSIAN_SIGMA,
              strip_year_start: int | None = None,
              strip_year_end: int | None = None,
              mc_year_start: int | None = None,
              mc_year_end: int | None = None,
              full_tracks: dict[str, pd.DataFrame] | None = None,
              tracks_directory: Path | None = None,
              out_suffix: str = "",
              title_tag: str | None = None,
              smooth_mc_anomalies: bool = True) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    season = BASIN_SEASON.get(basin, "JJASON")
    if clim_path is None:
        clim_path = Path(rwp_dir) / RWP_CLIMATOLOGY_FILENAME

    ys = strip_year_start if strip_year_start is not None else 2000
    ye = strip_year_end if strip_year_end is not None else 2023
    load_all_strips(rwp_dir=rwp_dir, year_start=ys, year_end=ye)

    mc_lo = mc_year_start if mc_year_start is not None else ys
    mc_hi = mc_year_end if mc_year_end is not None else ye

    _LOG.info("[%s, %s, %s] building composite...", basin, reference, season)
    comp = build_composite(storms_df=storms_df, reference=reference, basin=basin)
    if len(comp["ref_times"]) == 0:
        _LOG.info("  no storms, aborting")
        return

    F_clim, E_clim = load_clim(season=season, clim_path=clim_path)

    F_anom = comp["F"] - F_clim[None, :]
    E_anom = comp["E"] - E_clim[None, :]

    f_lo, f_hi, e_lo, e_hi = monte_carlo_sig(
        ref_times=comp["ref_times"],
        n_iter=n_mc,
        year_start=mc_lo,
        year_end=mc_hi,
        n_workers=n_workers,
        F_clim=F_clim,
        E_clim=E_clim,
        smooth_sigma=smooth_sigma,
        smooth_mc_anomalies=smooth_mc_anomalies,
    )
    if smooth_mc_anomalies:
        f_lo_anom, f_hi_anom = f_lo, f_hi
        e_lo_anom, e_hi_anom = e_lo, e_hi
    else:
        f_lo_anom = f_lo - F_clim[None, :]
        f_hi_anom = f_hi - F_clim[None, :]
        e_lo_anom = e_lo - E_clim[None, :]
        e_hi_anom = e_hi - E_clim[None, :]

    F_plot = scipy.ndimage.gaussian_filter(F_anom, sigma=smooth_sigma)
    E_plot = scipy.ndimage.gaussian_filter(E_anom, sigma=smooth_sigma)

    st = storms_df[storms_df["basin"] == basin].copy()
    ref_col = "recurv_lon" if reference == "recurvature" else "et_lon"
    if ref_col in st.columns and len(st) > 0:
        lon_vals = st[ref_col].to_numpy() % 360
        lon_range: tuple[float, float] | None = (
            np.nanmin(lon_vals), np.nanmax(lon_vals))
        recurv_lon_mean: float | None = float(np.nanmean(lon_vals))
    else:
        lon_range, recurv_lon_mean = None, None

    mean_track_lag, mean_track_lon = _mean_track(
        storms_df=storms_df, basin=basin, reference=reference,
        full_tracks=full_tracks, tracks_directory=tracks_directory)

    freq_levels = np.array([-24, -18, -12, -6, 0, 6, 12, 18, 24], dtype=float)
    ampl_levels = np.array([-2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4])

    fig = plt.figure(figsize=(12.5, 7.5))
    gs = fig.add_gridspec(2, 2,
                          height_ratios=[0.9, 5.0],
                          hspace=0.02, wspace=0.22,
                          left=0.07, right=0.985,
                          top=0.92, bottom=0.12)

    ax_map_a = fig.add_subplot(gs[0, 0])
    ax_map_b = fig.add_subplot(gs[0, 1])
    _draw_minimap(ax=ax_map_a)
    _draw_minimap(ax=ax_map_b)

    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])

    track_kw: dict[str, Any] = dict(
        lon_range=lon_range,
        mean_track_lag=mean_track_lag,
        mean_track_lon=mean_track_lon,
        recurv_lon_mean=recurv_lon_mean,
    )

    sig_f = None if smooth_mc_anomalies else F_anom * 100.0
    sig_e = None if smooth_mc_anomalies else E_anom
    _hovmoller_panel(
        ax=ax_a, data=F_plot * 100.0,
        mask_lo=f_lo_anom * 100.0, mask_hi=f_hi_anom * 100.0,
        levels=freq_levels, cmap=_BWOR_8,
        title="(a)",
        cbar_label="RWP frequency anomaly (%)",
        sig_field=sig_f,
        **track_kw,
    )

    _hovmoller_panel(
        ax=ax_b, data=E_plot,
        mask_lo=e_lo_anom, mask_hi=e_hi_anom,
        levels=ampl_levels, cmap=_BWOR_8,
        title="(b)",
        cbar_label=r"RWP amplitude anomaly (m s$^{-1}$)",
        sig_field=sig_e,
        **track_kw,
    )

    ref_label = {"recurvature": "Recurvature-relative",
                 "et": "ET-relative"}.get(reference, reference.title())
    tag = f" \N{EM DASH} {title_tag}" if title_tag else ""
    fig.suptitle(
        f"{ref_label} composites \N{EM DASH} {basin} recurving TCs "
        f"({season} climatology, N={len(comp['ref_times'])}){tag}",
        fontsize=11, y=0.985,
    )

    fn_suffix = f"_{out_suffix}" if out_suffix else ""
    out_path = out_dir / f"fig3_{basin}_{reference}{fn_suffix}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("  wrote %s", out_path)


def main(
        tracks_csv: Annotated[Path, typer.Option(
            help="recurving_nh_tracks.csv from the track database")],
        rwp_directory: Annotated[Path, typer.Option(
            help="Directory with rwp_envelope_YYYY_MM.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for fig3_<basin>_<reference>.png")],
        climatology_path: Annotated[Optional[Path], typer.Option(
            help="rwp_climatology.nc (default: "
                 "<rwp-directory>/rwp_climatology.nc)")] = None,
        tracks_directory: Annotated[Optional[Path], typer.Option(
            help="Track database directory for the mean-track overlay")] = None,
        basins: Annotated[Optional[list[str]], typer.Option()] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature or et")] = "recurvature",
        n_mc: Annotated[int, typer.Option()] = 500,
        n_workers: Annotated[int, typer.Option()] = 8,
        strip_year_start: Annotated[Optional[int], typer.Option()] = None,
        strip_year_end: Annotated[Optional[int], typer.Option()] = None,
        mc_year_start: Annotated[Optional[int], typer.Option()] = None,
        mc_year_end: Annotated[Optional[int], typer.Option()] = None,
        out_suffix: Annotated[str, typer.Option(
            help="Optional tag before .png, e.g. mpas_current")] = "",
        title_tag: Annotated[Optional[str], typer.Option(
            help="Optional text appended to figure suptitle")] = None,
        mc_legacy_envelope: Annotated[bool, typer.Option(
            help="Monte Carlo: raw-composite percentiles then subtract "
                 "climatology (legacy; wider null)")] = False,
        sigma_lag: Annotated[Optional[float], typer.Option(
            help="Gaussian sigma in lag grid (6 h)")] = None,
        sigma_lon: Annotated[Optional[float], typer.Option(
            help="Gaussian sigma along longitude (1 deg/column)")] = None,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if basins is None:
        basins = ["WP"]

    storms_df = pd.read_csv(
        tracks_csv,
        parse_dates=["recurv_time", "et_time"],
        keep_default_na=False,
        na_values=[""],
    )

    for basin in basins:
        smooth_sigma = (
            sigma_lag if sigma_lag is not None else HOVMOLLER_GAUSSIAN_SIGMA[0],
            sigma_lon if sigma_lon is not None else HOVMOLLER_GAUSSIAN_SIGMA[1],
        )
        plot_fig3(storms_df=storms_df, basin=basin, reference=reference,
                  n_mc=n_mc, n_workers=n_workers,
                  out_dir=output_directory,
                  rwp_dir=rwp_directory,
                  clim_path=climatology_path,
                  tracks_directory=tracks_directory,
                  strip_year_start=strip_year_start,
                  strip_year_end=strip_year_end,
                  mc_year_start=mc_year_start,
                  mc_year_end=mc_year_end,
                  out_suffix=out_suffix,
                  title_tag=title_tag,
                  smooth_mc_anomalies=not mc_legacy_envelope,
                  smooth_sigma=smooth_sigma)


if __name__ == "__main__":
    typer.run(main)
