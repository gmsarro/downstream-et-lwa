"""1-D meridional strips of MPAS barotropic LWA (LWAb_N) for Hovmoller-style
composites, mirroring the cos(lat)-weighted 20-80N average used for ERA5
Ghinassi strips; includes the Fig. 2-style scenario RWP/LWA row builder."""

from __future__ import annotations

import concurrent.futures
import datetime
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Sequence

import matplotlib.figure
import matplotlib.gridspec
import netCDF4 as nc
import numpy as np
import pandas as pd
import scipy.ndimage

import downstream_et_lwa.composites.mpas_composites as mpas_composites
import downstream_et_lwa.grid_utils as grid_utils
import downstream_et_lwa.plotting.qj_hovmoller as qj3

_LOG = logging.getLogger(__name__)

JJASON = {6, 7, 8, 9, 10, 11}

_STRIP_TIMES: np.ndarray | None = None
_STRIP_R: np.ndarray | None = None
_STRIP_INDEX: dict | None = None
_CACHE_KEY: tuple | None = None


def _meridional_mean_lwab(*, field_2d: np.ndarray) -> np.ndarray:
    sel = (qj3.LATS >= qj3.LAT_MIN) & (qj3.LATS <= qj3.LAT_MAX)
    w = np.cos(np.deg2rad(qj3.LATS[sel]))
    w_sum = float(w.sum())
    return (field_2d[sel, :] * w[:, None]).sum(axis=0) / w_sum


def load_mpas_lwab_strips(
        *,
        budget_root: Path,
        scenario: str,
        year_start: int,
        year_end: int,
        lat_min: float = qj3.LAT_MIN,
        lat_max: float = qj3.LAT_MAX,
) -> None:
    global _STRIP_TIMES, _STRIP_R, _STRIP_INDEX, _CACHE_KEY
    root = Path(budget_root)
    key = (str(root.resolve()), scenario, year_start, year_end,
           lat_min, lat_max)
    if _STRIP_TIMES is not None and _CACHE_KEY == key:
        return
    _STRIP_TIMES = _STRIP_R = _STRIP_INDEX = None
    _CACHE_KEY = key

    sel = (qj3.LATS >= lat_min) & (qj3.LATS <= lat_max)
    w = np.cos(np.deg2rad(qj3.LATS[sel]))
    w_sum = float(w.sum())

    times_list: list[np.datetime64] = []
    r_list: list[np.ndarray] = []

    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            path = mpas_composites.resolve_mpas_baro_month_nc(
                budget_root=str(root), scenario=scenario,
                year=year, month=month, suffix_stem="LWAb_N")
            if path is None or not Path(path).is_file():
                continue
            with nc.Dataset(path, "r") as d:
                lwa = np.asarray(d["lwa"][:], dtype=np.float64)
            nt, nmon, nlat, nlon = lwa.shape
            if nmon != 12:
                raise ValueError(f"Unexpected LWAb shape in {path}")
            mi = month - 1
            for ti in range(nt):
                day = ti // 4 + 1
                hour = (ti % 4) * 6
                try:
                    tdt = datetime.datetime(year, month, day, hour)
                except ValueError:
                    continue
                slab = grid_utils.deseam_longitude(field=lwa[ti, mi, :, :])
                assert slab is not None
                r1d = (slab[sel, :] * w[:, None]).sum(axis=0) / w_sum
                times_list.append(np.datetime64(tdt, "s"))
                r_list.append(r1d.astype(np.float32))

    if not times_list:
        raise FileNotFoundError(
            f"No MPAS LWAb strips for scenario={scenario!r} "
            f"{year_start}-{year_end} under {root}")

    _STRIP_TIMES = np.asarray(times_list, dtype="datetime64[s]")
    _STRIP_R = np.stack(r_list, axis=0)
    order = np.argsort(_STRIP_TIMES)
    _STRIP_TIMES = _STRIP_TIMES[order]
    _STRIP_R = _STRIP_R[order]
    _STRIP_INDEX = {t: i for i, t in enumerate(_STRIP_TIMES)}
    mem = _STRIP_R.nbytes / 1e9
    _LOG.info("[MPAS/%s] LWAb strips: %d times (%s .. %s), mem ~%.2f GB",
              scenario, len(_STRIP_TIMES), _STRIP_TIMES[0], _STRIP_TIMES[-1],
              mem)


def _strip_at(*, target_dt: datetime.datetime) -> np.ndarray | None:
    if _STRIP_TIMES is None or _STRIP_R is None:
        return None
    t64 = np.datetime64(target_dt.replace(tzinfo=None), "s")
    j = _STRIP_INDEX.get(t64) if _STRIP_INDEX is not None else None
    if j is not None:
        return _STRIP_R[j].copy()
    idx = int(np.searchsorted(_STRIP_TIMES, t64))
    best = None
    for cand in (idx - 1, idx):
        if 0 <= cand < len(_STRIP_TIMES):
            dt = abs((_STRIP_TIMES[cand] - t64)
                     .astype("timedelta64[s]").astype(int))
            if dt <= 3 * 3600 and (best is None or dt < best[0]):
                best = (dt, cand)
    if best is None:
        return None
    return _STRIP_R[best[1]].copy()


def build_raw_lwa_composite(
        *,
        storms_df: pd.DataFrame,
        reference: str = "recurvature",
        basin: str | None = "WPNA",
        storm_relative: bool = False,
) -> dict[str, Any]:
    if basin is not None:
        b = str(basin).upper()
        if b == "WPNA":
            storms_df = storms_df[storms_df["basin"].isin(("WP", "NA"))].copy()
        else:
            storms_df = storms_df[storms_df["basin"] == basin].copy()

    ref_col = "recurv_time" if reference == "recurvature" else "et_time"
    lon_col = "recurv_lon" if reference == "recurvature" else "et_lon"
    r_sum = np.zeros((qj3.N_LAGS, qj3.NLON), dtype=np.float64)
    n_cases = np.zeros_like(r_sum, dtype=np.int32)
    ref_times: list[datetime.datetime] = []
    rl_ints: list[int] = []

    for _, st in storms_df.iterrows():
        ref_time = pd.to_datetime(st[ref_col])
        if pd.isna(ref_time):
            continue
        if storm_relative:
            rl = st.get(lon_col)
            if rl is None or pd.isna(rl):
                continue
            rl_int = int(round(float(rl) % 360.0)) % qj3.NLON
        else:
            rl_int = -1
        ref_dt = qj3._round_to_6h(dt=ref_time.to_pydatetime())
        ref_times.append(ref_dt)
        rl_ints.append(rl_int)
        for li, lag_h in enumerate(qj3.LAGS):
            target = ref_dt + datetime.timedelta(hours=int(lag_h))
            r1d = _strip_at(target_dt=target)
            if r1d is None:
                continue
            if storm_relative:
                r1d = qj3._rotate_strip_relative(arr1d=r1d, rl_int=rl_int)
            r_sum[li] += r1d
            n_cases[li] += 1

    with np.errstate(invalid="ignore"):
        R = r_sum / np.maximum(n_cases, 1)
    out: dict[str, Any] = dict(R=R.astype(np.float32), n_cases=n_cases,
                               ref_times=ref_times)
    if storm_relative:
        out["recurv_lon_ints"] = rl_ints
    return out


def _r_clim_jjason_lon_only() -> np.ndarray:
    if _STRIP_TIMES is None or _STRIP_R is None:
        return np.zeros(qj3.NLON, dtype=np.float32)
    jj = np.array([int(str(t)[5:7]) in JJASON for t in _STRIP_TIMES])
    if not jj.any():
        return np.zeros(qj3.NLON, dtype=np.float32)
    return np.nanmean(_STRIP_R[jj], axis=0).astype(np.float32)


def _mc_one_raw(args: tuple) -> np.ndarray:
    seed, ref_times_utc, year_start, year_end, years_pool, recurv_lon_ints = (
        args)
    rng = np.random.default_rng(seed)
    r_sum = np.zeros((qj3.N_LAGS, qj3.NLON), dtype=np.float64)
    n_cases = np.zeros_like(r_sum, dtype=np.int32)
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
            new_ref = (ref.replace(year=year_r)
                       + datetime.timedelta(days=doy_shift))
        except ValueError:
            new_ref = (datetime.datetime(year_r, ref.month, min(ref.day, 28))
                       + datetime.timedelta(days=doy_shift))
        new_ref = qj3._round_to_6h(dt=new_ref)
        rl_int = (int(rl_seq[k]) if rl_seq is not None else -1)
        for li, lag_h in enumerate(qj3.LAGS):
            target = new_ref + datetime.timedelta(hours=int(lag_h))
            r1d = _strip_at(target_dt=target)
            if r1d is None:
                continue
            if rl_seq is not None:
                r1d = qj3._rotate_strip_relative(arr1d=r1d, rl_int=rl_int)
            r_sum[li] += r1d
            n_cases[li] += 1
    with np.errstate(invalid="ignore"):
        R = r_sum / np.maximum(n_cases, 1)
    return R.astype(np.float32)


def monte_carlo_raw_lwa(
        *,
        ref_times: Sequence[datetime.datetime],
        R_clim: np.ndarray,
        smooth_sigma: tuple[float, float] | None,
        smooth_mc_anomalies: bool,
        n_iter: int = 300,
        year_start: int = 1988,
        year_end: int = 2016,
        n_workers: int = 8,
        seed_base: int = 45123,
        years_pool: Sequence[int] | None = None,
        recurv_lon_ints: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    seeds = [seed_base + i for i in range(n_iter)]
    pool_tuple = tuple(int(y) for y in years_pool) if years_pool else None
    rl_tuple = (tuple(int(r) for r in recurv_lon_ints)
                if recurv_lon_ints is not None else None)
    args = [(s, ref_times, year_start, year_end, pool_tuple, rl_tuple)
            for s in seeds]
    R_draws = np.zeros((n_iter, qj3.N_LAGS, qj3.NLON), dtype=np.float32)
    if pool_tuple:
        _LOG.info("Running %d Monte Carlo draws (MPAS raw LWA)... pool=%d years",
                  n_iter, len(pool_tuple))
    else:
        _LOG.info("Running %d Monte Carlo draws (MPAS raw LWA)... range=%d..%d",
                  n_iter, year_start, year_end)
    ctx = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx) as ex:
        futs = {ex.submit(_mc_one_raw, a): i for i, a in enumerate(args)}
        for done, fut in enumerate(concurrent.futures.as_completed(futs)):
            i = futs[fut]
            R_draws[i] = fut.result()
            if (done + 1) % 50 == 0 or done == n_iter - 1:
                _LOG.info("  %d/%d MC draws done", done + 1, n_iter)

    if (smooth_mc_anomalies and smooth_sigma is not None
            and R_clim is not None):
        for i in range(n_iter):
            R_draws[i] = scipy.ndimage.gaussian_filter(
                R_draws[i].astype(np.float64) - R_clim[None, :],
                sigma=smooth_sigma,
            ).astype(np.float32)
        return (
            np.percentile(R_draws, 2.5, axis=0),
            np.percentile(R_draws, 97.5, axis=0),
        )
    return (
        np.percentile(R_draws, 2.5, axis=0),
        np.percentile(R_draws, 97.5, axis=0),
    )


def scenario_rwp_lwa_row(
        *,
        storms_df: pd.DataFrame,
        scenario: str,
        reference: str,
        rwp_dir: Path,
        clim_rwp: Path,
        mpas_budget_root: Path,
        year_strip_lo: int,
        year_strip_hi: int,
        mc_year_lo: int,
        mc_year_hi: int,
        n_mc: int,
        n_workers: int,
        smooth_sigma: tuple[float, float],
        smooth_mc_anomalies: bool,
        tau_star: float,
        fig: matplotlib.figure.Figure,
        gs: matplotlib.gridspec.GridSpec,
        map_row: int,
        hov_row: int,
        labels: tuple[str, str, str],
        hov_axes: list,
        ims: list,
        full_tracks: dict[str, pd.DataFrame] | None,
        lwab_year_lo: int,
        lwab_year_hi: int,
        mc_years_pool: Sequence[int] | None = None,
        basin: str = "WPNA",
        storm_relative: bool = False,
        show_minimap: bool = True,
) -> int:
    season = qj3.BASIN_SEASON.get(basin, qj3.BASIN_SEASON["WP"])

    qj3.load_all_strips(
        rwp_dir=rwp_dir, year_start=year_strip_lo, year_end=year_strip_hi)
    load_mpas_lwab_strips(
        budget_root=mpas_budget_root, scenario=scenario,
        year_start=lwab_year_lo, year_end=lwab_year_hi)

    comp = qj3.build_composite(storms_df=storms_df, reference=reference,
                               basin=basin, storm_relative=storm_relative)
    if len(comp["ref_times"]) == 0:
        raise RuntimeError(f"No MPAS/{scenario} storms for composite")

    F_clim_abs, E_clim_abs = qj3.load_clim(season=season, clim_path=clim_rwp)
    rl_ints: Sequence[int] | None
    if storm_relative:
        rl_ints = comp.get("recurv_lon_ints", [])
        F_clim = qj3._relative_clim_1d(clim_1d=F_clim_abs,
                                       recurv_lon_ints=rl_ints)
        E_clim = qj3._relative_clim_1d(clim_1d=E_clim_abs,
                                       recurv_lon_ints=rl_ints)
    else:
        rl_ints = None
        F_clim = F_clim_abs
        E_clim = E_clim_abs
    F_anom = comp["F"] - F_clim[None, :]
    E_anom = comp["E"] - E_clim[None, :]
    f_lo, f_hi, e_lo, e_hi = qj3.monte_carlo_sig(
        ref_times=comp["ref_times"],
        n_iter=n_mc,
        year_start=mc_year_lo,
        year_end=mc_year_hi,
        n_workers=n_workers,
        F_clim=F_clim,
        E_clim=E_clim,
        smooth_sigma=smooth_sigma,
        smooth_mc_anomalies=smooth_mc_anomalies,
        years_pool=mc_years_pool,
        recurv_lon_ints=rl_ints,
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

    comp_lwa = build_raw_lwa_composite(storms_df=storms_df,
                                       reference=reference,
                                       basin=basin,
                                       storm_relative=storm_relative)
    R_clim_1d_abs = _r_clim_jjason_lon_only()
    rl_ints_lwa: Sequence[int] | None
    if storm_relative:
        rl_ints_lwa = comp_lwa.get("recurv_lon_ints", [])
        R_clim_1d = qj3._relative_clim_1d(clim_1d=R_clim_1d_abs,
                                          recurv_lon_ints=rl_ints_lwa)
    else:
        rl_ints_lwa = None
        R_clim_1d = R_clim_1d_abs
    R_anom = comp_lwa["R"] - R_clim_1d[None, :]
    r_lo, r_hi = monte_carlo_raw_lwa(
        ref_times=comp_lwa["ref_times"],
        n_iter=n_mc,
        year_start=mc_year_lo,
        year_end=mc_year_hi,
        n_workers=n_workers,
        R_clim=R_clim_1d,
        smooth_sigma=smooth_sigma,
        smooth_mc_anomalies=smooth_mc_anomalies,
        years_pool=mc_years_pool,
        recurv_lon_ints=rl_ints_lwa,
    )
    if smooth_mc_anomalies:
        r_lo_anom, r_hi_anom = r_lo, r_hi
    else:
        r_lo_anom = r_lo - R_clim_1d[None, :]
        r_hi_anom = r_hi - R_clim_1d[None, :]
    R_plot = scipy.ndimage.gaussian_filter(R_anom, sigma=smooth_sigma)

    bup = str(basin).upper()
    lon_range: tuple[float, float] | None
    recurv_lon_mean: float | None
    if storm_relative:
        lon_range = None
        recurv_lon_mean = 0.0
        mean_track_lag, mean_track_lon = qj3._mean_track_relative(
            storms_df=storms_df, basin=basin, reference=reference,
            full_tracks=full_tracks)
        x_lon = qj3.REL_LONS
    else:
        ref_col = "recurv_lon" if reference == "recurvature" else "et_lon"
        if bup == "WPNA":
            st = storms_df[storms_df["basin"].isin(("WP", "NA"))].copy()
        else:
            st = storms_df[storms_df["basin"] == basin].copy()
        if ref_col in st.columns and len(st) > 0:
            lon_vals = st[ref_col].to_numpy() % 360
            lon_range = (float(np.nanmin(lon_vals)),
                         float(np.nanmax(lon_vals)))
            recurv_lon_mean = float(np.nanmean(lon_vals))
        else:
            lon_range, recurv_lon_mean = None, None
        mean_track_lag, mean_track_lon = qj3._mean_track(
            storms_df=storms_df, basin=basin, reference=reference,
            full_tracks=full_tracks)
        x_lon = None

    freq_levels = np.array([-24, -18, -12, -6, 0, 6, 12, 18, 24], dtype=float)
    ampl_levels = np.array([-2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4])
    raw_levels = np.array([-12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0])

    if show_minimap:
        for c in range(3):
            ax_m = fig.add_subplot(gs[map_row, c])
            qj3._draw_minimap(ax=ax_m)
            ax_m.set_title("", fontsize=9)

    track_kw: dict[str, Any] = dict(
        lon_range=lon_range,
        mean_track_lag=mean_track_lag,
        mean_track_lon=mean_track_lon,
        recurv_lon_mean=recurv_lon_mean,
        show_lon_extent_hline=False,
    )
    kw: dict[str, Any] = dict(
        tick_labelsize=11,
        title_fontsize=12,
        with_colorbar=False,
        x_lon=x_lon,
        **track_kw,
    )
    sig_f = None if smooth_mc_anomalies else F_anom * 100.0
    sig_e = None if smooth_mc_anomalies else E_anom
    sig_r = None if smooth_mc_anomalies else R_anom

    ax0 = fig.add_subplot(gs[hov_row, 0])
    im0 = qj3._hovmoller_panel(
        ax=ax0, data=F_plot, mask_lo=f_lo_p, mask_hi=f_hi_p,
        levels=freq_levels, cmap=qj3._BWOR_8,
        title=labels[0],
        cbar_label="RWP frequency anomaly (%)",
        sig_field=sig_f,
        **kw,
    )
    ax1 = fig.add_subplot(gs[hov_row, 1])
    im1 = qj3._hovmoller_panel(
        ax=ax1, data=E_plot, mask_lo=e_lo_p, mask_hi=e_hi_p,
        levels=ampl_levels, cmap=qj3._BWOR_8,
        title=labels[1],
        cbar_label=r"RWP amplitude anomaly (m s$^{-1}$)",
        sig_field=sig_e,
        **kw,
    )
    ax2 = fig.add_subplot(gs[hov_row, 2])
    im2 = qj3._hovmoller_panel(
        ax=ax2, data=R_plot, mask_lo=r_lo_anom, mask_hi=r_hi_anom,
        levels=raw_levels, cmap=qj3._BWOR_8,
        title=labels[2],
        cbar_label=r"Barotropic LWA anomaly (m s$^{-1}$) (MPAS LWAb)",
        sig_field=sig_r,
        **kw,
    )
    hov_axes.extend([ax0, ax1, ax2])
    ims.extend([
        (im0, freq_levels, "RWP frequency anomaly (%)"),
        (im1, ampl_levels, r"RWP amplitude anomaly (m s$^{-1}$)"),
        (im2, raw_levels, r"LWA anomaly (m s$^{-1}$) (MPAS)"),
    ])
    return len(comp["ref_times"])
