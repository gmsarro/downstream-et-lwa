"""QJ Fig. 3 analog using Ghinassi et al. (2018) filtered LWA as the
RWP-envelope: (F, E, R) strip composites, Monte Carlo significance, plots."""

from __future__ import annotations

import concurrent.futures
import datetime
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib
import matplotlib.axes
import matplotlib.figure
import netCDF4 as nc
import numpy as np
import pandas as pd
import scipy.ndimage
import typer
from typing_extensions import Annotated

import downstream_et_lwa.plotting.qj_hovmoller as qj3

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)

NLAT = qj3.NLAT
NLON = qj3.NLON
LATS = qj3.LATS
LONS = qj3.LONS
LAGS = qj3.LAGS
N_LAGS = qj3.N_LAGS
LAT_MIN = qj3.LAT_MIN
LAT_MAX = qj3.LAT_MAX
BASIN_SEASON = qj3.BASIN_SEASON

LWA_FILTERED_FILENAME = "lwa_filtered_{year}_{month:02d}.nc"
LWA_CLIMATOLOGY_FILENAME = "lwa_climatology.nc"

_STRIP_CACHE_KEY: tuple | None = None
_STRIP_TIMES: np.ndarray | None = None
_STRIP_E: np.ndarray | None = None
_STRIP_M: np.ndarray | None = None
_STRIP_RAW: np.ndarray | None = None
_STRIP_INDEX: dict | None = None


def load_all_strips(*, lwa_dir: Path,
                    year_start: int = 2000, year_end: int = 2023,
                    lat_min: float = LAT_MIN, lat_max: float = LAT_MAX,
                    tau_star: float = 1.9) -> None:
    global _STRIP_TIMES, _STRIP_E, _STRIP_M, _STRIP_RAW, _STRIP_INDEX
    global _STRIP_CACHE_KEY
    root = Path(lwa_dir)
    key = (str(root.resolve()), year_start, year_end, lat_min, lat_max, tau_star)
    if _STRIP_TIMES is not None and _STRIP_CACHE_KEY == key:
        return

    _STRIP_TIMES = _STRIP_E = _STRIP_M = _STRIP_RAW = _STRIP_INDEX = None
    _STRIP_CACHE_KEY = key

    sel = (LATS >= lat_min) & (LATS <= lat_max)
    w = np.cos(np.deg2rad(LATS[sel]))
    w_sum = w.sum()

    times_list, e_list, m_list, r_list = [], [], [], []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            path = root / LWA_FILTERED_FILENAME.format(year=year, month=month)
            if not path.exists():
                continue
            with nc.Dataset(path, "r") as d:
                ts_any: Any = nc.num2date(
                    d["time"][:], d["time"].units,
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True,
                )
                ts = np.atleast_1d(ts_any)
                lwa_f = np.asarray(d["lwa_filt"][:])
                hem = np.asarray(d["hem_mean_filt"][:])
                lwa_raw = np.asarray(d["lwa_raw"][:])
            thr = (tau_star * hem)[:, None, None]
            mask_b = (lwa_f > thr)
            env_thr = np.where(mask_b, lwa_f, 0.0)
            e1d = (env_thr[:, sel, :] * w[None, :, None]).sum(axis=1) / w_sum
            r1d = (lwa_raw[:, sel, :] * w[None, :, None]).sum(axis=1) / w_sum
            m1d = (e1d > 0).astype(np.float32)
            times_list.append(np.array([np.datetime64(t, "s") for t in ts]))
            e_list.append(e1d.astype(np.float32))
            m_list.append(m1d)
            r_list.append(r1d.astype(np.float32))

    if not times_list:
        raise FileNotFoundError(
            f"No lwa_filtered_*.nc under {root} for years {year_start}-{year_end}")

    _STRIP_TIMES = np.concatenate(times_list)
    _STRIP_E = np.concatenate(e_list, axis=0)
    _STRIP_M = np.concatenate(m_list, axis=0)
    _STRIP_RAW = np.concatenate(r_list, axis=0)

    order = np.argsort(_STRIP_TIMES)
    _STRIP_TIMES = _STRIP_TIMES[order]
    _STRIP_E = _STRIP_E[order]
    _STRIP_M = _STRIP_M[order]
    _STRIP_RAW = _STRIP_RAW[order]
    _STRIP_INDEX = {t: i for i, t in enumerate(_STRIP_TIMES)}

    mem_gb = (_STRIP_E.nbytes + _STRIP_M.nbytes + _STRIP_RAW.nbytes) / 1e9
    _LOG.info("Loaded %d LWA strips covering %s .. %s (mem ~%.2f GB)",
              len(_STRIP_TIMES), _STRIP_TIMES[0], _STRIP_TIMES[-1], mem_gb)


def _strip_at(*, target_dt: datetime.datetime
              ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if (_STRIP_TIMES is None or _STRIP_E is None or _STRIP_M is None
            or _STRIP_RAW is None or _STRIP_INDEX is None):
        return None, None, None
    t64 = np.datetime64(target_dt.replace(tzinfo=None), "s")
    j = _STRIP_INDEX.get(t64)
    if j is not None:
        return _STRIP_E[j], _STRIP_M[j], _STRIP_RAW[j]
    idx = int(np.searchsorted(_STRIP_TIMES, t64))
    best = None
    for cand in (idx - 1, idx):
        if 0 <= cand < len(_STRIP_TIMES):
            dt = abs((_STRIP_TIMES[cand] - t64).astype("timedelta64[s]").astype(int))
            if dt <= 3 * 3600 and (best is None or dt < best[0]):
                best = (dt, cand)
    if best is None:
        return None, None, None
    j = best[1]
    return _STRIP_E[j], _STRIP_M[j], _STRIP_RAW[j]


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
    r_sum = np.zeros_like(e_sum)
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
            e1d, m1d, r1d = _strip_at(target_dt=target)
            if e1d is None or m1d is None or r1d is None:
                continue
            if storm_relative:
                e1d = qj3._rotate_strip_relative(arr1d=e1d, rl_int=rl_int)
                m1d = qj3._rotate_strip_relative(arr1d=m1d, rl_int=rl_int)
                r1d = qj3._rotate_strip_relative(arr1d=r1d, rl_int=rl_int)
            e_sum[li] += e1d
            m_sum[li] += m1d
            r_sum[li] += r1d
            n_cases[li] += 1

    with np.errstate(invalid="ignore"):
        F = m_sum / np.maximum(n_cases, 1)
        E = np.where(m_sum > 0, e_sum / np.maximum(m_sum, 1), 0.0)
        R = r_sum / np.maximum(n_cases, 1)

    out: dict[str, Any] = dict(F=F.astype(np.float32), E=E.astype(np.float32),
                               R=R.astype(np.float32),
                               n_cases=n_cases, ref_times=ref_times)
    if storm_relative:
        out["recurv_lon_ints"] = rl_ints
    return out


def _mc_one(args: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed, ref_times_utc, year_start, year_end, recurv_lon_ints = args
    rng = np.random.default_rng(seed)
    e_sum = np.zeros((N_LAGS, NLON), dtype=np.float64)
    m_sum = np.zeros_like(e_sum)
    r_sum = np.zeros_like(e_sum)
    n_cases = np.zeros_like(e_sum, dtype=np.int32)
    rl_seq = recurv_lon_ints
    for k, ref in enumerate(ref_times_utc):
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
            e1d, m1d, r1d = _strip_at(target_dt=target)
            if e1d is None or m1d is None or r1d is None:
                continue
            if rl_seq is not None:
                e1d = qj3._rotate_strip_relative(arr1d=e1d, rl_int=rl_int)
                m1d = qj3._rotate_strip_relative(arr1d=m1d, rl_int=rl_int)
                r1d = qj3._rotate_strip_relative(arr1d=r1d, rl_int=rl_int)
            e_sum[li] += e1d
            m_sum[li] += m1d
            r_sum[li] += r1d
            n_cases[li] += 1
    with np.errstate(invalid="ignore"):
        F = m_sum / np.maximum(n_cases, 1)
        E = np.where(m_sum > 0, e_sum / np.maximum(m_sum, 1), 0.0)
        R = r_sum / np.maximum(n_cases, 1)
    return F.astype(np.float32), E.astype(np.float32), R.astype(np.float32)


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
        R_clim: np.ndarray | None = None,
        smooth_sigma: tuple[float, float] | None = None,
        smooth_mc_anomalies: bool = True,
        recurv_lon_ints: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seeds = [seed_base + i for i in range(n_iter)]
    rl_tuple = (tuple(int(r) for r in recurv_lon_ints)
                if recurv_lon_ints is not None else None)
    args = [(s, ref_times, year_start, year_end, rl_tuple) for s in seeds]
    F_draws = np.zeros((n_iter, N_LAGS, NLON), dtype=np.float32)
    E_draws = np.zeros_like(F_draws)
    R_draws = np.zeros_like(F_draws)
    _LOG.info("Running %d Monte Carlo draws with %d workers...",
              n_iter, n_workers)
    ctx = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx) as ex:
        futs = {ex.submit(_mc_one, a): i for i, a in enumerate(args)}
        for done, fut in enumerate(concurrent.futures.as_completed(futs)):
            i = futs[fut]
            F_draws[i], E_draws[i], R_draws[i] = fut.result()
            if (done + 1) % 50 == 0 or done == n_iter - 1:
                _LOG.info("  %d/%d MC draws done", done + 1, n_iter)

    if (
        smooth_mc_anomalies
        and smooth_sigma is not None
        and F_clim is not None
        and E_clim is not None
        and R_clim is not None
    ):
        for i in range(n_iter):
            F_draws[i] = scipy.ndimage.gaussian_filter(
                F_draws[i].astype(np.float64) - F_clim[None, :],
                sigma=smooth_sigma,
            ).astype(np.float32)
            E_draws[i] = scipy.ndimage.gaussian_filter(
                E_draws[i].astype(np.float64) - E_clim[None, :],
                sigma=smooth_sigma,
            ).astype(np.float32)
            R_draws[i] = scipy.ndimage.gaussian_filter(
                R_draws[i].astype(np.float64) - R_clim[None, :],
                sigma=smooth_sigma,
            ).astype(np.float32)

    return (
        np.percentile(F_draws, 2.5, axis=0),
        np.percentile(F_draws, 97.5, axis=0),
        np.percentile(E_draws, 2.5, axis=0),
        np.percentile(E_draws, 97.5, axis=0),
        np.percentile(R_draws, 2.5, axis=0),
        np.percentile(R_draws, 97.5, axis=0),
    )


def load_clim(*, season: str, clim_path: Path
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with nc.Dataset(Path(clim_path), "r") as d:
        grp = d[season]
        F = grp["freq_1d"][:].astype(np.float32)
        E = grp["ampl_1d"][:].astype(np.float32)
        R = grp["raw_1d"][:].astype(np.float32)
    return F, E, R


def _make_fig_with_maps() -> tuple[
        matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes]:
    fig = plt.figure(figsize=(12.5, 7.5))
    gs = fig.add_gridspec(2, 2,
                          height_ratios=[0.9, 5.0],
                          hspace=0.02, wspace=0.22,
                          left=0.07, right=0.985,
                          top=0.92, bottom=0.12)
    ax_map_a = fig.add_subplot(gs[0, 0])
    ax_map_b = fig.add_subplot(gs[0, 1])
    qj3._draw_minimap(ax=ax_map_a)
    qj3._draw_minimap(ax=ax_map_b)
    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])
    return fig, ax_a, ax_b


def _make_fig_single_panel() -> tuple[
        matplotlib.figure.Figure, matplotlib.axes.Axes]:
    fig = plt.figure(figsize=(7.5, 7.5))
    gs = fig.add_gridspec(2, 1,
                          height_ratios=[0.9, 5.0],
                          hspace=0.02,
                          left=0.12, right=0.975,
                          top=0.92, bottom=0.12)
    ax_map = fig.add_subplot(gs[0, 0])
    qj3._draw_minimap(ax=ax_map)
    ax = fig.add_subplot(gs[1, 0])
    return fig, ax


def plot_fig3_lwa(*, storms_df: pd.DataFrame, basin: str,
                  lwa_dir: Path,
                  out_dir: Path,
                  clim_path: Path | None = None,
                  reference: str = "recurvature",
                  n_mc: int = 500, n_workers: int = 8,
                  smooth_sigma: tuple[float, float] | None = None,
                  tau_star: float = 1.9,
                  year_start: int = 2000, year_end: int = 2022,
                  tracks_directory: Path | None = None,
                  smooth_mc_anomalies: bool = True) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    season = BASIN_SEASON.get(basin, "JJASON")
    if clim_path is None:
        clim_path = Path(lwa_dir) / LWA_CLIMATOLOGY_FILENAME

    if smooth_sigma is None:
        smooth_sigma = qj3.HOVMOLLER_GAUSSIAN_SIGMA

    load_all_strips(lwa_dir=lwa_dir, year_start=year_start, year_end=year_end,
                    tau_star=tau_star)

    _LOG.info("[%s, %s, %s] building LWA composite...", basin, reference, season)
    comp = build_composite(storms_df=storms_df, reference=reference, basin=basin)
    if len(comp["ref_times"]) == 0:
        _LOG.info("  no storms, aborting")
        return

    F_clim, E_clim, R_clim = load_clim(season=season, clim_path=clim_path)
    F_anom = comp["F"] - F_clim[None, :]
    E_anom = comp["E"] - E_clim[None, :]
    R_anom = comp["R"] - R_clim[None, :]

    (
        f_lo,
        f_hi,
        e_lo,
        e_hi,
        r_lo,
        r_hi,
    ) = monte_carlo_sig(
        ref_times=comp["ref_times"],
        n_iter=n_mc,
        year_start=year_start,
        year_end=year_end,
        n_workers=n_workers,
        F_clim=F_clim,
        E_clim=E_clim,
        R_clim=R_clim,
        smooth_sigma=smooth_sigma,
        smooth_mc_anomalies=smooth_mc_anomalies,
    )
    if smooth_mc_anomalies:
        f_lo_anom, f_hi_anom = f_lo, f_hi
        e_lo_anom, e_hi_anom = e_lo, e_hi
        r_lo_anom, r_hi_anom = r_lo, r_hi
    else:
        f_lo_anom = f_lo - F_clim[None, :]
        f_hi_anom = f_hi - F_clim[None, :]
        e_lo_anom = e_lo - E_clim[None, :]
        e_hi_anom = e_hi - E_clim[None, :]
        r_lo_anom = r_lo - R_clim[None, :]
        r_hi_anom = r_hi - R_clim[None, :]

    F_plot = scipy.ndimage.gaussian_filter(F_anom, sigma=smooth_sigma)
    E_plot = scipy.ndimage.gaussian_filter(E_anom, sigma=smooth_sigma)
    R_plot = scipy.ndimage.gaussian_filter(R_anom, sigma=smooth_sigma)

    st = storms_df[storms_df["basin"] == basin].copy()
    ref_col = "recurv_lon" if reference == "recurvature" else "et_lon"
    if ref_col in st.columns and len(st) > 0:
        lon_vals = st[ref_col].to_numpy() % 360
        lon_range: tuple[float, float] | None = (
            np.nanmin(lon_vals), np.nanmax(lon_vals))
        recurv_lon_mean: float | None = float(np.nanmean(lon_vals))
    else:
        lon_range, recurv_lon_mean = None, None
    mean_track_lag, mean_track_lon = qj3._mean_track(
        storms_df=storms_df, basin=basin, reference=reference,
        tracks_directory=tracks_directory)
    track_kw: dict[str, Any] = dict(
        lon_range=lon_range,
        mean_track_lag=mean_track_lag,
        mean_track_lon=mean_track_lon,
        recurv_lon_mean=recurv_lon_mean,
    )

    freq_levels = np.array([-24, -18, -12, -6, 0, 6, 12, 18, 24], dtype=float)
    ampl_levels = np.array([-12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0])
    raw_levels = np.array([-12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0])

    ref_label = {"recurvature": "Recurvature-relative",
                 "et": "ET-relative"}.get(reference, reference.title())

    fig, ax_a, ax_b = _make_fig_with_maps()

    sig_f = None if smooth_mc_anomalies else F_anom
    sig_e = None if smooth_mc_anomalies else E_anom
    sig_r = None if smooth_mc_anomalies else R_anom

    qj3._hovmoller_panel(
        ax=ax_a, data=F_plot * 100.0,
        mask_lo=f_lo_anom * 100.0, mask_hi=f_hi_anom * 100.0,
        levels=freq_levels, cmap=qj3._BWOR_8,
        title="(a)",
        cbar_label="LWA-based RWP frequency anomaly (%)",
        sig_field=None if sig_f is None else sig_f * 100.0,
        **track_kw,
    )
    qj3._hovmoller_panel(
        ax=ax_b, data=E_plot,
        mask_lo=e_lo_anom, mask_hi=e_hi_anom,
        levels=ampl_levels, cmap=qj3._BWOR_8,
        title="(b)",
        cbar_label=r"LWA-based RWP amplitude anomaly (m s$^{-1}$)",
        sig_field=sig_e,
        **track_kw,
    )
    fig.suptitle(
        f"{ref_label} composites (filtered LWA, Ghinassi 2018) \N{EM DASH} "
        f"{basin} recurving TCs "
        f"({season} climatology, N={len(comp['ref_times'])})",
        fontsize=11, y=0.985,
    )
    out_path = out_dir / f"fig3_lwa_{basin}_{reference}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("  wrote %s", out_path)

    fig, ax = _make_fig_single_panel()
    qj3._hovmoller_panel(
        ax=ax, data=R_plot,
        mask_lo=r_lo_anom, mask_hi=r_hi_anom,
        levels=raw_levels, cmap=qj3._BWOR_8,
        title="(a)",
        cbar_label=r"Raw LWA anomaly (m s$^{-1}$)",
        sig_field=sig_r,
        **track_kw,
    )
    fig.suptitle(
        f"{ref_label} raw-LWA composite \N{EM DASH} {basin} recurving TCs "
        f"({season} climatology, N={len(comp['ref_times'])})",
        fontsize=11, y=0.985,
    )
    out_path = out_dir / f"fig3_lwa_raw_{basin}_{reference}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("  wrote %s", out_path)


def main(
        tracks_csv: Annotated[Path, typer.Option(
            help="recurving_nh_tracks.csv from the track database")],
        lwa_directory: Annotated[Path, typer.Option(
            help="Directory with lwa_filtered_YYYY_MM.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for fig3_lwa_<basin>_<reference>.png")],
        climatology_path: Annotated[Optional[Path], typer.Option(
            help="lwa_climatology.nc (default: "
                 "<lwa-directory>/lwa_climatology.nc)")] = None,
        tracks_directory: Annotated[Optional[Path], typer.Option(
            help="Track database directory for the mean-track overlay")] = None,
        basins: Annotated[Optional[list[str]], typer.Option()] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature or et")] = "recurvature",
        n_mc: Annotated[int, typer.Option()] = 500,
        n_workers: Annotated[int, typer.Option()] = 8,
        tau_star: Annotated[float, typer.Option()] = 1.9,
        year_start: Annotated[int, typer.Option()] = 2000,
        year_end: Annotated[int, typer.Option()] = 2022,
        mc_legacy_envelope: Annotated[bool, typer.Option(
            help="MC significance from unsmoothed composites")] = False,
        sigma_lag: Annotated[Optional[float], typer.Option(
            help="Gaussian sigma in lag grid (6 h)")] = None,
        sigma_lon: Annotated[Optional[float], typer.Option(
            help="Gaussian sigma along longitude (1 deg/cell)")] = None,
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
            sigma_lag if sigma_lag is not None
            else qj3.HOVMOLLER_GAUSSIAN_SIGMA[0],
            sigma_lon if sigma_lon is not None
            else qj3.HOVMOLLER_GAUSSIAN_SIGMA[1],
        )
        plot_fig3_lwa(
            storms_df=storms_df,
            basin=basin,
            lwa_dir=lwa_directory,
            out_dir=output_directory,
            clim_path=climatology_path,
            reference=reference,
            n_mc=n_mc,
            n_workers=n_workers,
            tau_star=tau_star,
            year_start=year_start,
            year_end=year_end,
            tracks_directory=tracks_directory,
            smooth_mc_anomalies=not mc_legacy_envelope,
            smooth_sigma=smooth_sigma,
        )


if __name__ == "__main__":
    typer.run(main)
