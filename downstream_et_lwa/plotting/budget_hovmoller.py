"""QJ-Fig-3-style Hovmoller composites of the LWA budget and physical source
terms (ERA5 / MERRA-2), including WP|NA combined and RWB-stratified layouts."""

from __future__ import annotations

import concurrent.futures
import datetime
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.gridspec
import matplotlib.lines
import netCDF4 as nc
import numpy as np
import pandas as pd
import scipy.ndimage
import typer
from typing_extensions import Annotated

import downstream_et_lwa.plotting.qj_hovmoller as qj3
import downstream_et_lwa.tracks as tracks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)

NLON = qj3.NLON
LONS = qj3.LONS
LAGS = qj3.LAGS
N_LAGS = qj3.N_LAGS
BASIN_SEASON = qj3.BASIN_SEASON

SOURCE_CONFIG: dict[str, dict[str, Any]] = {
    "era5": {
        "extra_terms": ("lh_lwa", "nonqg_lwa"),
        "panel_order": ("tendency", "termI", "termII", "termIII",
                        "residual", "lh_lwa", "nonqg_lwa"),
        "panel_title": {
            "tendency": r"(a) $\partial A / \partial t$  (tendency)",
            "termI":    r"(b) Term I: $-\partial F_\lambda / \partial x$",
            "termII":   r"(c) Term II: meridional flux",
            "termIII":  r"(d) Term III",
            "residual": r"(e) Residual  (diabatic closure)",
            "lh_lwa":   r"(f) LH-LWA source",
            "nonqg_lwa": r"(a) Non-QG source (ERA5)",
        },
    },
    "merra2": {
        "extra_terms": ("lh_lwa", "rad_lwa", "ana_lwa", "tot_lwa"),
        "panel_order": ("tendency", "termI", "termII", "termIII",
                        "residual",
                        "lh_lwa", "rad_lwa", "ana_lwa", "tot_lwa",
                        "lwa_raw"),
        "panel_title": {
            "tendency": r"(a) $\partial A / \partial t$  (tendency)",
            "termI":    r"(b) Term I: $-\partial F_\lambda / \partial x$",
            "termII":   r"(c) Term II: meridional flux",
            "termIII":  r"(d) Term III",
            "residual": r"(e) Residual  (diabatic closure)",
            "lh_lwa":   r"(f) LH source",
            "rad_lwa":  r"(g) Radiation source",
            "ana_lwa":  r"(h) Analysis increment",
            "tot_lwa":  r"(i) Total diabatic source",
            "lwa_raw":  r"(j) Raw LWA anomaly",
            "nonqg_lwa": r"(a) Non-QG source (ERA5)",
        },
    },
}

HOVMOLLER_GAUSSIAN_SIGMA = (1.0, 2.5)

BUDGET_LEVELS = np.linspace(-4.0, 4.0, 9, dtype=float)

SOURCE_TERM_LEVELS = np.array(
    [-6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6], dtype=float)

FIG4_MERRA_PANEL_ORDER = ("residual", "lh_lwa", "rad_lwa", "ana_lwa")

FIG4_ALT_PANEL_ORDER = ("nonqg_lwa", "lh_lwa", "rad_lwa", "ana_lwa")
FIG4_MERRA_POSITIONS = [(0, 0), (0, 1), (1, 0), (1, 1)]
FIG4_ALT_PANEL_TITLE = {
    "nonqg_lwa": r"(a) Non-QG source (ERA5)",
    "lh_lwa":    r"(a) MERRA-2 latent heating",
    "rad_lwa":   r"(a) MERRA-2 radiation",
    "ana_lwa":   r"(a) MERRA-2 analysis increment",
}

PANEL_CBAR = {
    "tendency": r"$\partial A / \partial t$  (m s$^{-1}$ day$^{-1}$)",
    "termI":    r"Term I  (m s$^{-1}$ day$^{-1}$)",
    "termII":   r"Term II  (m s$^{-1}$ day$^{-1}$)",
    "termIII":  r"Term III  (m s$^{-1}$ day$^{-1}$)",
    "residual": r"Residual  (m s$^{-1}$ day$^{-1}$)",
    "lh_lwa":   r"LH-LWA source  (m s$^{-1}$ day$^{-1}$)",
    "rad_lwa":  r"RAD-LWA source  (m s$^{-1}$ day$^{-1}$)",
    "ana_lwa":  r"ANA-LWA source  (m s$^{-1}$ day$^{-1}$)",
    "tot_lwa":  r"TOT-LWA source  (m s$^{-1}$ day$^{-1}$)",
    "nonqg_lwa": r"Non-QG source  (m s$^{-1}$ day$^{-1}$)",
    "lwa_raw":  r"Raw LWA anomaly  (m s$^{-1}$)",
}

BUDGET_STRIPS_FILENAME = "budget_strips_{source}_{year}_{month:02d}.nc"
BUDGET_CLIMATOLOGY_FILENAME = "lwa_budget_climatology_{source}.nc"

LAYOUTS = {
    "era5":   (4, 2, (12.5, 17.0)),
    "merra2": (6, 2, (12.5, 24.0)),
}

_STRIP_TIMES: np.ndarray | None = None
_STRIP_DATA: dict | None = None
_STRIP_INDEX: dict | None = None
_STRIP_TERMS: tuple | None = None
_STRIP_CACHE_KEY: tuple | None = None


def _default_mc_workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 8)


def load_all_strips(*, source: str, strip_dir: Path,
                    year_start: int = 2000, year_end: int = 2022) -> None:
    global _STRIP_TIMES, _STRIP_DATA, _STRIP_INDEX, _STRIP_TERMS
    global _STRIP_CACHE_KEY
    root = Path(strip_dir)
    key = (source, str(root.resolve()), year_start, year_end)
    if _STRIP_TERMS is not None and _STRIP_CACHE_KEY == key:
        return

    terms = (("lwa_raw", "tendency", "termI", "termII", "termIII", "residual")
             + SOURCE_CONFIG[source]["extra_terms"])
    times_list = []
    per_term: dict[str, list[np.ndarray]] = {t: [] for t in terms}
    n_files = 0
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            path = root / BUDGET_STRIPS_FILENAME.format(
                source=source, year=year, month=month)
            if not path.exists():
                continue
            n_files += 1
            with nc.Dataset(path, "r") as d:
                ts_any: Any = nc.num2date(
                    d["time"][:], d["time"].units,
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True)
                ts = np.atleast_1d(ts_any)
                times_list.append(np.array([np.datetime64(t, "s") for t in ts]))
                for t in terms:
                    if t in d.variables:
                        per_term[t].append(np.asarray(d[t][:], dtype=np.float32))
                    else:
                        per_term[t].append(np.full_like(per_term[t][-1]
                                                        if per_term[t]
                                                        else np.zeros((ts.size, NLON),
                                                                      dtype=np.float32),
                                                        np.nan))

    if not times_list:
        raise RuntimeError(
            f"No budget strip files found for source={source} under {root}")

    _STRIP_TERMS = (source, terms)
    _STRIP_CACHE_KEY = key
    _STRIP_TIMES = np.concatenate(times_list)
    _STRIP_DATA = {t: np.concatenate(v, axis=0) for t, v in per_term.items()}
    order = np.argsort(_STRIP_TIMES)
    _STRIP_TIMES = _STRIP_TIMES[order]
    _STRIP_DATA = {t: v[order] for t, v in _STRIP_DATA.items()}
    _STRIP_INDEX = {t: i for i, t in enumerate(_STRIP_TIMES)}

    mem_gb = sum(v.nbytes for v in _STRIP_DATA.values()) / 1e9
    _LOG.info("[%s] loaded %d strips from %d monthly files, covering %s .. %s"
              "  (mem ~%.2f GB)",
              source, len(_STRIP_TIMES), n_files,
              _STRIP_TIMES[0], _STRIP_TIMES[-1], mem_gb)


def _strip_at(*, target_dt: datetime.datetime) -> dict[str, np.ndarray] | None:
    if (_STRIP_TERMS is None or _STRIP_TIMES is None or _STRIP_DATA is None
            or _STRIP_INDEX is None):
        return None
    terms = _STRIP_TERMS[1]
    t64 = np.datetime64(target_dt.replace(tzinfo=None), "s")
    j = _STRIP_INDEX.get(t64)
    if j is not None:
        return {t: _STRIP_DATA[t][j] for t in terms}
    idx = int(np.searchsorted(_STRIP_TIMES, t64))
    best = None
    for cand in (idx - 1, idx):
        if 0 <= cand < len(_STRIP_TIMES):
            dt = abs((_STRIP_TIMES[cand] - t64).astype("timedelta64[s]").astype(int))
            if dt <= 3 * 3600 and (best is None or dt < best[0]):
                best = (dt, cand)
    if best is None:
        return None
    j = best[1]
    return {t: _STRIP_DATA[t][j] for t in terms}


def _round_to_6h(*, dt: datetime.datetime) -> datetime.datetime:
    hr = int(round(dt.hour / 6.0)) * 6
    if hr == 24:
        dt = dt + datetime.timedelta(days=1)
        hr = 0
    return dt.replace(hour=hr, minute=0, second=0, microsecond=0)


def build_composite(*, storms_df: pd.DataFrame, reference: str = "recurvature",
                    basin: str | None = None,
                    return_per_storm: bool = False) -> dict[str, Any]:
    assert _STRIP_TERMS is not None
    terms = _STRIP_TERMS[1]
    if basin is not None:
        b = str(basin).upper()
        if b == "WPNA":
            storms_df = storms_df[storms_df["basin"].isin(("WP", "NA"))].copy()
        else:
            storms_df = storms_df[storms_df["basin"] == basin].copy()
    ref_col = "recurv_time" if reference == "recurvature" else "et_time"
    sums = {t: np.zeros((N_LAGS, NLON), dtype=np.float64) for t in terms}
    n_cases = np.zeros((N_LAGS, NLON), dtype=np.int32)
    ref_times = []
    per_storm: dict[str, list[np.ndarray]] | None = (
        {t: [] for t in terms} if return_per_storm else None)
    per_storm_n: list[np.ndarray] | None = ([] if return_per_storm else None)
    for _, st in storms_df.iterrows():
        ref_time = pd.to_datetime(st[ref_col])
        if pd.isna(ref_time):
            continue
        ref_dt = _round_to_6h(dt=ref_time.to_pydatetime())
        ref_times.append(ref_dt)
        if per_storm is not None:
            psm = {t: np.full((N_LAGS, NLON), np.nan, dtype=np.float32)
                   for t in terms}
            psn = np.zeros((N_LAGS, NLON), dtype=np.int32)
        for li, lag_h in enumerate(LAGS):
            target = ref_dt + datetime.timedelta(hours=int(lag_h))
            strips = _strip_at(target_dt=target)
            if strips is None:
                continue
            for t in terms:
                vals = np.nan_to_num(strips[t])
                sums[t][li] += vals
                if per_storm is not None:
                    psm[t][li] = strips[t]
            n_cases[li] += 1
            if per_storm is not None:
                psn[li] = 1
        if per_storm is not None and per_storm_n is not None:
            for t in terms:
                per_storm[t].append(psm[t])
            per_storm_n.append(psn)
    means = {t: (sums[t] / np.maximum(n_cases, 1)).astype(np.float32)
             for t in terms}
    out: dict[str, Any] = dict(means=means, n_cases=n_cases, ref_times=ref_times)
    if per_storm is not None and per_storm_n is not None:
        out["per_storm"] = {t: (np.stack(v, axis=0)
                                if v else np.zeros((0, N_LAGS, NLON), dtype=np.float32))
                            for t, v in per_storm.items()}
        out["per_storm_n"] = (np.stack(per_storm_n, axis=0)
                              if per_storm_n
                              else np.zeros((0, N_LAGS, NLON), dtype=np.int32))
    return out


def bootstrap_diff_sig(*, strips_a: np.ndarray, strips_b: np.ndarray,
                       n_boot: int = 300,
                       smooth_sigma: tuple[float, float] | None = None,
                       seed: int = 12345,
                       alpha: float = 0.05,
                       clim: np.ndarray | None = None) -> np.ndarray:
    a = np.asarray(strips_a, dtype=np.float64)
    b = np.asarray(strips_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((N_LAGS, NLON), dtype=bool)
    if clim is not None:
        a = a - clim[None, None, :]
        b = b - clim[None, None, :]
    n_a, n_b = a.shape[0], b.shape[0]
    pooled = np.concatenate([a, b], axis=0)
    n_pool = pooled.shape[0]

    def _smooth(x: np.ndarray) -> np.ndarray:
        if smooth_sigma is None:
            return x
        return scipy.ndimage.gaussian_filter(x, sigma=smooth_sigma)

    obs = _smooth(np.nanmean(a, axis=0) - np.nanmean(b, axis=0))
    rng = np.random.default_rng(seed)
    extreme_count = np.zeros_like(obs, dtype=np.int64)
    obs_abs = np.abs(obs)
    for _ in range(n_boot):
        idx = rng.permutation(n_pool)
        ia = idx[:n_a]
        ib = idx[n_a:n_a + n_b]
        diff = _smooth(np.nanmean(pooled[ia], axis=0)
                       - np.nanmean(pooled[ib], axis=0))
        extreme_count += (np.abs(diff) >= obs_abs).astype(np.int64)
    p_value = (extreme_count + 1.0) / (n_boot + 1.0)
    return p_value <= alpha


def _mc_one(args: tuple) -> dict[str, np.ndarray]:
    seed, ref_times_utc, year_start, year_end = args
    assert _STRIP_TERMS is not None
    terms = _STRIP_TERMS[1]
    rng = np.random.default_rng(seed)
    sums = {t: np.zeros((N_LAGS, NLON), dtype=np.float64) for t in terms}
    n_cases = np.zeros((N_LAGS, NLON), dtype=np.int32)
    for ref in ref_times_utc:
        year_r = int(rng.integers(year_start, year_end + 1))
        doy_shift = int(rng.integers(-7, 8))
        try:
            new_ref = ref.replace(year=year_r) + datetime.timedelta(days=doy_shift)
        except ValueError:
            new_ref = (datetime.datetime(year_r, ref.month, min(ref.day, 28))
                       + datetime.timedelta(days=doy_shift))
        new_ref = _round_to_6h(dt=new_ref)
        for li, lag_h in enumerate(LAGS):
            target = new_ref + datetime.timedelta(hours=int(lag_h))
            strips = _strip_at(target_dt=target)
            if strips is None:
                continue
            for t in terms:
                sums[t][li] += np.nan_to_num(strips[t])
            n_cases[li] += 1
    return {t: (sums[t] / np.maximum(n_cases, 1)).astype(np.float32)
            for t in terms}


def monte_carlo_sig(*, ref_times: Sequence[datetime.datetime],
                    n_iter: int = 300, year_start: int = 2000,
                    year_end: int = 2022,
                    n_workers: int | None = None,
                    seed_base: int = 12345
                    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if n_workers is None:
        n_workers = _default_mc_workers()
    assert _STRIP_TERMS is not None
    terms = _STRIP_TERMS[1]
    args = [(seed_base + i, ref_times, year_start, year_end)
            for i in range(n_iter)]
    draws = {t: np.zeros((n_iter, N_LAGS, NLON), dtype=np.float32)
             for t in terms}
    _LOG.info("Running %d MC draws with %d workers...", n_iter, n_workers)
    ctx = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx) as ex:
        futs = {ex.submit(_mc_one, a): i for i, a in enumerate(args)}
        for done, fut in enumerate(concurrent.futures.as_completed(futs)):
            i = futs[fut]
            res = fut.result()
            for t in terms:
                draws[t][i] = res[t]
            if (done + 1) % 50 == 0 or done == n_iter - 1:
                _LOG.info("  %d/%d MC draws done", done + 1, n_iter)
    lo = {t: np.percentile(draws[t], 2.5, axis=0) for t in terms}
    hi = {t: np.percentile(draws[t], 97.5, axis=0) for t in terms}
    return lo, hi


def load_clim(*, source: str, season: str, strip_dir: Path
              ) -> dict[str, np.ndarray]:
    path = Path(strip_dir) / BUDGET_CLIMATOLOGY_FILENAME.format(source=source)
    with nc.Dataset(path, "r") as d:
        grp = d[season]
        return {t: np.asarray(grp[t][:], dtype=np.float32)
                for t in grp.variables}


def _make_fig(*, source: str, n_panels: int
              ) -> tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]:
    n_rows, n_cols, figsize = LAYOUTS[source]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        n_rows, n_cols,
        height_ratios=[0.55] + [4.3] * (n_rows - 1),
        hspace=0.28, wspace=0.22,
        left=0.06, right=0.985,
        top=0.965, bottom=0.035,
    )
    ax_map_a = fig.add_subplot(gs[0, 0])
    ax_map_b = fig.add_subplot(gs[0, 1])
    qj3._draw_minimap(ax=ax_map_a)
    qj3._draw_minimap(ax=ax_map_b)
    axes = []
    for r in range(1, n_rows):
        for c in range(n_cols):
            axes.append(fig.add_subplot(gs[r, c]))
    return fig, axes[:n_panels]


def plot_budget(*, storms_df: pd.DataFrame, basin: str,
                strip_dir: Path,
                out_dir: Path,
                source: str = "era5",
                reference: str = "recurvature",
                n_mc: int = 300, n_workers: int | None = None,
                smooth_sigma: tuple[float, float] = HOVMOLLER_GAUSSIAN_SIGMA,
                year_start: int = 2000, year_end: int = 2022,
                tracks_dir: Path | None = None) -> None:
    cfg = SOURCE_CONFIG[source]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    season = BASIN_SEASON.get(basin, "JJASON")

    load_all_strips(source=source, strip_dir=strip_dir,
                    year_start=year_start, year_end=year_end)
    assert _STRIP_TERMS is not None
    terms = _STRIP_TERMS[1]

    _LOG.info("[%s] [%s, %s, %s] building composite...",
              source, basin, reference, season)
    comp = build_composite(storms_df=storms_df, reference=reference, basin=basin)
    if len(comp["ref_times"]) == 0:
        _LOG.info("  no storms, aborting")
        return

    clim = load_clim(source=source, season=season, strip_dir=strip_dir)
    anom = {t: comp["means"][t] - clim[t][None, :] for t in terms}

    lo, hi = monte_carlo_sig(ref_times=comp["ref_times"], n_iter=n_mc,
                             year_start=year_start, year_end=year_end,
                             n_workers=n_workers)
    lo_anom = {t: lo[t] - clim[t][None, :] for t in terms}
    hi_anom = {t: hi[t] - clim[t][None, :] for t in terms}

    smooth = {t: scipy.ndimage.gaussian_filter(anom[t], sigma=smooth_sigma)
              for t in terms}

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
        tracks_directory=tracks_dir)
    track_kw: dict[str, Any] = dict(lon_range=lon_range,
                                    mean_track_lag=mean_track_lag,
                                    mean_track_lon=mean_track_lon,
                                    recurv_lon_mean=recurv_lon_mean)

    budget_levels = BUDGET_LEVELS
    source_levels = SOURCE_TERM_LEVELS
    raw_levels = np.array([-12, -9, -6, -3, 0, 3, 6, 9, 12], dtype=float)
    levels_for = {}
    for t in terms:
        if t == "lwa_raw":
            levels_for[t] = raw_levels
        elif t in ("tendency", "termI", "termII", "termIII", "residual"):
            levels_for[t] = budget_levels
        else:
            levels_for[t] = source_levels

    fig, axes = _make_fig(source=source, n_panels=len(cfg["panel_order"]))
    for ax, t in zip(axes, cfg["panel_order"]):
        qj3._hovmoller_panel(
            ax=ax, data=smooth[t],
            mask_lo=lo_anom[t], mask_hi=hi_anom[t],
            levels=levels_for[t], cmap=qj3._BWOR_8,
            title=cfg["panel_title"][t],
            cbar_label=PANEL_CBAR[t],
            sig_field=anom[t],
            **track_kw,
        )

    ref_label = {"recurvature": "Recurvature-relative",
                 "et": "ET-relative"}.get(reference, reference.title())
    src_label = "ERA5" if source == "era5" else "MERRA-2"
    fig.suptitle(
        f"{ref_label} LWA-budget composite  \N{EM DASH}  {src_label}  \N{EM DASH}  "
        f"{basin} recurving TCs ({season} climatology, "
        f"N={len(comp['ref_times'])})",
        fontsize=12, y=0.985,
    )
    out_path = out_dir / f"fig3_lwa_budget_{source}_{basin}_{reference}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("  wrote %s", out_path)


def _title_reletter(*, letter: str, template_title: str) -> str:
    body = (template_title.split(")", 1)[-1].strip()
            if ")" in template_title else template_title)
    return f"({letter}) {body}"


def _add_wp_na_column_separator(*, fig: matplotlib.figure.Figure,
                                data_axes: np.ndarray,
                                ax_maps: list[matplotlib.axes.Axes],
                                n_data_rows: int = 3) -> None:
    p1 = data_axes[0, 1].get_position()
    p2 = data_axes[0, 2].get_position()
    gap = p2.x0 - p1.x1
    if gap > 1e-4:
        x = p1.x1 + 0.07 * gap
    else:
        x = 0.5 * (p1.x1 + p2.x0)
    y0 = min(data_axes[r, c].get_position().y0
             for r in range(n_data_rows) for c in range(4))
    y1 = max(ax_maps[j].get_position().y1 for j in range(4))
    fig.add_artist(
        matplotlib.lines.Line2D(
            [x, x], [y0, y1], transform=fig.transFigure,
            color="black", linewidth=1.8, zorder=200, clip_on=False))


def _add_wp_na_group_titles(*, fig: matplotlib.figure.Figure,
                            ax_maps: list[matplotlib.axes.Axes],
                            fontsize: int = 19) -> None:
    if len(ax_maps) < 4:
        return
    fig.canvas.draw()
    p0, p1, p2, p3 = [ax.get_position() for ax in ax_maps[:4]]
    y = min(0.985, max(p.y1 for p in (p0, p1, p2, p3)) + 0.020)
    fig.text(0.5 * (p0.x0 + p1.x1), y, "WP", fontsize=fontsize,
             fontweight="bold", ha="center", va="top",
             transform=fig.transFigure)
    fig.text(0.5 * (p2.x0 + p3.x1), y, "NA", fontsize=fontsize,
             fontweight="bold", ha="center", va="top",
             transform=fig.transFigure)


def wb_strat_storm_ids(*, classification_csv: Path,
                       reference: str,
                       wb_group: str) -> set[str]:
    df = pd.read_csv(classification_csv, keep_default_na=False, na_values=[""])
    sub = df[(df["reference"] == reference) & (df["wb_group"] == wb_group)]
    return set(sub["storm_id"].astype(str))


def _load_full_tracks(*, tracks_dir: Path | None
                      ) -> dict[str, pd.DataFrame] | None:
    if tracks_dir is None:
        _LOG.info("  [tracks] mean-track overlay disabled (no tracks_dir)")
        return None
    try:
        _, full_tracks = tracks.load_track_database(
            tracks_directory=Path(tracks_dir))
    except Exception:
        _LOG.exception("  [tracks] mean-track overlay disabled")
        return None
    return full_tracks


def _build_wp_na_era5_budget_results(
        *,
        storms_df: pd.DataFrame,
        strip_dir: Path,
        reference: str,
        basins: tuple[str, ...] = ("WP", "NA"),
        n_mc: int,
        n_workers: int | None,
        smooth_sigma: tuple[float, float] | None,
        year_start: int,
        year_end: int,
        tracks_dir: Path | None,
) -> tuple[dict[str, Any], tuple[str, ...], dict, dict | None,
           Path | None, np.ndarray, tuple[float, float]]:
    source = "era5"
    cfg = SOURCE_CONFIG[source]
    panel_keys = tuple(k for k in cfg["panel_order"] if k != "nonqg_lwa")
    if len(panel_keys) != 6:
        raise ValueError("Expected six ERA5 panels for WP|NA combined figure.")

    sig = smooth_sigma if smooth_sigma is not None else HOVMOLLER_GAUSSIAN_SIGMA

    load_all_strips(source=source, strip_dir=strip_dir,
                    year_start=year_start, year_end=year_end)
    assert _STRIP_TERMS is not None
    terms = _STRIP_TERMS[1]

    full_tracks = _load_full_tracks(tracks_dir=tracks_dir)

    budget_levels = BUDGET_LEVELS

    results: dict = {}
    for basin in basins:
        season = BASIN_SEASON.get(basin, "JJASON")
        _LOG.info("[%s] [%s, %s, %s] building composite...",
                  source, basin, reference, season)
        comp = build_composite(storms_df=storms_df, reference=reference,
                               basin=basin)
        if len(comp["ref_times"]) == 0:
            raise RuntimeError(f"No storms for basin={basin}, cannot build "
                               "budget figure.")
        clim = load_clim(source=source, season=season, strip_dir=strip_dir)
        anom = {t: comp["means"][t] - clim[t][None, :] for t in terms}
        lo, hi = monte_carlo_sig(ref_times=comp["ref_times"], n_iter=n_mc,
                                 year_start=year_start, year_end=year_end,
                                 n_workers=n_workers)
        lo_anom = {t: lo[t] - clim[t][None, :] for t in terms}
        hi_anom = {t: hi[t] - clim[t][None, :] for t in terms}
        smooth = {t: scipy.ndimage.gaussian_filter(anom[t], sigma=sig)
                  for t in terms}

        bkey = str(basin).upper()
        if bkey == "WPNA":
            st = storms_df.copy()
        else:
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
            full_tracks=full_tracks)
        track_kw = dict(lon_range=lon_range,
                        mean_track_lag=mean_track_lag,
                        mean_track_lon=mean_track_lon,
                        recurv_lon_mean=recurv_lon_mean)

        results[basin] = dict(
            smooth=smooth, anom=anom, lo_anom=lo_anom, hi_anom=hi_anom,
            track_kw=track_kw, n_storms=len(comp["ref_times"]),
        )
    return cfg, panel_keys, results, full_tracks, tracks_dir, budget_levels, sig


def _render_wp_na_budget_block(
        *,
        fig: matplotlib.figure.Figure,
        outer: matplotlib.gridspec.GridSpec,
        results: dict,
        cfg: dict[str, Any],
        panel_keys: tuple[str, ...],
        budget_levels: np.ndarray,
        letter_index0: int,
        sig_mask_per_basin: dict | None = None,
        basins: tuple[str, ...] = ("WP", "NA"),
        block_titles: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[matplotlib.axes.Axes]]:
    gs_maps = outer[0, 0].subgridspec(1, 4, wspace=0.20)
    gs_data = outer[1, 0].subgridspec(3, 4, hspace=0.30, wspace=0.20)

    ax_maps = []
    for j in range(4):
        ax = fig.add_subplot(gs_maps[0, j])
        ax_maps.append(ax)
        qj3._draw_minimap(ax=ax)

    data_ax = np.empty((3, 4), dtype=object)
    for r in range(3):
        data_ax[r, 0] = fig.add_subplot(gs_data[r, 0])
        data_ax[r, 1] = fig.add_subplot(gs_data[r, 1], sharey=data_ax[r, 0])
        data_ax[r, 2] = fig.add_subplot(gs_data[r, 2])
        data_ax[r, 3] = fig.add_subplot(gs_data[r, 3], sharey=data_ax[r, 2])
        for c in (1, 3):
            data_ax[r, c].tick_params(labelleft=False)

    letters = "abcdefghijklmnopqrstuvwxyz"
    li = letter_index0
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    if len(basins) != 2:
        raise ValueError("_render_wp_na_budget_block expects exactly 2 basins")

    if block_titles is not None and len(block_titles) == 2:
        for col0_pos, txt in ((0.27, block_titles[0]), (0.73, block_titles[1])):
            fig.text(col0_pos, 0.0, "", transform=fig.transFigure)

    for basin, col0 in zip(basins, (0, 2)):
        st = results[basin]
        track_kw = st["track_kw"]
        sig_basin = ((sig_mask_per_basin or {}).get(basin, {})
                     if sig_mask_per_basin else {})
        for (pr, pc), key in zip(positions, panel_keys):
            ax = data_ax[pr, pc + col0]
            abs_col = pc + col0
            title = _title_reletter(letter=letters[li],
                                    template_title=cfg["panel_title"][key])
            li += 1
            qj3._hovmoller_panel(
                ax=ax, data=st["smooth"][key],
                mask_lo=st["lo_anom"][key], mask_hi=st["hi_anom"][key],
                levels=budget_levels, cmap=qj3._BWOR_8,
                title=title,
                cbar_label=PANEL_CBAR[key],
                with_colorbar=False,
                title_fontsize=16,
                tick_labelsize=13,
                title_pad=9,
                show_xlabel=(pr == 2),
                show_ylabel=(abs_col == 0),
                sig_field=st["anom"][key],
                sig_mask=sig_basin.get(key) if sig_basin else None,
                **track_kw,
            )

    fig.canvas.draw()
    br = [data_ax[2, c] for c in range(4)]
    y0 = min(ax.get_position().y0 for ax in br)
    h = max(ax.get_position().height for ax in br)
    for c, ax in enumerate(br):
        x0, _, w, _ = ax.get_position().bounds
        ax.set_position([x0, y0, w, h])
    fig.align_xlabels([data_ax[2, c] for c in range(4)])
    return data_ax, ax_maps


def plot_budget_wp_na_combined(
        *,
        storms_df: pd.DataFrame,
        strip_dir: Path,
        out_path: Path,
        source: str = "era5",
        reference: str = "recurvature",
        n_mc: int = 300,
        n_workers: int | None = None,
        smooth_sigma: tuple[float, float] | None = None,
        year_start: int = 2000,
        year_end: int = 2022,
        tracks_dir: Path | None = None,
        storm_id_filter: set[str] | frozenset[str] | None = None) -> Path:
    if source != "era5":
        raise ValueError("WP|NA combined layout is only wired for ERA5 "
                         "(six budget panels).")
    if storm_id_filter is not None:
        storms_df = storms_df[storms_df["storm_id"].isin(storm_id_filter)].copy()

    (cfg, panel_keys, results, _ft, _td, budget_levels, _sig
     ) = _build_wp_na_era5_budget_results(
        storms_df=storms_df,
        strip_dir=strip_dir,
        reference=reference,
        n_mc=n_mc,
        n_workers=n_workers,
        smooth_sigma=smooth_sigma,
        year_start=year_start,
        year_end=year_end,
        tracks_dir=tracks_dir,
    )

    fig = plt.figure(figsize=(23.0, 12.4))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[0.55, 5.55],
        hspace=0.14,
        left=0.055, right=0.94, top=0.95, bottom=0.08,
    )
    data_ax, ax_maps = _render_wp_na_budget_block(
        fig=fig, outer=outer, results=results, cfg=cfg, panel_keys=panel_keys,
        budget_levels=budget_levels, letter_index0=0)

    norm_b = matplotlib.colors.BoundaryNorm(budget_levels, ncolors=qj3._BWOR_8.N)
    sm_b = matplotlib.cm.ScalarMappable(norm=norm_b, cmap=qj3._BWOR_8)
    sm_b.set_array(np.array([0.0]))

    cax = fig.add_axes((0.955, 0.20, 0.015, 0.60))
    cb = fig.colorbar(
        sm_b, cax=cax, orientation="vertical",
        ticks=budget_levels, extend="both",
    )
    cb.set_label(r"Anomaly (m s$^{-1}$ day$^{-1}$)", fontsize=16, labelpad=8)
    cb.ax.tick_params(labelsize=14)

    fig.canvas.draw()
    _add_wp_na_column_separator(fig=fig, data_axes=data_ax, ax_maps=ax_maps,
                                n_data_rows=3)
    _add_wp_na_group_titles(fig=fig, ax_maps=ax_maps, fontsize=19)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    _LOG.info("  wrote %s", out)
    return out


def plot_budget_wp_na_wb_strat_high_low(
        *,
        storms_df: pd.DataFrame,
        strip_dir: Path,
        classification_csv: Path,
        out_path: Path,
        source: str = "era5",
        reference: str = "recurvature",
        n_mc: int = 300,
        n_workers: int | None = None,
        smooth_sigma: tuple[float, float] | None = None,
        year_start: int = 2000,
        year_end: int = 2022,
        tracks_dir: Path | None = None,
        n_bootstrap_diff: int = 0,
        group_high: str = "highwb",
        group_low: str = "lowwb",
        title_high: str = ("High downstream RWB (upper quintile) "
                           "\N{EM DASH} WP+NA pooled"),
        title_low: str = ("Low downstream RWB (lower quintile) "
                          "\N{EM DASH} WP+NA pooled"),
        footer_label_high: str = "High downstream RWB",
        footer_label_low: str = "Low downstream RWB",
) -> Path:
    if source != "era5":
        raise ValueError("Stratified layout is only wired for ERA5.")

    ids_hi = wb_strat_storm_ids(classification_csv=classification_csv,
                                reference=reference, wb_group=group_high)
    ids_lo = wb_strat_storm_ids(classification_csv=classification_csv,
                                reference=reference, wb_group=group_low)
    df_hi = storms_df[storms_df["storm_id"].isin(ids_hi)].copy()
    df_lo = storms_df[storms_df["storm_id"].isin(ids_lo)].copy()

    _LOG.info("[Fig 8] building pooled WP+NA composites for high RWB...")
    (cfg, panel_keys, res_hi_raw, *_hi
     ) = _build_wp_na_era5_budget_results(
        storms_df=df_hi, strip_dir=strip_dir, reference=reference,
        basins=("WPNA",), n_mc=n_mc,
        n_workers=n_workers, smooth_sigma=smooth_sigma,
        year_start=year_start, year_end=year_end, tracks_dir=tracks_dir)
    _LOG.info("[Fig 8] building pooled WP+NA composites for low RWB...")
    (_, panel_keys2, res_lo_raw, *_lo
     ) = _build_wp_na_era5_budget_results(
        storms_df=df_lo, strip_dir=strip_dir, reference=reference,
        basins=("WPNA",), n_mc=n_mc,
        n_workers=n_workers, smooth_sigma=smooth_sigma,
        year_start=year_start, year_end=year_end, tracks_dir=tracks_dir)
    if panel_keys != panel_keys2:
        raise AssertionError("panel_key mismatch")

    results = {"HIGH": res_hi_raw["WPNA"], "LOW": res_lo_raw["WPNA"]}

    sig_diff: dict | None = None
    if n_bootstrap_diff and int(n_bootstrap_diff) > 0:
        _LOG.info("  Building per-storm strips for bootstrap-of-differences "
                  "(n_boot=%d)...", int(n_bootstrap_diff))
        sig = smooth_sigma if smooth_sigma is not None else HOVMOLLER_GAUSSIAN_SIGMA
        comp_hi = build_composite(storms_df=df_hi, reference=reference,
                                  basin="WPNA", return_per_storm=True)
        comp_lo = build_composite(storms_df=df_lo, reference=reference,
                                  basin="WPNA", return_per_storm=True)
        clim = load_clim(source=source, season=BASIN_SEASON.get("WP", "JJASON"),
                         strip_dir=strip_dir)
        sig_diff = {"HIGH": {}, "LOW": {}}
        for key in panel_keys:
            a = comp_hi["per_storm"].get(key)
            b = comp_lo["per_storm"].get(key)
            if a is None or b is None or a.size == 0 or b.size == 0:
                continue
            mask = bootstrap_diff_sig(
                strips_a=a, strips_b=b,
                n_boot=int(n_bootstrap_diff),
                smooth_sigma=sig,
                clim=clim.get(key),
                seed=12345,
            )
            sig_diff["HIGH"][key] = mask
            sig_diff["LOW"][key] = mask

    budget_levels = BUDGET_LEVELS

    fig = plt.figure(figsize=(23.0, 12.4))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[0.55, 5.55],
        hspace=0.14,
        left=0.055, right=0.94, top=0.93, bottom=0.08,
    )

    fig.text(
        0.27, 0.965, title_high,
        fontsize=12, fontweight="semibold", ha="center",
        transform=fig.transFigure)
    fig.text(
        0.73, 0.965, title_low,
        fontsize=12, fontweight="semibold", ha="center",
        transform=fig.transFigure)

    data_ax, ax_maps = _render_wp_na_budget_block(
        fig=fig, outer=outer, results=results, cfg=cfg, panel_keys=panel_keys,
        budget_levels=budget_levels, letter_index0=0,
        sig_mask_per_basin=sig_diff,
        basins=("HIGH", "LOW"))

    norm_b = matplotlib.colors.BoundaryNorm(budget_levels, ncolors=qj3._BWOR_8.N)
    sm_b = matplotlib.cm.ScalarMappable(norm=norm_b, cmap=qj3._BWOR_8)
    sm_b.set_array(np.array([0.0]))
    cax = fig.add_axes((0.955, 0.20, 0.015, 0.60))
    cb = fig.colorbar(
        sm_b, cax=cax, orientation="vertical",
        ticks=budget_levels, extend="both",
    )
    cb.set_label(r"Anomaly (m s$^{-1}$ day$^{-1}$)", fontsize=16, labelpad=8)
    cb.ax.tick_params(labelsize=14)

    fig.canvas.draw()
    _add_wp_na_column_separator(fig=fig, data_axes=data_ax, ax_maps=ax_maps,
                                n_data_rows=3)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    _LOG.info("  wrote %s", out)
    return out


def plot_merra2_fig4_wp_na_combined(
        *,
        storms_df: pd.DataFrame,
        strip_dir: Path,
        out_path: Path,
        reference: str = "recurvature",
        n_mc: int = 300,
        n_workers: int | None = None,
        smooth_sigma: tuple[float, float] | None = None,
        year_start: int = 2000,
        year_end: int = 2022,
        tracks_dir: Path | None = None) -> Path:
    source = "merra2"
    cfg = SOURCE_CONFIG[source]
    panel_keys = FIG4_MERRA_PANEL_ORDER
    positions = FIG4_MERRA_POSITIONS
    for k in panel_keys:
        if k not in cfg["panel_title"]:
            raise KeyError(f"Missing panel_title for {k!r}")

    sig = smooth_sigma if smooth_sigma is not None else HOVMOLLER_GAUSSIAN_SIGMA
    n_data_rows = 2
    _fig4_bottom_margin = 0.18

    load_all_strips(source=source, strip_dir=strip_dir,
                    year_start=year_start, year_end=year_end)
    assert _STRIP_TERMS is not None
    terms = _STRIP_TERMS[1]
    for k in panel_keys:
        if k not in terms:
            raise RuntimeError(f"MERRA-2 strips missing term {k!r}")

    full_tracks = _load_full_tracks(tracks_dir=tracks_dir)

    results: dict = {}
    for basin in ("WP", "NA"):
        season = BASIN_SEASON.get(basin, "JJASON")
        _LOG.info("[%s] [%s, %s, %s] (Fig. 4 WP|NA) building composite...",
                  source, basin, reference, season)
        comp = build_composite(storms_df=storms_df, reference=reference,
                               basin=basin)
        if len(comp["ref_times"]) == 0:
            raise RuntimeError(f"No storms for basin={basin}, cannot build "
                               "Fig. 4.")
        clim = load_clim(source=source, season=season, strip_dir=strip_dir)
        anom = {t: comp["means"][t] - clim[t][None, :] for t in terms}
        lo, hi = monte_carlo_sig(ref_times=comp["ref_times"], n_iter=n_mc,
                                 year_start=year_start, year_end=year_end,
                                 n_workers=n_workers)
        lo_anom = {t: lo[t] - clim[t][None, :] for t in terms}
        hi_anom = {t: hi[t] - clim[t][None, :] for t in terms}
        smooth = {t: scipy.ndimage.gaussian_filter(anom[t], sigma=sig)
                  for t in terms}

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
            full_tracks=full_tracks)
        track_kw = dict(lon_range=lon_range,
                        mean_track_lag=mean_track_lag,
                        mean_track_lon=mean_track_lon,
                        recurv_lon_mean=recurv_lon_mean)

        results[basin] = dict(
            smooth=smooth, anom=anom, lo_anom=lo_anom, hi_anom=hi_anom,
            track_kw=track_kw, n_storms=len(comp["ref_times"]),
        )

    fig = plt.figure(figsize=(23.0, 10.1))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[0.55, 3.95],
        hspace=0.14,
        left=0.055, right=0.94, top=0.95, bottom=0.08,
    )
    gs_maps = outer[0, 0].subgridspec(1, 4, wspace=0.20)
    gs_data = outer[1, 0].subgridspec(n_data_rows, 4, hspace=0.30, wspace=0.20)

    ax_maps = []
    for j in range(4):
        ax = fig.add_subplot(gs_maps[0, j])
        ax_maps.append(ax)
        qj3._draw_minimap(ax=ax)

    data_ax = np.empty((n_data_rows, 4), dtype=object)
    for r in range(n_data_rows):
        data_ax[r, 0] = fig.add_subplot(gs_data[r, 0])
        data_ax[r, 1] = fig.add_subplot(gs_data[r, 1], sharey=data_ax[r, 0])
        data_ax[r, 2] = fig.add_subplot(gs_data[r, 2])
        data_ax[r, 3] = fig.add_subplot(gs_data[r, 3], sharey=data_ax[r, 2])
        for c in (1, 3):
            data_ax[r, c].tick_params(labelleft=False)

    letters = "abcdefgh"
    li = 0
    last_row = n_data_rows - 1

    for basin, col0 in (("WP", 0), ("NA", 2)):
        st = results[basin]
        track_kw = st["track_kw"]
        for (pr, pc), key in zip(positions, panel_keys):
            ax = data_ax[pr, pc + col0]
            abs_col = pc + col0
            title = _title_reletter(letter=letters[li],
                                    template_title=cfg["panel_title"][key])
            li += 1
            levels = BUDGET_LEVELS
            qj3._hovmoller_panel(
                ax=ax, data=st["smooth"][key],
                mask_lo=st["lo_anom"][key], mask_hi=st["hi_anom"][key],
                levels=levels, cmap=qj3._BWOR_8,
                title=title,
                cbar_label=PANEL_CBAR[key],
                with_colorbar=False,
                title_fontsize=16,
                tick_labelsize=13,
                title_pad=9,
                show_xlabel=(pr == last_row),
                show_ylabel=(abs_col == 0),
                sig_field=st["anom"][key],
                **track_kw,
            )

    fig.canvas.draw()
    br = [data_ax[last_row, c] for c in range(4)]
    y0 = min(ax.get_position().y0 for ax in br)
    h = max(ax.get_position().height for ax in br)
    for c, ax in enumerate(br):
        x0, _, w, _ = ax.get_position().bounds
        ax.set_position([x0, y0, w, h])
    fig.align_xlabels([data_ax[last_row, c] for c in range(4)])

    norm_b = matplotlib.colors.BoundaryNorm(BUDGET_LEVELS, ncolors=qj3._BWOR_8.N)
    sm_b = matplotlib.cm.ScalarMappable(norm=norm_b, cmap=qj3._BWOR_8)
    sm_b.set_array(np.array([0.0]))

    cax = fig.add_axes((0.955, 0.24, 0.015, 0.52))
    cb = fig.colorbar(
        sm_b, cax=cax, orientation="vertical",
        ticks=BUDGET_LEVELS, extend="both",
    )
    cb.set_label(r"Anomaly (m s$^{-1}$ day$^{-1}$)", fontsize=16, labelpad=8)
    cb.ax.tick_params(labelsize=14)

    fig.canvas.draw()
    _add_wp_na_column_separator(fig=fig, data_axes=data_ax, ax_maps=ax_maps,
                                n_data_rows=n_data_rows)
    _add_wp_na_group_titles(fig=fig, ax_maps=ax_maps, fontsize=19)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    _LOG.info("  wrote %s", out)
    return out


def plot_fig4_alt_nonqg_wp_na(
        *,
        storms_df: pd.DataFrame,
        strip_dir: Path,
        out_path: Path,
        reference: str = "recurvature",
        n_mc: int = 300,
        n_workers: int | None = None,
        smooth_sigma: tuple[float, float] | None = None,
        year_start: int = 2000,
        year_end: int = 2022,
        tracks_dir: Path | None = None) -> Path:
    panel_keys = FIG4_ALT_PANEL_ORDER
    positions = FIG4_MERRA_POSITIONS
    for k in panel_keys:
        if k not in FIG4_ALT_PANEL_TITLE:
            raise KeyError(f"Missing FIG4_ALT_PANEL_TITLE for {k!r}")

    sig = smooth_sigma if smooth_sigma is not None else HOVMOLLER_GAUSSIAN_SIGMA
    n_data_rows = 2

    era5_terms = [k for k in panel_keys if k == "nonqg_lwa"]
    merra2_terms = [k for k in panel_keys if k != "nonqg_lwa"]

    full_tracks = _load_full_tracks(tracks_dir=tracks_dir)

    results: dict = {}
    for basin in ("WP", "NA"):
        season = BASIN_SEASON.get(basin, "JJASON")
        smooth_merged: dict[str, np.ndarray] = {}
        anom_merged: dict[str, np.ndarray] = {}
        lo_anom_merged: dict[str, np.ndarray] = {}
        hi_anom_merged: dict[str, np.ndarray] = {}
        ref_times: list[datetime.datetime] = []

        if era5_terms:
            _LOG.info("[era5] [%s, %s, %s] (Fig. 4 ALT) building composite "
                      "for nonqg_lwa...", basin, reference, season)
            load_all_strips(source="era5", strip_dir=strip_dir,
                            year_start=year_start, year_end=year_end)
            comp_era5 = build_composite(storms_df=storms_df,
                                        reference=reference, basin=basin)
            if len(comp_era5["ref_times"]) == 0:
                raise RuntimeError(
                    f"No storms for basin={basin}, cannot build Fig. 4 ALT.")
            clim_era5 = load_clim(source="era5", season=season,
                                  strip_dir=strip_dir)
            assert _STRIP_TERMS is not None
            terms_era5 = _STRIP_TERMS[1]
            for t in era5_terms:
                if t not in terms_era5:
                    raise RuntimeError(f"ERA5 strips missing term {t!r}")
                anom_merged[t] = comp_era5["means"][t] - clim_era5[t][None, :]
            lo_era5, hi_era5 = monte_carlo_sig(
                ref_times=comp_era5["ref_times"], n_iter=n_mc,
                year_start=year_start, year_end=year_end,
                n_workers=n_workers)
            for t in era5_terms:
                lo_anom_merged[t] = lo_era5[t] - clim_era5[t][None, :]
                hi_anom_merged[t] = hi_era5[t] - clim_era5[t][None, :]
                smooth_merged[t] = scipy.ndimage.gaussian_filter(
                    anom_merged[t], sigma=sig)
            ref_times = comp_era5["ref_times"]

        if merra2_terms:
            _LOG.info("[merra2] [%s, %s, %s] (Fig. 4 ALT) building composite "
                      "for diabatic terms...", basin, reference, season)
            load_all_strips(source="merra2", strip_dir=strip_dir,
                            year_start=year_start, year_end=year_end)
            comp_m2 = build_composite(storms_df=storms_df,
                                      reference=reference, basin=basin)
            if len(comp_m2["ref_times"]) == 0:
                raise RuntimeError(
                    f"No storms for basin={basin}, cannot build Fig. 4 ALT.")
            clim_m2 = load_clim(source="merra2", season=season,
                                strip_dir=strip_dir)
            assert _STRIP_TERMS is not None
            terms_m2 = _STRIP_TERMS[1]
            for t in merra2_terms:
                if t not in terms_m2:
                    raise RuntimeError(f"MERRA-2 strips missing term {t!r}")
                anom_merged[t] = comp_m2["means"][t] - clim_m2[t][None, :]
            lo_m2, hi_m2 = monte_carlo_sig(
                ref_times=comp_m2["ref_times"], n_iter=n_mc,
                year_start=year_start, year_end=year_end,
                n_workers=n_workers)
            for t in merra2_terms:
                lo_anom_merged[t] = lo_m2[t] - clim_m2[t][None, :]
                hi_anom_merged[t] = hi_m2[t] - clim_m2[t][None, :]
                smooth_merged[t] = scipy.ndimage.gaussian_filter(
                    anom_merged[t], sigma=sig)
            if not era5_terms:
                ref_times = comp_m2["ref_times"]

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
            full_tracks=full_tracks)
        track_kw = dict(lon_range=lon_range,
                        mean_track_lag=mean_track_lag,
                        mean_track_lon=mean_track_lon,
                        recurv_lon_mean=recurv_lon_mean)

        results[basin] = dict(
            smooth=smooth_merged, anom=anom_merged,
            lo_anom=lo_anom_merged, hi_anom=hi_anom_merged,
            track_kw=track_kw, n_storms=len(ref_times),
        )

    fig = plt.figure(figsize=(23.0, 10.1))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[0.55, 3.95],
        hspace=0.14,
        left=0.055, right=0.94, top=0.95, bottom=0.08,
    )
    gs_maps = outer[0, 0].subgridspec(1, 4, wspace=0.20)
    gs_data = outer[1, 0].subgridspec(n_data_rows, 4, hspace=0.30, wspace=0.20)

    ax_maps = []
    for j in range(4):
        ax = fig.add_subplot(gs_maps[0, j])
        ax_maps.append(ax)
        qj3._draw_minimap(ax=ax)

    data_ax = np.empty((n_data_rows, 4), dtype=object)
    for r in range(n_data_rows):
        data_ax[r, 0] = fig.add_subplot(gs_data[r, 0])
        data_ax[r, 1] = fig.add_subplot(gs_data[r, 1], sharey=data_ax[r, 0])
        data_ax[r, 2] = fig.add_subplot(gs_data[r, 2])
        data_ax[r, 3] = fig.add_subplot(gs_data[r, 3], sharey=data_ax[r, 2])
        for c in (1, 3):
            data_ax[r, c].tick_params(labelleft=False)

    letters = "abcdefgh"
    li = 0
    last_row = n_data_rows - 1

    for basin, col0 in (("WP", 0), ("NA", 2)):
        st_res = results[basin]
        track_kw = st_res["track_kw"]
        for (pr, pc), key in zip(positions, panel_keys):
            ax = data_ax[pr, pc + col0]
            abs_col = pc + col0
            title = _title_reletter(letter=letters[li],
                                    template_title=FIG4_ALT_PANEL_TITLE[key])
            li += 1
            levels = BUDGET_LEVELS
            qj3._hovmoller_panel(
                ax=ax, data=st_res["smooth"][key],
                mask_lo=st_res["lo_anom"][key], mask_hi=st_res["hi_anom"][key],
                levels=levels, cmap=qj3._BWOR_8,
                title=title,
                cbar_label=PANEL_CBAR[key],
                with_colorbar=False,
                title_fontsize=16,
                tick_labelsize=13,
                title_pad=9,
                show_xlabel=(pr == last_row),
                show_ylabel=(abs_col == 0),
                sig_field=st_res["anom"][key],
                **track_kw,
            )

    fig.canvas.draw()
    br = [data_ax[last_row, c] for c in range(4)]
    y0 = min(ax.get_position().y0 for ax in br)
    h = max(ax.get_position().height for ax in br)
    for c, ax in enumerate(br):
        x0, _, w, _ = ax.get_position().bounds
        ax.set_position([x0, y0, w, h])
    fig.align_xlabels([data_ax[last_row, c] for c in range(4)])

    norm_b = matplotlib.colors.BoundaryNorm(BUDGET_LEVELS, ncolors=qj3._BWOR_8.N)
    sm_b = matplotlib.cm.ScalarMappable(norm=norm_b, cmap=qj3._BWOR_8)
    sm_b.set_array(np.array([0.0]))

    cax = fig.add_axes((0.955, 0.24, 0.015, 0.52))
    cb = fig.colorbar(
        sm_b, cax=cax, orientation="vertical",
        ticks=BUDGET_LEVELS, extend="both",
    )
    cb.set_label(r"Anomaly (m s$^{-1}$ day$^{-1}$)", fontsize=16, labelpad=8)
    cb.ax.tick_params(labelsize=14)

    fig.canvas.draw()
    _add_wp_na_column_separator(fig=fig, data_axes=data_ax, ax_maps=ax_maps,
                                n_data_rows=n_data_rows)
    _add_wp_na_group_titles(fig=fig, ax_maps=ax_maps, fontsize=19)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    _LOG.info("  wrote %s", out)
    return out


def main(
        tracks_csv: Annotated[Path, typer.Option(
            help="recurving_nh_tracks.csv from the track database")],
        strip_directory: Annotated[Path, typer.Option(
            help="Directory with budget_strips_<source>_YYYY_MM.nc and "
                 "lwa_budget_climatology_<source>.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for figure outputs")],
        source: Annotated[str, typer.Option(help="era5 or merra2")] = "era5",
        tracks_directory: Annotated[Optional[Path], typer.Option(
            help="Track database directory for the mean-track overlay")] = None,
        basins: Annotated[Optional[list[str]], typer.Option()] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature, et, or both")] = "both",
        n_mc: Annotated[int, typer.Option()] = 300,
        n_workers: Annotated[Optional[int], typer.Option()] = None,
        year_start: Annotated[int, typer.Option()] = 2000,
        year_end: Annotated[int, typer.Option()] = 2022,
        combine_wp_na: Annotated[bool, typer.Option(
            help="Single ERA5 figure: WP | NA six-panel blocks, one shared "
                 "colorbar, column separator, no suptitle.")] = False,
        fig4_merra2_wp_na: Annotated[bool, typer.Option(
            help="MERRA-2 Fig. 4: WP | NA residual + LH + RAD + ANA "
                 "(two data rows x four columns); one colour bar.")] = False,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if basins is None:
        basins = ["WP", "NA"]
    if source not in SOURCE_CONFIG:
        raise typer.BadParameter(f"source must be one of {list(SOURCE_CONFIG)}")
    if fig4_merra2_wp_na and combine_wp_na:
        raise typer.BadParameter(
            "Use only one of --combine-wp-na and --fig4-merra2-wp-na")

    storms_df = pd.read_csv(
        tracks_csv,
        parse_dates=["recurv_time", "et_time"],
        keep_default_na=False,
        na_values=[""],
    )

    td = (tracks_directory if tracks_directory is not None
          else Path(tracks_csv).resolve().parent)

    if fig4_merra2_wp_na:
        refs = (["recurvature", "et"] if reference == "both" else [reference])
        for ref in refs:
            plot_merra2_fig4_wp_na_combined(
                storms_df=storms_df, strip_dir=strip_directory, reference=ref,
                n_mc=n_mc, n_workers=n_workers,
                year_start=year_start, year_end=year_end,
                tracks_dir=td,
                out_path=Path(output_directory)
                / f"fig4_lwa_merra2_wp_na_{ref}.png",
            )
        return

    if combine_wp_na:
        if source != "era5":
            raise typer.BadParameter("--combine-wp-na requires --source era5")
        refs = (["recurvature", "et"] if reference == "both" else [reference])
        for ref in refs:
            plot_budget_wp_na_combined(
                storms_df=storms_df, strip_dir=strip_directory,
                source=source, reference=ref,
                n_mc=n_mc, n_workers=n_workers,
                year_start=year_start, year_end=year_end,
                tracks_dir=td,
                out_path=Path(output_directory)
                / f"fig3_lwa_budget_{source}_wp_na_{ref}.png",
            )
        return

    refs = (["recurvature", "et"] if reference == "both" else [reference])

    for basin in basins:
        for ref in refs:
            plot_budget(storms_df=storms_df, basin=basin,
                        strip_dir=strip_directory,
                        out_dir=output_directory,
                        source=source,
                        reference=ref, n_mc=n_mc, n_workers=n_workers,
                        year_start=year_start, year_end=year_end,
                        tracks_dir=td)


if __name__ == "__main__":
    typer.run(main)
