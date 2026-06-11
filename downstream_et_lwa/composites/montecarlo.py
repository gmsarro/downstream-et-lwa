"""Monte Carlo significance for 2-D recurvature-anchored composite maps:
QJ16-style random time reshuffle with fixed per-storm spatial centers, parallel
iterations, per-pixel 5th/95th percentile envelopes."""

from __future__ import annotations

import concurrent.futures
import datetime
import logging
import multiprocessing as mp
import time
from typing import Sequence

import numpy as np
import pandas as pd

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.composites.engine as engine
import downstream_et_lwa.composites.io as composite_io
import downstream_et_lwa.data_registry as data_registry
import downstream_et_lwa.grid_utils as grid_utils

_LOG = logging.getLogger(__name__)

DEFAULT_QUANTITY = {
    "era5_budget_termI": "anom",
    "era5_budget_termII": "anom",
    "era5_budget_termIII": "anom",
    "era5_lh_lwa": "anom",
    "era5_dadt": "anom",
    "era5_residual": "anom",
    "merra2_budget_termI": "anom",
    "merra2_budget_termII": "anom",
    "merra2_budget_termIII": "anom",
    "merra2_dadt": "anom",
    "merra2_residual": "anom",
    "merra2_heat_fortran_DTDTMST": "anom",
    "merra2_heat_fortran_DTDTRAD": "anom",
    "merra2_heat_fortran_DTDTANA": "anom",
    "mpas_budget_termI": "anom",
    "mpas_budget_termII": "anom",
    "mpas_budget_termIII": "anom",
    "mpas_lh_lwa": "anom",
    "mpas_dadt": "anom",
    "mpas_residual": "anom",
    "mpas_current_budget_termI": "anom",
    "mpas_current_budget_termII": "anom",
    "mpas_current_budget_termIII": "anom",
    "mpas_current_lh_lwa": "anom",
    "mpas_current_dadt": "anom",
    "mpas_current_residual": "anom",
    "mpas_future_budget_termI": "anom",
    "mpas_future_budget_termII": "anom",
    "mpas_future_budget_termIII": "anom",
    "mpas_future_lh_lwa": "anom",
    "mpas_future_dadt": "anom",
    "mpas_future_residual": "anom",
}


def panel_quantity(*, var_key: str, override: dict[str, str] | None = None) -> str:
    if override and var_key in override:
        return override[var_key]
    return DEFAULT_QUANTITY.get(var_key, "absolute")


def _draw_random_ref(
        *,
        ref_time: pd.Timestamp,
        years_available: Sequence[int],
        day_offset_range: tuple[int, int],
        rng: np.random.Generator,
) -> pd.Timestamp:
    yr = int(rng.choice(years_available))
    offset = int(rng.integers(day_offset_range[0], day_offset_range[1] + 1))
    try:
        new_ref = ref_time.replace(year=yr) + datetime.timedelta(days=offset)
    except ValueError:
        new_ref = ref_time + datetime.timedelta(days=offset)
    return new_ref


def _per_storm_random_volume(
        *,
        storm: dict,
        var_keys: list[str],
        premian: tuple[float, ...],
        rng: np.random.Generator,
        years_available: Sequence[int],
        day_offset_range: tuple[int, int],
        file_cache: dict,
        reference: str = "recurvature",
) -> dict[str, np.ndarray]:
    if reference == "recurvature":
        ref_time = pd.Timestamp(storm["recurv_time"])
        center_lat = float(storm["recurv_lat"])
        center_lon = float(storm["recurv_lon"]) % 360
    else:
        ref_time = pd.Timestamp(storm["et_time"])
        center_lat = float(storm["et_lat"])
        center_lon = float(storm["et_lon"]) % 360

    if pd.isna(ref_time) or center_lat < 0 or center_lat > 89:
        return {}

    rand_ref = _draw_random_ref(
        ref_time=ref_time, years_available=years_available,
        day_offset_range=day_offset_range, rng=rng)

    shape = (composite_config.BOX_NLAT, composite_config.BOX_NLON,
             composite_config.N_LAGS)
    out: dict[str, np.ndarray] = {}

    for vk in var_keys:
        ds = data_registry.get(key=vk)
        if ds is None:
            continue
        vol = np.full(shape, np.nan, dtype=np.float64)
        for lag_idx, lag_h in enumerate(composite_config.LAG_HOURS):
            target_dt = (rand_ref + datetime.timedelta(hours=int(lag_h))).to_pydatetime()
            raw = data_registry.load_snapshot(
                source=ds, target_dt=target_dt, cache=file_cache)
            if raw is None:
                continue
            field = grid_utils.prepare_field(data=raw, source=ds)
            if field is None:
                continue
            patch = grid_utils.extract_storm_patch(
                field_2d=field, center_lat=center_lat, center_lon=center_lon)
            if patch is None or patch.shape != (composite_config.BOX_NLAT,
                                                composite_config.BOX_NLON):
                continue
            vol[:, :, lag_idx] = patch

        if not np.isfinite(vol).any():
            continue
        if not all(s <= 0.0 for s in premian):
            smoothed = composite_io.smooth_composite_volume(
                field_3d=vol, sigma_3d=premian)
            if smoothed is None:
                continue
            vol = smoothed
        out[vk] = vol.astype(np.float32)
    return out


def _add_per_storm_rate_volumes(*, storm_volumes: dict[str, np.ndarray]) -> None:
    coord_sec = np.asarray(composite_config.LAG_HOURS, dtype=np.float64) * 3600.0
    for prefix in engine._DADT_RESID_PREFIXES:
        lwa_key = f"{prefix}_lwa"
        if lwa_key not in storm_volumes:
            continue
        lwa_vol = storm_volumes[lwa_key].astype(np.float64)
        dadt_vol = np.gradient(lwa_vol, coord_sec, axis=2).astype(np.float32)
        storm_volumes[f"{prefix}_dadt"] = dadt_vol
        term_keys = [f"{prefix}_budget_term{n}" for n in ("I", "II", "III")]
        if all(k in storm_volumes for k in term_keys):
            res = dadt_vol.astype(np.float64)
            for tk in term_keys:
                res = res - storm_volumes[tk].astype(np.float64)
            storm_volumes[f"{prefix}_residual"] = res.astype(np.float32)


def _collapse_iteration(
        *,
        sum_d: dict[tuple[str, str], np.ndarray],
        count_d: dict[tuple[str, str], np.ndarray],
        var_keys: list[str],
        accum_basins: list[str],
        t_window: tuple[float, float],
        baseline_window: tuple[float, float],
        lag_hours: np.ndarray,
        var_quantity: dict[str, str] | None,
) -> dict[tuple[str, str], np.ndarray]:
    mask_t = (lag_hours >= t_window[0]) & (lag_hours <= t_window[1])
    mask_b = (lag_hours >= baseline_window[0]) & (lag_hours <= baseline_window[1])

    out = {}
    for basin in accum_basins:
        for vk in var_keys:
            key = (basin, vk)
            if key not in sum_d:
                continue
            n = count_d[key].astype(np.float64)
            mean = np.full(sum_d[key].shape, np.nan, dtype=np.float64)
            valid = n > 0
            mean[valid] = sum_d[key][valid] / n[valid]

            mw = np.full(mean.shape[:2], np.nan, dtype=np.float64)
            if mask_t.any():
                mw_slab = mean[:, :, mask_t]
                mw_finite = np.isfinite(mw_slab)
                with np.errstate(invalid="ignore"):
                    cnt = mw_finite.sum(axis=2)
                    sm = np.where(mw_finite, mw_slab, 0.0).sum(axis=2)
                mw = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)

            quant = panel_quantity(var_key=vk, override=var_quantity)
            if quant == "absolute":
                out[key] = mw
            else:
                bw = np.full(mean.shape[:2], np.nan, dtype=np.float64)
                if mask_b.any():
                    bw_slab = mean[:, :, mask_b]
                    bw_finite = np.isfinite(bw_slab)
                    with np.errstate(invalid="ignore"):
                        cntb = bw_finite.sum(axis=2)
                        smb = np.where(bw_finite, bw_slab, 0.0).sum(axis=2)
                    bw = np.where(cntb > 0, smb / np.maximum(cntb, 1), np.nan)
                out[key] = mw - bw
    return out


def _process_iteration(args: tuple) -> dict[tuple[str, str], np.ndarray]:
    (storm_records, track_dict, var_keys, accum_basins, pool_wpna, reference,
     years_available, day_offset_range, mc_seed, premian,
     t_window, baseline_window, var_quantity) = args

    rng = np.random.default_rng(mc_seed)

    shape = (composite_config.BOX_NLAT, composite_config.BOX_NLON,
             composite_config.N_LAGS)
    sum_d: dict[tuple[str, str], np.ndarray] = {}
    count_d: dict[tuple[str, str], np.ndarray] = {}
    for b in accum_basins:
        for vk in var_keys:
            k = (b, vk)
            sum_d[k] = np.zeros(shape, dtype=np.float64)
            count_d[k] = np.zeros(shape, dtype=np.int32)

    file_cache: dict = {}
    for storm in storm_records:
        sid = storm["storm_id"]
        if sid not in track_dict:
            continue
        basin = storm["basin"]
        target_basin = "WPNA" if pool_wpna else basin

        storm_volumes = _per_storm_random_volume(
            storm=storm,
            var_keys=[vk for vk in var_keys
                      if not vk.endswith("_dadt") and not vk.endswith("_residual")],
            premian=premian, rng=rng, years_available=years_available,
            day_offset_range=day_offset_range, file_cache=file_cache,
            reference=reference,
        )

        _add_per_storm_rate_volumes(storm_volumes=storm_volumes)

        for vk in var_keys:
            if vk not in storm_volumes:
                continue
            vol = storm_volumes[vk]
            key = (target_basin, vk)
            if key not in sum_d:
                continue
            for lag_idx in range(composite_config.N_LAGS):
                sl = vol[:, :, lag_idx]
                finite = np.isfinite(sl)
                if not finite.any():
                    continue
                vals = np.where(finite, sl, 0.0).astype(np.float64)
                sum_d[key][:, :, lag_idx] += vals
                count_d[key][:, :, lag_idx] += finite.astype(np.int32)

    data_registry.close_cache(cache=file_cache)

    return _collapse_iteration(
        sum_d=sum_d, count_d=count_d, var_keys=var_keys,
        accum_basins=accum_basins, t_window=t_window,
        baseline_window=baseline_window,
        lag_hours=np.asarray(composite_config.LAG_HOURS, dtype=np.float64),
        var_quantity=var_quantity,
    )


def build_mc_envelope_2d(
        *,
        storms_df: pd.DataFrame,
        full_tracks: dict[str, pd.DataFrame],
        var_keys: list[str],
        basins: list[str],
        t_window: tuple[float, float],
        baseline_window: tuple[float, float],
        n_iter: int = 400,
        n_workers: int = 24,
        seed: int = 42,
        day_offset_range: tuple[int, int] = (-7, 7),
        reference: str = "recurvature",
        premean_volume_sigma_3d: tuple[float, ...] | None = None,
        var_quantity: dict[str, str] | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    var_keys = engine._add_per_storm_rate_keys(var_keys=list(var_keys))
    acc_basins, filt_basins, pool_wpna = engine.composite_accumulator_basins(
        basins=basins)
    storm_records = (
        storms_df[storms_df["basin"].isin(filt_basins)].to_dict("records"))
    needed_sids = {r["storm_id"] for r in storm_records}
    track_dict = {sid: full_tracks[sid] for sid in needed_sids
                  if sid in full_tracks}

    years_available = sorted(set(int(r["season"]) for r in storm_records
                                 if not pd.isna(r.get("season"))))
    if not years_available:
        years_available = sorted(
            set(pd.to_datetime(r["recurv_time"]).year
                for r in storm_records
                if not pd.isna(r.get("recurv_time"))))

    premian = engine._resolved_premean_volume_sigma(
        sigma_3d=premean_volume_sigma_3d)

    _LOG.info(
        "[mc_2d] n_iter=%s, n_workers=%s, n_storms=%s, n_var=%s, basins=%s, "
        "t_window=%s, baseline_window=%s, day_offset=%s, years=%s",
        n_iter, n_workers, len(storm_records), len(var_keys), acc_basins,
        t_window, baseline_window, day_offset_range, years_available)

    rng = np.random.default_rng(seed)
    iter_seeds = rng.integers(low=0, high=2**31 - 1, size=n_iter)

    worker_args = [
        (storm_records, track_dict, var_keys, acc_basins, pool_wpna, reference,
         years_available, day_offset_range, int(iter_seeds[i]), premian,
         t_window, baseline_window, var_quantity)
        for i in range(n_iter)
    ]

    out: dict[tuple[str, str], np.ndarray] = {}
    for b in acc_basins:
        for vk in var_keys:
            out[(b, vk)] = np.full(
                (n_iter, composite_config.BOX_NLAT, composite_config.BOX_NLON),
                np.nan, dtype=np.float32)

    t0 = time.time()
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("fork")) as pool:
        futures = {pool.submit(_process_iteration, a): i
                   for i, a in enumerate(worker_args)}
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                res = fut.result()
            except Exception:
                _LOG.exception("[mc_2d] iteration %s failed", i)
                continue
            for key, m in res.items():
                if key in out:
                    out[key][i] = m.astype(np.float32)
            completed += 1
            if (completed % max(1, n_iter // 20) == 0
                    or completed == n_iter):
                el = time.time() - t0
                rate = completed / max(el, 1e-9)
                eta = (n_iter - completed) / max(rate, 1e-9)
                _LOG.info("[mc_2d]   %s/%s done (%.0fs elapsed, ETA %.0fs)",
                          completed, n_iter, el, eta)
    return out


def percentile_envelope(
        *,
        iter_maps: np.ndarray,
        p_low: float = 5.0,
        p_high: float = 95.0,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(iter_maps)
    n_finite = finite.sum(axis=0)
    p5 = np.full(iter_maps.shape[1:], np.nan, dtype=np.float64)
    p95 = np.full(iter_maps.shape[1:], np.nan, dtype=np.float64)
    valid = n_finite >= 10
    if valid.any():
        p5[valid] = np.nanpercentile(iter_maps, p_low, axis=0)[valid]
        p95[valid] = np.nanpercentile(iter_maps, p_high, axis=0)[valid]
    return p5, p95


def significance_mask_from_envelope(
        *,
        actual_2d: np.ndarray | None,
        p5: np.ndarray | None,
        p95: np.ndarray | None,
) -> np.ndarray | None:
    if actual_2d is None or p5 is None or p95 is None:
        return None
    out = np.zeros(actual_2d.shape, dtype=bool)
    finite = np.isfinite(actual_2d) & np.isfinite(p5) & np.isfinite(p95)
    out[finite] = (actual_2d[finite] < p5[finite]) | (actual_2d[finite] > p95[finite])
    return out
