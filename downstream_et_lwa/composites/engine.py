"""Fixed-center storm-relative compositing engine: per-storm (rel_lat, rel_lon, lag)
volumes are Gaussian-smoothed before NaN-aware accumulation into ensemble sums;
per-storm dA/dt and budget-residual pseudo-variables are derived alongside.
Parallel mode splits storms across forked workers and merges partial accumulators."""

from __future__ import annotations

import concurrent.futures
import datetime
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.composites.io as composite_io
import downstream_et_lwa.data_registry as data_registry
import downstream_et_lwa.grid_utils as grid_utils
import downstream_et_lwa.tracks as tracks

_LOG = logging.getLogger(__name__)

_DADT_RESID_PREFIXES = (
    "era5", "mpas",
    "mpas_current", "mpas_future",
)


def _resolved_premean_volume_sigma(
        *,
        sigma_3d: tuple[float, ...] | None,
) -> tuple[float, ...]:
    if sigma_3d is None:
        return tuple(float(x) for x in composite_config.COMPOSITE_PREMEAN_VOLUME_SIGMA_3D)
    return tuple(float(x) for x in sigma_3d)


def _add_per_storm_rate_keys(*, var_keys: list[str]) -> list[str]:
    out = list(var_keys)
    for prefix in _DADT_RESID_PREFIXES:
        if f"{prefix}_lwa" in var_keys:
            dadt_key = f"{prefix}_dadt"
            if dadt_key not in out:
                out.append(dadt_key)
            term_keys = [f"{prefix}_budget_term{n}" for n in ("I", "II", "III")]
            if all(k in var_keys for k in term_keys):
                res_key = f"{prefix}_residual"
                if res_key not in out:
                    out.append(res_key)
    return out


def _accumulate_per_storm_volume(
        *,
        vol: np.ndarray,
        target_basin: str,
        var_key: str,
        partial_sum: dict[tuple[str, str], np.ndarray],
        partial_sumsq: dict[tuple[str, str], np.ndarray],
        partial_n: dict[tuple[str, str], np.ndarray],
) -> None:
    key = (target_basin, var_key)
    if key not in partial_sum:
        return
    for lag_idx in range(composite_config.N_LAGS):
        sl = vol[:, :, lag_idx]
        finite = np.isfinite(sl)
        if not finite.any():
            continue
        vals = np.where(finite, sl, 0.0)
        partial_sum[key][:, :, lag_idx] += vals
        partial_sumsq[key][:, :, lag_idx] += vals * vals
        partial_n[key][:, :, lag_idx] += finite.astype(np.int32)


def _compute_per_storm_dadt_residual(
        *,
        storm_volumes: dict[str, np.ndarray],
        target_basin: str,
        partial_sum: dict[tuple[str, str], np.ndarray],
        partial_sumsq: dict[tuple[str, str], np.ndarray],
        partial_n: dict[tuple[str, str], np.ndarray],
) -> None:
    coord_sec = np.asarray(composite_config.LAG_HOURS, dtype=np.float64) * 3600.0
    for prefix in _DADT_RESID_PREFIXES:
        lwa_key = f"{prefix}_lwa"
        if lwa_key not in storm_volumes:
            continue
        lwa_vol = storm_volumes[lwa_key]
        dadt_vol = np.gradient(lwa_vol, coord_sec, axis=2)
        _accumulate_per_storm_volume(
            vol=dadt_vol, target_basin=target_basin, var_key=f"{prefix}_dadt",
            partial_sum=partial_sum, partial_sumsq=partial_sumsq, partial_n=partial_n,
        )
        term_keys = [f"{prefix}_budget_term{n}" for n in ("I", "II", "III")]
        if all(k in storm_volumes for k in term_keys):
            residual_vol = dadt_vol.copy()
            for tk in term_keys:
                residual_vol = residual_vol - storm_volumes[tk]
            _accumulate_per_storm_volume(
                vol=residual_vol, target_basin=target_basin,
                var_key=f"{prefix}_residual",
                partial_sum=partial_sum, partial_sumsq=partial_sumsq,
                partial_n=partial_n,
            )


def composite_accumulator_basins(
        *,
        basins: list[str] | None,
) -> tuple[list[str], list[str], bool]:
    if basins is not None and len(basins) == 1 and str(basins[0]).upper() == "WPNA":
        return (["WPNA"], ["WP", "NA"], True)
    b = list(basins or [])
    return (b, b, False)


class CompositeAccumulator:
    def __init__(self, *, var_keys: list[str], basins: list[str]) -> None:
        self.var_keys = list(var_keys)
        self.basins = list(basins)

        self.composite_sum: dict[tuple[str, str], np.ndarray] = {}
        self.composite_sumsq: dict[tuple[str, str], np.ndarray] = {}
        self.composite_n: dict[tuple[str, str], np.ndarray] = {}

        for basin in basins:
            for vk in var_keys:
                key = (basin, vk)
                self.composite_sum[key] = np.zeros(
                    (composite_config.BOX_NLAT, composite_config.BOX_NLON,
                     composite_config.N_LAGS), dtype=np.float64)
                self.composite_sumsq[key] = np.zeros(
                    (composite_config.BOX_NLAT, composite_config.BOX_NLON,
                     composite_config.N_LAGS), dtype=np.float64)
                self.composite_n[key] = np.zeros(
                    (composite_config.BOX_NLAT, composite_config.BOX_NLON,
                     composite_config.N_LAGS), dtype=np.int32)

        self.metadata: dict[str, list[dict]] = {basin: [] for basin in basins}
        self.track_lats: dict[str, list[np.ndarray]] = {basin: [] for basin in basins}
        self.track_lons: dict[str, list[np.ndarray]] = {basin: [] for basin in basins}
        self.track_winds: dict[str, list[np.ndarray]] = {basin: [] for basin in basins}
        self.track_pres: dict[str, list[np.ndarray]] = {basin: [] for basin in basins}
        self.track_rel_lats: dict[str, list[np.ndarray]] = {basin: [] for basin in basins}
        self.track_rel_lons: dict[str, list[np.ndarray]] = {basin: [] for basin in basins}

    def add_patch(self, *, basin: str, var_key: str, lag_idx: int,
                  patch: np.ndarray | None) -> None:
        key = (basin, var_key)
        if key not in self.composite_sum:
            return
        if patch is None or patch.shape != (composite_config.BOX_NLAT,
                                            composite_config.BOX_NLON):
            return
        finite_mask = np.isfinite(patch)
        if not finite_mask.any():
            return
        vals = np.where(finite_mask, patch, 0.0)
        self.composite_sum[key][:, :, lag_idx] += vals
        self.composite_sumsq[key][:, :, lag_idx] += vals * vals
        self.composite_n[key][:, :, lag_idx] += finite_mask.astype(np.int32)

    def add_storm_metadata(self, *, basin: str, storm_info: dict,
                           lats: np.ndarray, lons: np.ndarray,
                           winds: np.ndarray, pres: np.ndarray,
                           center_lat: float | None = None,
                           center_lon: float | None = None) -> None:
        self.metadata[basin].append(storm_info)
        self.track_lats[basin].append(lats)
        self.track_lons[basin].append(lons)
        self.track_winds[basin].append(winds)
        self.track_pres[basin].append(pres)

        if center_lat is not None and center_lon is not None:
            rel_lat = lats - center_lat
            rel_lon = lons - center_lon
            rel_lon = np.where(rel_lon > 180, rel_lon - 360, rel_lon)
            rel_lon = np.where(rel_lon < -180, rel_lon + 360, rel_lon)
        else:
            rel_lat = np.full_like(lats, np.nan)
            rel_lon = np.full_like(lons, np.nan)
        self.track_rel_lats[basin].append(rel_lat)
        self.track_rel_lons[basin].append(rel_lon)

    def get_mean(self, *, basin: str, var_key: str) -> np.ndarray:
        key = (basin, var_key)
        n = self.composite_n[key].astype(np.float64)
        mean = np.full_like(self.composite_sum[key], np.nan)
        valid = n > 0
        mean[valid] = self.composite_sum[key][valid] / n[valid]
        return mean

    def get_count(self, *, basin: str, var_key: str) -> np.ndarray:
        key = (basin, var_key)
        return self.composite_n[key].max(axis=(0, 1))

    def get_count_field(self, *, basin: str, var_key: str) -> np.ndarray:
        return self.composite_n[(basin, var_key)]

    def get_sumsq(self, *, basin: str, var_key: str) -> np.ndarray:
        return self.composite_sumsq[(basin, var_key)]

    def get_stderr(self, *, basin: str, var_key: str) -> np.ndarray:
        key = (basin, var_key)
        n = self.composite_n[key].astype(np.float64)
        mean = self.get_mean(basin=basin, var_key=var_key)
        sumsq = self.composite_sumsq[key]
        var = np.full_like(mean, np.nan)
        valid = n > 1
        var[valid] = sumsq[valid] / n[valid] - mean[valid] ** 2
        var = np.maximum(var, 0.0)
        se = np.full_like(mean, np.nan)
        se[valid] = np.sqrt(var[valid] / n[valid])
        return se

    def get_mean_rel_track(self, *, basin: str) -> tuple[np.ndarray, np.ndarray]:
        if not self.track_rel_lats[basin]:
            return (np.full(composite_config.N_LAGS, np.nan),
                    np.full(composite_config.N_LAGS, np.nan))
        arr_lat = np.array(self.track_rel_lats[basin])
        arr_lon = np.array(self.track_rel_lons[basin])
        return np.nanmean(arr_lat, axis=0), np.nanmean(arr_lon, axis=0)

    def get_all_rel_tracks(self, *, basin: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.track_rel_lats[basin]:
            return None, None
        return (np.array(self.track_rel_lats[basin]),
                np.array(self.track_rel_lons[basin]))


def _get_fixed_center(*, storm: pd.Series, reference: str) -> tuple[float, float]:
    if reference == "recurvature":
        return float(storm["recurv_lat"]), float(storm["recurv_lon"])
    return float(storm["et_lat"]), float(storm["et_lon"])


def build_composites(
        *,
        storms_df: pd.DataFrame,
        full_tracks: dict[str, pd.DataFrame],
        var_keys: list[str] | None = None,
        basins: list[str] | None = None,
        reference: str = "recurvature",
        premean_volume_sigma_3d: tuple[float, ...] | None = None,
) -> CompositeAccumulator:
    if var_keys is None:
        var_keys = data_registry.list_keys()
    if basins is None:
        basins = ["WP", "NA", "EP"]

    var_keys = _add_per_storm_rate_keys(var_keys=var_keys)

    acc_basins, filt_basins, pool_wpna = composite_accumulator_basins(basins=basins)
    accum = CompositeAccumulator(var_keys=var_keys, basins=acc_basins)
    file_cache: dict = {}

    premian = _resolved_premean_volume_sigma(sigma_3d=premean_volume_sigma_3d)

    storm_list = storms_df[storms_df["basin"].isin(filt_basins)]
    n_storms = len(storm_list)

    _LOG.info("Building %s-relative composites for %d storms, %d variables...",
              reference, n_storms, len(var_keys))
    _LOG.info("Center: FIXED at %s location for all lags", reference)

    for storm_idx, (_, storm) in enumerate(storm_list.iterrows()):
        sid = storm["storm_id"]
        basin = storm["basin"]
        target_basin = "WPNA" if pool_wpna else basin

        if sid not in full_tracks:
            continue

        track_df = full_tracks[sid]

        if reference == "recurvature":
            ref_time = pd.Timestamp(storm["recurv_time"])
        else:
            ref_time = pd.Timestamp(storm["et_time"])

        if pd.isna(ref_time):
            continue

        center_lat, center_lon = _get_fixed_center(storm=storm, reference=reference)
        center_lon = center_lon % 360

        if center_lat < 0 or center_lat > 89:
            continue

        lats, lons, winds, pres = tracks.interpolate_track_to_lags(
            track_df=track_df, reference=reference)

        storm_info = {
            "storm_id": sid, "name": storm["name"],
            "basin": basin, "season": storm["season"],
            "recurv_lat": storm["recurv_lat"],
            "recurv_lon": storm["recurv_lon"],
            "et_lat": storm["et_lat"], "et_lon": storm["et_lon"],
        }
        accum.add_storm_metadata(basin=target_basin, storm_info=storm_info,
                                 lats=lats, lons=lons, winds=winds, pres=pres,
                                 center_lat=center_lat, center_lon=center_lon)

        if (storm_idx + 1) % 20 == 0:
            _LOG.info("[%d/%d] %s (%s, %s, %s)", storm_idx + 1, n_storms,
                      sid, storm["name"], basin, storm["season"])

        rate_source_keys = set()
        for prefix in _DADT_RESID_PREFIXES:
            rate_source_keys.add(f"{prefix}_lwa")
            for n in ("I", "II", "III"):
                rate_source_keys.add(f"{prefix}_budget_term{n}")
        storm_volumes: dict[str, np.ndarray] = {}

        for vk in var_keys:
            ds = data_registry.get(key=vk)
            if ds is None:
                continue

            vol = np.full((composite_config.BOX_NLAT, composite_config.BOX_NLON,
                           composite_config.N_LAGS), np.nan, dtype=np.float64)
            for lag_idx, lag_h in enumerate(composite_config.LAG_HOURS):
                target_dt = (ref_time
                             + datetime.timedelta(hours=int(lag_h))).to_pydatetime()

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
                if smoothed is not None:
                    vol = smoothed

            for lag_idx in range(composite_config.N_LAGS):
                sl = vol[:, :, lag_idx]
                if np.isfinite(sl).any():
                    accum.add_patch(basin=target_basin, var_key=vk,
                                    lag_idx=lag_idx, patch=sl.astype(np.float32))

            if vk in rate_source_keys:
                storm_volumes[vk] = vol

        if storm_volumes:
            _compute_per_storm_dadt_residual(
                storm_volumes=storm_volumes, target_basin=target_basin,
                partial_sum=accum.composite_sum,
                partial_sumsq=accum.composite_sumsq,
                partial_n=accum.composite_n,
            )

    data_registry.close_cache(cache=file_cache)

    _LOG.info("Composite complete. Storm counts per basin:")
    for bsn in acc_basins:
        _LOG.info("  %s: %d storms", bsn, len(accum.metadata[bsn]))

    return accum


def _process_storm_chunk(args: tuple) -> tuple:
    (storm_records, track_dict, var_keys, accum_basins, pool_wpna, reference,
     worker_id, premian) = args

    shape = (composite_config.BOX_NLAT, composite_config.BOX_NLON,
             composite_config.N_LAGS)
    partial_sum: dict[tuple[str, str], np.ndarray] = {}
    partial_sumsq: dict[tuple[str, str], np.ndarray] = {}
    partial_n: dict[tuple[str, str], np.ndarray] = {}
    for b in accum_basins:
        for vk in var_keys:
            k = (b, vk)
            partial_sum[k] = np.zeros(shape, dtype=np.float64)
            partial_sumsq[k] = np.zeros(shape, dtype=np.float64)
            partial_n[k] = np.zeros(shape, dtype=np.int32)
    meta_list = []

    file_cache: dict = {}
    n_done = 0

    for storm in storm_records:
        sid = storm["storm_id"]
        basin = storm["basin"]
        target_basin = "WPNA" if pool_wpna else basin
        if sid not in track_dict:
            continue

        track_df = track_dict[sid]

        if reference == "recurvature":
            ref_time = pd.Timestamp(storm["recurv_time"])
        else:
            ref_time = pd.Timestamp(storm["et_time"])
        if pd.isna(ref_time):
            continue

        center_lat, center_lon = (
            (float(storm["recurv_lat"]), float(storm["recurv_lon"]))
            if reference == "recurvature"
            else (float(storm["et_lat"]), float(storm["et_lon"]))
        )
        center_lon = center_lon % 360
        if center_lat < 0 or center_lat > 89:
            continue

        lats, lons, winds, pres = tracks.interpolate_track_to_lags(
            track_df=track_df, reference=reference)

        rel_lat = lats - center_lat
        rel_lon = lons - center_lon
        rel_lon = np.where(rel_lon > 180, rel_lon - 360, rel_lon)
        rel_lon = np.where(rel_lon < -180, rel_lon + 360, rel_lon)

        meta_list.append({
            "storm_info": {
                "storm_id": sid, "name": storm["name"],
                "basin": basin, "season": storm["season"],
                "recurv_lat": storm["recurv_lat"],
                "recurv_lon": storm["recurv_lon"],
                "et_lat": storm["et_lat"], "et_lon": storm["et_lon"],
            },
            "basin": target_basin,
            "lats": lats, "lons": lons, "winds": winds, "pres": pres,
            "rel_lat": rel_lat, "rel_lon": rel_lon,
        })

        rate_source_keys = set()
        for prefix in _DADT_RESID_PREFIXES:
            rate_source_keys.add(f"{prefix}_lwa")
            for n in ("I", "II", "III"):
                rate_source_keys.add(f"{prefix}_budget_term{n}")
        storm_volumes: dict[str, np.ndarray] = {}

        for vk in var_keys:
            ds = data_registry.get(key=vk)
            if ds is None:
                continue
            vol = np.full(shape, np.nan, dtype=np.float64)
            for lag_idx, lag_h in enumerate(composite_config.LAG_HOURS):
                target_dt = (ref_time
                             + datetime.timedelta(hours=int(lag_h))).to_pydatetime()
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
                if smoothed is not None:
                    vol = smoothed

            key = (target_basin, vk)
            for lag_idx in range(composite_config.N_LAGS):
                patch = vol[:, :, lag_idx]
                finite = np.isfinite(patch)
                if not finite.any():
                    continue
                vals = np.where(finite, patch, 0.0)
                partial_sum[key][:, :, lag_idx] += vals
                partial_sumsq[key][:, :, lag_idx] += vals * vals
                partial_n[key][:, :, lag_idx] += finite.astype(np.int32)

            if vk in rate_source_keys:
                storm_volumes[vk] = vol

        if storm_volumes:
            _compute_per_storm_dadt_residual(
                storm_volumes=storm_volumes, target_basin=target_basin,
                partial_sum=partial_sum, partial_sumsq=partial_sumsq,
                partial_n=partial_n,
            )

        n_done += 1

    data_registry.close_cache(cache=file_cache)
    return partial_sum, partial_sumsq, partial_n, meta_list, worker_id, n_done


def build_composites_parallel(
        *,
        storms_df: pd.DataFrame,
        full_tracks: dict[str, pd.DataFrame],
        var_keys: list[str] | None = None,
        basins: list[str] | None = None,
        reference: str = "recurvature",
        n_workers: int = 12,
        premean_volume_sigma_3d: tuple[float, ...] | None = None,
) -> CompositeAccumulator:
    if var_keys is None:
        var_keys = data_registry.list_keys()
    if basins is None:
        basins = ["WP", "NA", "EP"]

    acc_basins, filt_basins, pool_wpna = composite_accumulator_basins(basins=basins)
    storm_list = storms_df[storms_df["basin"].isin(filt_basins)]
    n_storms = len(storm_list)

    var_keys = _add_per_storm_rate_keys(var_keys=var_keys)

    _LOG.info("Building %s-relative composites for %d storms, %d variables, %d workers...",
              reference, n_storms, len(var_keys), n_workers)

    storm_records = storm_list.to_dict("records")

    chunks: list[list[dict]] = [[] for _ in range(n_workers)]
    for i, rec in enumerate(storm_records):
        chunks[i % n_workers].append(rec)

    needed_sids = set(storm_list["storm_id"])
    track_dict = {sid: full_tracks[sid] for sid in needed_sids
                  if sid in full_tracks}

    premian = _resolved_premean_volume_sigma(sigma_3d=premean_volume_sigma_3d)

    worker_args = [
        (chunk, track_dict, var_keys, acc_basins, pool_wpna, reference, wid,
         premian)
        for wid, chunk in enumerate(chunks) if chunk
    ]

    accum = CompositeAccumulator(var_keys=var_keys, basins=acc_basins)
    total_done = 0

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("fork")) as pool:
        futures = {pool.submit(_process_storm_chunk, a): a[6]
                   for a in worker_args}

        for future in concurrent.futures.as_completed(futures):
            wid = futures[future]
            try:
                p_sum, p_sumsq, p_n, meta, _, n_done = future.result()
            except Exception:
                _LOG.exception("Worker %s failed", wid)
                continue

            for key in p_sum:
                if key in accum.composite_sum:
                    accum.composite_sum[key] += p_sum[key]
                    accum.composite_sumsq[key] += p_sumsq[key]
                    accum.composite_n[key] += p_n[key]

            for m in meta:
                b = m["basin"]
                accum.metadata[b].append(m["storm_info"])
                accum.track_lats[b].append(m["lats"])
                accum.track_lons[b].append(m["lons"])
                accum.track_winds[b].append(m["winds"])
                accum.track_pres[b].append(m["pres"])
                accum.track_rel_lats[b].append(m["rel_lat"])
                accum.track_rel_lons[b].append(m["rel_lon"])

            total_done += n_done
            _LOG.info("Worker %s done: %d storms (total %d/%d)",
                      wid, n_done, total_done, n_storms)

    _LOG.info("Composite complete. Storm counts per basin:")
    for bsn in acc_basins:
        _LOG.info("  %s: %d storms", bsn, len(accum.metadata[bsn]))

    return accum


def extract_single_storm(
        *,
        storms_df: pd.DataFrame,
        full_tracks: dict[str, pd.DataFrame],
        storm_id: str,
        var_keys: list[str] | None = None,
        reference: str = "recurvature",
        premean_volume_sigma_3d: tuple[float, ...] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if var_keys is None:
        var_keys = data_registry.list_keys()

    storm = storms_df[storms_df["storm_id"] == storm_id].iloc[0]
    track_df = full_tracks[storm_id]

    if reference == "recurvature":
        ref_time = pd.Timestamp(storm["recurv_time"])
    else:
        ref_time = pd.Timestamp(storm["et_time"])

    center_lat, center_lon = _get_fixed_center(storm=storm, reference=reference)
    center_lon = center_lon % 360

    lats, lons, winds, pres = tracks.interpolate_track_to_lags(
        track_df=track_df, reference=reference)

    result = {}
    file_cache: dict = {}
    premian = _resolved_premean_volume_sigma(sigma_3d=premean_volume_sigma_3d)

    for vk in var_keys:
        arr = np.full((composite_config.BOX_NLAT, composite_config.BOX_NLON,
                       composite_config.N_LAGS), np.nan)
        ds = data_registry.get(key=vk)
        if ds is None:
            result[vk] = arr
            continue

        for lag_idx, lag_h in enumerate(composite_config.LAG_HOURS):
            target_dt = (ref_time
                         + datetime.timedelta(hours=int(lag_h))).to_pydatetime()

            raw = data_registry.load_snapshot(
                source=ds, target_dt=target_dt, cache=file_cache)
            if raw is None:
                continue

            field = grid_utils.prepare_field(data=raw, source=ds)
            if field is None:
                continue

            patch = grid_utils.extract_storm_patch(
                field_2d=field, center_lat=center_lat, center_lon=center_lon)
            if patch is not None and patch.shape == (composite_config.BOX_NLAT,
                                                     composite_config.BOX_NLON):
                arr[:, :, lag_idx] = patch

        if np.isfinite(arr).any() and not all(s <= 0.0 for s in premian):
            smoothed = composite_io.smooth_composite_volume(
                field_3d=arr.astype(np.float64), sigma_3d=premian)
            if smoothed is not None:
                arr = smoothed.astype(np.float32)
        result[vk] = arr

    data_registry.close_cache(cache=file_cache)
    return result, lats, lons, winds, pres
