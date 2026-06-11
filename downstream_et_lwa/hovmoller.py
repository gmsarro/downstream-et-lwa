"""Hovmoller (longitude-time) composites and the Quinting & Jones (2016) RWP
envelope (Hilbert transform of wavenumber-filtered 250-hPa meridional wind)."""

from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.data_registry as data_registry
import downstream_et_lwa.grid_utils as grid_utils

_LOG = logging.getLogger(__name__)


def compute_rwp_envelope(
        *,
        v_field: np.ndarray,
        kmin: int | None = None,
        kmax: int | None = None,
) -> np.ndarray:
    kmin = kmin or composite_config.RWP_KMIN
    kmax = kmax or composite_config.RWP_KMAX
    nlon = v_field.shape[-1]
    envelope = np.zeros_like(v_field)

    for j in range(v_field.shape[0]):
        row = v_field[j, :]
        if np.all(np.isnan(row)):
            continue
        row_clean = np.where(np.isfinite(row), row, 0.0)

        fft_coeffs = np.fft.fft(row_clean)

        filtered = np.zeros_like(fft_coeffs)
        for k in range(kmin, min(kmax + 1, nlon // 2)):
            filtered[k] = fft_coeffs[k]

        analytic = np.fft.ifft(filtered * 2)
        envelope[j, :] = np.abs(analytic)

    return envelope


def extract_hovmoller_strip(
        *,
        field_2d: np.ndarray,
        lat_min: float | None = None,
        lat_max: float | None = None,
) -> np.ndarray:
    lat_min = lat_min or composite_config.HOV_LAT_MIN
    lat_max = lat_max or composite_config.HOV_LAT_MAX

    j_min = max(int(round(lat_min)), 0)
    j_max = min(int(round(lat_max)) + 1, field_2d.shape[0])

    strip = field_2d[j_min:j_max, :]
    weights = composite_config.COSPHI[j_min:j_max, np.newaxis]

    valid = np.isfinite(strip)
    w = np.where(valid, weights, 0.0)
    w_sum = np.sum(w, axis=0)
    w_sum[w_sum == 0] = np.nan
    return np.nansum(strip * w, axis=0) / w_sum


def _load_field_for_hovmoller(
        *,
        var_key: str,
        target_dt: datetime.datetime,
        file_cache: dict,
) -> np.ndarray | None:
    ds = data_registry.get(key=var_key)
    if ds is None:
        return None
    raw = data_registry.load_snapshot(source=ds, target_dt=target_dt, cache=file_cache)
    if raw is None:
        return None
    return grid_utils.prepare_field(data=raw, source=ds)


def build_hovmoller_3d_single_storm(
        *,
        storms_df: pd.DataFrame,
        full_tracks: dict[str, pd.DataFrame],
        storm_id: str,
        var_keys: list[str] | None = None,
        reference: str = "recurvature",
        compute_envelope: bool = True,
) -> dict[str, np.ndarray]:
    if var_keys is None:
        var_keys = ["era5_v_250hPa", "era5_lwa", "era5_rwb_awb", "era5_rwb_cwb"]

    storm = storms_df[storms_df["storm_id"] == storm_id].iloc[0]
    ref_time = pd.Timestamp(
        storm["recurv_time"] if reference == "recurvature" else storm["et_time"])

    shape3d = (composite_config.NLAT, composite_config.NLON, composite_config.N_LAGS)
    hovs = {vk: np.full(shape3d, np.nan) for vk in var_keys}
    if compute_envelope:
        hovs["rwp_envelope"] = np.full(shape3d, np.nan)

    file_cache: dict = {}
    for lag_idx, lag_h in enumerate(composite_config.LAG_HOURS):
        target_dt = (ref_time + datetime.timedelta(hours=int(lag_h))).to_pydatetime()
        for vk in var_keys:
            field = _load_field_for_hovmoller(
                var_key=vk, target_dt=target_dt, file_cache=file_cache)
            if field is None:
                continue
            hovs[vk][:, :, lag_idx] = field
            if compute_envelope and vk == "era5_v_250hPa":
                hovs["rwp_envelope"][:, :, lag_idx] = compute_rwp_envelope(v_field=field)

    data_registry.close_cache(cache=file_cache)
    return hovs


def build_hovmoller_3d_composite(
        *,
        storms_df: pd.DataFrame,
        full_tracks: dict[str, pd.DataFrame],
        var_keys: list[str] | None = None,
        basins: list[str] | None = None,
        reference: str = "recurvature",
        compute_envelope: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    basins = basins or ["WP", "NA", "EP"]
    if var_keys is None:
        var_keys = ["era5_v_250hPa", "era5_lwa", "era5_rwb_awb", "era5_rwb_cwb"]

    all_keys = list(var_keys)
    if compute_envelope:
        all_keys.append("rwp_envelope")

    shape3d = (composite_config.NLAT, composite_config.NLON, composite_config.N_LAGS)
    hov_sum = {b: {k: np.zeros(shape3d) for k in all_keys} for b in basins}
    counts = {b: {k: np.zeros(composite_config.N_LAGS, dtype=int) for k in all_keys}
              for b in basins}

    storm_list = storms_df[storms_df["basin"].isin(basins)]
    file_cache: dict = {}

    for idx, (_, storm) in enumerate(storm_list.iterrows()):
        basin = storm["basin"]
        ref_time = pd.Timestamp(
            storm["recurv_time"] if reference == "recurvature" else storm["et_time"])
        if pd.isna(ref_time):
            continue

        for lag_idx, lag_h in enumerate(composite_config.LAG_HOURS):
            target_dt = (ref_time + datetime.timedelta(hours=int(lag_h))).to_pydatetime()
            for vk in var_keys:
                field = _load_field_for_hovmoller(
                    var_key=vk, target_dt=target_dt, file_cache=file_cache)
                if field is None:
                    continue
                hov_sum[basin][vk][:, :, lag_idx] += np.nan_to_num(field)
                counts[basin][vk][lag_idx] += 1

                if compute_envelope and vk == "era5_v_250hPa":
                    env = compute_rwp_envelope(v_field=field)
                    hov_sum[basin]["rwp_envelope"][:, :, lag_idx] += np.nan_to_num(env)
                    counts[basin]["rwp_envelope"][lag_idx] += 1

        if (idx + 1) % 20 == 0:
            _LOG.info("Hovmoller 3D [%d/%d]", idx + 1, len(storm_list))

    data_registry.close_cache(cache=file_cache)

    hov_mean: dict[str, dict[str, np.ndarray]] = {}
    for b in basins:
        hov_mean[b] = {}
        for k in all_keys:
            n = counts[b][k].copy().astype(float)
            n[n == 0] = np.nan
            hov_mean[b][k] = hov_sum[b][k] / n[np.newaxis, np.newaxis, :]

    return hov_mean, counts


def slice_hovmoller_3d(*, hov_3d: np.ndarray, lat_deg: float) -> np.ndarray:
    j = int(round(np.clip(lat_deg, 0, 90)))
    j = min(j, hov_3d.shape[0] - 1)
    return hov_3d[j, :, :]


def average_hovmoller_3d(
        *,
        hov_3d: np.ndarray,
        lat_min: float | None = None,
        lat_max: float | None = None,
) -> np.ndarray:
    lat_min = lat_min or composite_config.HOV_LAT_MIN
    lat_max = lat_max or composite_config.HOV_LAT_MAX

    j_min = max(int(round(lat_min)), 0)
    j_max = min(int(round(lat_max)) + 1, hov_3d.shape[0])

    strip = hov_3d[j_min:j_max, :, :]
    weights = composite_config.COSPHI[j_min:j_max, np.newaxis, np.newaxis]

    valid = np.isfinite(strip)
    w = np.where(valid, weights, 0.0)
    w_sum = np.sum(w, axis=0)
    w_sum[w_sum == 0] = np.nan
    return np.nansum(strip * w, axis=0) / w_sum


def time_integrate_hovmoller(*, hov_2d: np.ndarray) -> np.ndarray:
    nlon, nlags = hov_2d.shape
    ref_idx = int(np.argmin(np.abs(composite_config.LAG_HOURS)))
    dt = composite_config.DT_SEC

    result = np.full_like(hov_2d, np.nan)
    result[:, ref_idx] = 0.0

    for t in range(ref_idx + 1, nlags):
        prev = result[:, t - 1]
        c_rate = hov_2d[:, t]
        p_rate = hov_2d[:, t - 1]
        ok = np.isfinite(prev) & np.isfinite(c_rate) & np.isfinite(p_rate)
        result[:, t] = np.where(ok, prev + 0.5 * (p_rate + c_rate) * dt, np.nan)

    for t in range(ref_idx - 1, -1, -1):
        nxt = result[:, t + 1]
        c_rate = hov_2d[:, t]
        n_rate = hov_2d[:, t + 1]
        ok = np.isfinite(nxt) & np.isfinite(c_rate) & np.isfinite(n_rate)
        result[:, t] = np.where(ok, nxt - 0.5 * (c_rate + n_rate) * dt, np.nan)

    return result


def build_hovmoller_single_storm(
        *,
        storms_df: pd.DataFrame,
        full_tracks: dict[str, pd.DataFrame],
        storm_id: str,
        var_keys: list[str] | None = None,
        reference: str = "recurvature",
        compute_envelope: bool = True,
) -> dict[str, np.ndarray]:
    hovs_3d = build_hovmoller_3d_single_storm(
        storms_df=storms_df, full_tracks=full_tracks, storm_id=storm_id,
        var_keys=var_keys, reference=reference,
        compute_envelope=compute_envelope)

    hovs_2d = {}
    for key, arr in hovs_3d.items():
        hovs_2d[key] = average_hovmoller_3d(hov_3d=arr)
    return hovs_2d


def build_hovmoller_composite(
        *,
        storms_df: pd.DataFrame,
        full_tracks: dict[str, pd.DataFrame],
        var_keys: list[str] | None = None,
        basins: list[str] | None = None,
        reference: str = "recurvature",
        compute_envelope: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    hov3d_mean, counts = build_hovmoller_3d_composite(
        storms_df=storms_df, full_tracks=full_tracks,
        var_keys=var_keys, basins=basins,
        reference=reference, compute_envelope=compute_envelope)

    hov2d_mean: dict[str, dict[str, np.ndarray]] = {}
    for b in hov3d_mean:
        hov2d_mean[b] = {}
        for k, arr in hov3d_mean[b].items():
            hov2d_mean[b][k] = average_hovmoller_3d(hov_3d=arr)

    return hov2d_mean, counts
