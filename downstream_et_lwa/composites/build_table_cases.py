"""Track-following recomputation of the Table-1 case statistics: per-storm,
per-lag box means of precipitation, latent-heating LWA source, and carrying
capacity, averaged over the table integration windows."""

from __future__ import annotations

import datetime
import logging
import multiprocessing
from pathlib import Path
from typing import Optional

import netCDF4 as nc
import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.composites.mpas_composites as mpas_composites
import downstream_et_lwa.data_registry as data_registry
import downstream_et_lwa.grid_utils as grid_utils
import downstream_et_lwa.tracks as tracks

_LOG = logging.getLogger(__name__)

ERA5_POOLS: dict[str, tuple[str, str] | None] = {
    "ERA5_WP": ("composite_2d_recurvature_WP.nc", "era5"),
    "ERA5_NA": ("composite_2d_recurvature_NA.nc", "era5"),
    "ERA5_WPNA": None,
    "ERA5_RW": ("composite_2d_recurvature_WP_rwcase.nc", "era5"),
    "ERA5_noRW": ("composite_2d_recurvature_WP_norwcase.nc", "era5"),
}

MPAS_POOLS = {
    "MPAS_current_WP": ("composite_2d_recurvature_mpas_current_WP.nc", "current", "WP"),
    "MPAS_current_NA": ("composite_2d_recurvature_mpas_current_NA.nc", "current", "NA"),
    "MPAS_future_WP": ("composite_2d_recurvature_mpas_future_WP.nc", "future", "WP"),
    "MPAS_future_NA": ("composite_2d_recurvature_mpas_future_NA.nc", "future", "NA"),
}

PRECIP_KEY = {"era5": "imerg_precip",
              "current": "mpas_current_mpas_precip",
              "future": "mpas_future_mpas_precip"}
LH_LWA_KEY = {"era5": "era5_lh_lwa",
              "current": "mpas_current_mpas_lh_lwa",
              "future": "mpas_future_mpas_lh_lwa"}
FC_KEY = {"era5": "era5_cc_Fc",
          "current": "mpas_current_mpas_cc_Fc",
          "future": "mpas_future_mpas_cc_Fc"}

WIN_PRECIP_H = (0.0, 120.0)
WIN_LH_H = (-24.0, 120.0)
WIN_FC_H = (0.0, 120.0)

HALF_PRECIP = 5.0
HALF_LH = 10.0

FC_DLAT_LO = 5.0
FC_DLAT_HI = 20.0
FC_DLON_LO = 0.0
FC_DLON_HI = 40.0


def _decode_storm_ids(*, arr: np.ndarray) -> list[str]:
    out = []
    for row in arr:
        s = b"".join(row).decode("utf-8", errors="ignore")
        out.append(s.replace("\x00", "").strip())
    return out


def _load_storm_meta(
        *,
        comp_path: Path,
        recurv_csv: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with nc.Dataset(comp_path) as d:
        sids = _decode_storm_ids(arr=d["storm_id"][:])
        track_lat = np.array(d["track_lat"][:])
        track_lon = np.array(d["track_lon"][:])
        track_pres = np.array(d["track_pres"][:])
        track_wind = np.array(d["track_wind"][:])
        recurv_lat = np.array(d["recurv_lat"][:])
        recurv_lon = np.array(d["recurv_lon"][:])
        season = np.array(d["season"][:])
        lag_h = np.array(d["lag_hours"][:])
    df = pd.DataFrame({
        "storm_id": sids, "season": season,
        "recurv_lat_cube": recurv_lat, "recurv_lon_cube": recurv_lon,
        "recurv_lon_cube_w": recurv_lon % 360.0,
    })
    df["_idx_in_cube"] = np.arange(len(df))
    j = df.merge(recurv_csv[["storm_id", "basin", "recurv_time"]],
                 on="storm_id", how="left")
    j["recurv_time"] = pd.to_datetime(j["recurv_time"])
    if j["recurv_time"].isna().any():
        miss = j[j["recurv_time"].isna()]["storm_id"].tolist()
        raise RuntimeError(
            f"Missing recurv_time for {len(miss)} storms in {comp_path.name}: {miss[:3]}")
    return j, track_lat, track_lon, track_pres, track_wind, lag_h


def _box_mean(
        *,
        field_2d_1deg: np.ndarray,
        lat_c: float,
        lon_c: float,
        halfwidth: float,
) -> float:
    if not (np.isfinite(lat_c) and np.isfinite(lon_c)):
        return float(np.nan)
    lat = grid_utils.GRID_1DEG_NH.lat
    lon = grid_utils.GRID_1DEG_NH.lon
    lat_mask = (lat >= lat_c - halfwidth) & (lat <= lat_c + halfwidth)
    if not lat_mask.any():
        return float(np.nan)
    lon_c_w = lon_c % 360.0
    lo = (lon_c_w - halfwidth) % 360.0
    hi = (lon_c_w + halfwidth) % 360.0
    if lo <= hi:
        lon_mask = (lon >= lo) & (lon <= hi)
    else:
        lon_mask = (lon >= lo) | (lon <= hi)
    if not lon_mask.any():
        return float(np.nan)
    sub = field_2d_1deg[np.ix_(lat_mask, lon_mask)]
    if not np.isfinite(sub).any():
        return float(np.nan)
    return float(np.nanmean(sub))


def _box_mean_ne_cosweighted(
        *,
        field_2d_1deg: np.ndarray,
        lat_c: float,
        lon_c: float,
        dlat_lo: float,
        dlat_hi: float,
        dlon_lo: float,
        dlon_hi: float,
) -> float:
    if not (np.isfinite(lat_c) and np.isfinite(lon_c)):
        return float(np.nan)
    lat = grid_utils.GRID_1DEG_NH.lat
    lon = grid_utils.GRID_1DEG_NH.lon
    lat_mask = (lat >= lat_c + dlat_lo) & (lat <= lat_c + dlat_hi)
    if not lat_mask.any():
        return float(np.nan)
    lon_c_w = lon_c % 360.0
    lo = (lon_c_w + dlon_lo) % 360.0
    hi = (lon_c_w + dlon_hi) % 360.0
    if lo <= hi:
        lon_mask = (lon >= lo) & (lon <= hi)
    else:
        lon_mask = (lon >= lo) | (lon <= hi)
    if not lon_mask.any():
        return float(np.nan)
    sub = field_2d_1deg[np.ix_(lat_mask, lon_mask)]
    valid = np.isfinite(sub)
    if not valid.any():
        return float(np.nan)
    sub_lat = lat[lat_mask]
    w = np.broadcast_to(np.cos(np.deg2rad(sub_lat))[:, None], sub.shape)
    return float(np.sum(sub[valid] * w[valid]) / np.sum(w[valid]))


def _ensure_registry(*, data_config: dict[str, str],
                     register_mpas: tuple | None) -> None:
    data_registry.register_all(data_config=data_config)
    if register_mpas:
        scenario = register_mpas[0]
        mpas_composites.register_mpas_sources(
            scenario=scenario, data_config=data_config)
        mpas_composites.register_mpas_map_diagnostics(
            scenario=scenario, data_config=data_config)


def _process_storm_chunk_fc(args: tuple) -> list[float]:
    (chunk, var_key, win_h, dlat_lo, dlat_hi, dlon_lo, dlon_hi, lag_h,
     register_mpas, data_config) = args
    _ensure_registry(data_config=data_config, register_mpas=register_mpas)
    ds = data_registry.get(key=var_key)
    if ds is None:
        raise RuntimeError(f"No registered DataSource for key='{var_key}'")
    in_window = (lag_h >= win_h[0]) & (lag_h <= win_h[1])
    file_cache: dict = {}
    out = []
    for rec in chunk:
        ref_time = rec["ref_time"]
        rlat = float(rec["recurv_lat"])
        rlon = float(rec["recurv_lon"])
        per_lag = []
        for lag_idx in np.where(in_window)[0]:
            lh = float(lag_h[lag_idx])
            target = ref_time + datetime.timedelta(hours=lh)
            raw = data_registry.load_snapshot(
                source=ds, target_dt=target, cache=file_cache)
            if raw is None:
                continue
            field = grid_utils.prepare_field(data=raw, source=ds)
            if field is None:
                continue
            v = _box_mean_ne_cosweighted(
                field_2d_1deg=field, lat_c=rlat, lon_c=rlon,
                dlat_lo=dlat_lo, dlat_hi=dlat_hi,
                dlon_lo=dlon_lo, dlon_hi=dlon_hi)
            if np.isfinite(v):
                per_lag.append(v)
        out.append(float(np.nanmean(per_lag)) if per_lag else float(np.nan))
    data_registry.close_cache(cache=file_cache)
    return out


def _per_storm_fc_recurv(
        *,
        records: list,
        lag_h: np.ndarray,
        var_key: str,
        win_h: tuple[float, float],
        dlat_lo: float,
        dlat_hi: float,
        dlon_lo: float,
        dlon_hi: float,
        label: str,
        data_config: dict[str, str],
        n_workers: int = 1,
        register_mpas: tuple | None = None,
) -> np.ndarray:
    if not records:
        return np.array([], dtype=np.float64)
    if n_workers <= 1 or len(records) <= 1:
        return np.array(_process_storm_chunk_fc(
            (records, var_key, win_h, dlat_lo, dlat_hi, dlon_lo, dlon_hi,
             lag_h, register_mpas, data_config)))
    chunks = np.array_split(records, n_workers)
    args_list = [(list(c), var_key, win_h, dlat_lo, dlat_hi, dlon_lo,
                  dlon_hi, lag_h, register_mpas, data_config) for c in chunks]
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(_process_storm_chunk_fc, args_list)
    flat: list[float] = []
    for r in results:
        flat.extend(r)
    _LOG.info("[%s] done (%d storms, %d workers)", label, len(records), n_workers)
    return np.array(flat, dtype=np.float64)


def _process_storm_chunk_era5(args: tuple) -> list[float]:
    (chunk, var_key, win_h, halfwidth, lag_h, register_mpas, data_config) = args
    _ensure_registry(data_config=data_config, register_mpas=register_mpas)
    ds = data_registry.get(key=var_key)
    if ds is None:
        raise RuntimeError(f"No registered DataSource for key='{var_key}'")
    in_window = (lag_h >= win_h[0]) & (lag_h <= win_h[1])
    file_cache: dict = {}
    out = []
    for rec in chunk:
        ref_time = rec["ref_time"]
        track_lat_i = rec["track_lat"]
        track_lon_i = rec["track_lon"]
        per_lag = []
        for lag_idx in np.where(in_window)[0]:
            lh = float(lag_h[lag_idx])
            target = ref_time + datetime.timedelta(hours=lh)
            raw = data_registry.load_snapshot(
                source=ds, target_dt=target, cache=file_cache)
            if raw is None:
                continue
            field = grid_utils.prepare_field(data=raw, source=ds)
            if field is None:
                continue
            v = _box_mean(field_2d_1deg=field,
                          lat_c=float(track_lat_i[lag_idx]),
                          lon_c=float(track_lon_i[lag_idx]),
                          halfwidth=halfwidth)
            if np.isfinite(v):
                per_lag.append(v)
        out.append(float(np.nanmean(per_lag)) if per_lag else float(np.nan))
    data_registry.close_cache(cache=file_cache)
    return out


def _per_storm_window_mean(
        *,
        storm_meta: pd.DataFrame,
        track_lat: np.ndarray,
        track_lon: np.ndarray,
        lag_h: np.ndarray,
        var_key: str,
        win_h: tuple[float, float],
        halfwidth: float,
        label: str,
        data_config: dict[str, str],
        n_workers: int = 1,
        register_mpas: tuple | None = None,
) -> np.ndarray:
    n_storms = len(storm_meta)
    records = []
    for i, (_, st) in enumerate(storm_meta.iterrows()):
        records.append({
            "ref_time": pd.Timestamp(st["recurv_time"]).to_pydatetime(),
            "track_lat": np.array(track_lat[i, :], dtype=np.float64),
            "track_lon": np.array(track_lon[i, :], dtype=np.float64),
        })
    if n_workers <= 1 or n_storms <= 1:
        return np.array(_process_storm_chunk_era5(
            (records, var_key, win_h, halfwidth, lag_h, register_mpas,
             data_config)))

    chunks = np.array_split(np.array(records, dtype=object), n_workers)
    args_list = [(list(c), var_key, win_h, halfwidth, lag_h, register_mpas,
                  data_config) for c in chunks]
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(_process_storm_chunk_era5, args_list)
    flat: list[float] = []
    for r in results:
        flat.extend(r)
    _LOG.info("[%s] done (%d storms, %d workers)", label, n_storms, n_workers)
    return np.array(flat, dtype=np.float64)


def _pool_mean_se(*, per_storm: np.ndarray) -> tuple[float, float, int]:
    valid = per_storm[np.isfinite(per_storm)]
    n = valid.size
    if n == 0:
        return (float(np.nan), float(np.nan), 0)
    mean = float(np.nanmean(valid))
    se = float(np.nanstd(valid, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return mean, se, n


def _build_mpas_storms_for_basin(
        *,
        scenario: str,
        basin: str,
        et_track_directory: Path | None,
        all_tracks_root: Path | None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    storms_df, full_tracks = mpas_composites.build_mpas_tracks(
        scenario=scenario,
        et_track_dir=et_track_directory,
        all_tracks_root=all_tracks_root)
    _LOG.info("build_mpas_tracks returned %d rows; columns=%s",
              len(storms_df), list(storms_df.columns))
    if len(storms_df) == 0:
        return storms_df, full_tracks
    sub = storms_df[storms_df["basin"] == basin].reset_index(drop=True)
    return sub, full_tracks


def _build_mpas_records(
        *,
        storms_df: pd.DataFrame,
        full_tracks: dict,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    n = len(storms_df)
    slp_min = np.full(n, np.nan, dtype=np.float64)
    vmax = np.full(n, np.nan, dtype=np.float64)
    rec_lat = np.full(n, np.nan, dtype=np.float64)
    records: list = []
    for i, (_, st) in enumerate(storms_df.iterrows()):
        rec_lat[i] = float(st["recurv_lat"])
        track_df = full_tracks.get(st["storm_id"])
        if track_df is None:
            records.append(None)
            continue
        try:
            lats, lons, winds, pres = tracks.interpolate_track_to_lags(
                track_df=track_df, reference="recurvature")
        except Exception:
            _LOG.exception("interpolate_track_to_lags failed for %s",
                           st["storm_id"])
            records.append(None)
            continue
        slp_min[i] = float(np.nanmin(pres)) if np.any(np.isfinite(pres)) else np.nan
        vmax[i] = float(np.nanmax(winds)) if np.any(np.isfinite(winds)) else np.nan
        records.append({
            "ref_time": pd.Timestamp(st["recurv_time"]).to_pydatetime(),
            "track_lat": np.array(lats, dtype=np.float64),
            "track_lon": np.array(lons, dtype=np.float64),
        })
    return records, slp_min, vmax, rec_lat


def _per_storm_window_mean_mpas(
        *,
        records: list,
        lag_h: np.ndarray,
        scenario: str,
        var_key: str,
        win_h: tuple[float, float],
        halfwidth: float,
        label: str,
        data_config: dict[str, str],
        n_workers: int = 1,
) -> np.ndarray:
    valid_idx = [i for i, r in enumerate(records) if r is not None]
    valid_records = [records[i] for i in valid_idx]
    if not valid_records:
        return np.full(len(records), np.nan, dtype=np.float64)

    if n_workers <= 1:
        per_valid = _process_storm_chunk_era5(
            (valid_records, var_key, win_h, halfwidth, lag_h,
             (scenario, True), data_config))
    else:
        chunks = np.array_split(valid_records, n_workers)
        args_list = [(list(c), var_key, win_h, halfwidth, lag_h,
                      (scenario, True), data_config) for c in chunks]
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.map(_process_storm_chunk_era5, args_list)
        per_valid = []
        for r in results:
            per_valid.extend(r)
        _LOG.info("[%s] done (%d storms, %d workers)",
                  label, len(valid_records), n_workers)
    out = np.full(len(records), np.nan, dtype=np.float64)
    for i, v in zip(valid_idx, per_valid):
        out[i] = v
    return out


def _compute_pool_mpas(
        *,
        label: str,
        scenario: str,
        basin: str,
        data_config: dict[str, str],
        et_track_directory: Path | None,
        all_tracks_root: Path | None,
        n_workers: int = 1,
) -> dict:
    _LOG.info("=== %s (re-parsing ettracks for %s/%s) ===", label, scenario, basin)
    mpas_composites.register_mpas_sources(
        scenario=scenario, data_config=data_config)
    mpas_composites.register_mpas_map_diagnostics(
        scenario=scenario, data_config=data_config)
    sub, full_tracks = _build_mpas_storms_for_basin(
        scenario=scenario, basin=basin,
        et_track_directory=et_track_directory,
        all_tracks_root=all_tracks_root)
    _LOG.info("storms in pool: %d", len(sub))
    if len(sub) == 0:
        return dict(label=label, n_storms_total=0,
                    P_mean=np.nan, P_se=np.nan, P_n=0,
                    A_mean=np.nan, A_se=np.nan, A_n=0,
                    SLPmin=np.nan, Vmax=np.nan, recurv_lat=np.nan)
    lag_h = composite_config.LAG_HOURS.astype(float)
    records, slp_min, vmax, rec_lat = _build_mpas_records(
        storms_df=sub, full_tracks=full_tracks)
    p_key = PRECIP_KEY[scenario]
    lh_key = LH_LWA_KEY[scenario]
    P_per_storm = _per_storm_window_mean_mpas(
        records=records, lag_h=lag_h, scenario=scenario, var_key=p_key,
        win_h=WIN_PRECIP_H, halfwidth=HALF_PRECIP,
        label=f"{label}/P5deg", data_config=data_config, n_workers=n_workers)
    A_per_storm = _per_storm_window_mean_mpas(
        records=records, lag_h=lag_h, scenario=scenario, var_key=lh_key,
        win_h=WIN_LH_H, halfwidth=HALF_LH,
        label=f"{label}/dot_A_L10deg", data_config=data_config,
        n_workers=n_workers)

    fc_records = []
    for i, (_, st) in enumerate(sub.iterrows()):
        if records[i] is None:
            continue
        fc_records.append({
            "ref_time": pd.Timestamp(st["recurv_time"]).to_pydatetime(),
            "recurv_lat": float(st["recurv_lat"]),
            "recurv_lon": float(st["recurv_lon"]) % 360.0,
        })
    Fc_per_storm = _per_storm_fc_recurv(
        records=fc_records, lag_h=lag_h, var_key=FC_KEY[scenario],
        win_h=WIN_FC_H, dlat_lo=FC_DLAT_LO, dlat_hi=FC_DLAT_HI,
        dlon_lo=FC_DLON_LO, dlon_hi=FC_DLON_HI,
        label=f"{label}/Fc_dwg", data_config=data_config,
        n_workers=n_workers, register_mpas=(scenario, True))

    P_mean, P_se, P_n = _pool_mean_se(per_storm=P_per_storm)
    A_mean, A_se, A_n = _pool_mean_se(per_storm=A_per_storm * 86400.0)
    Fc_mean, Fc_se, Fc_n = _pool_mean_se(per_storm=Fc_per_storm)
    slp_hPa = float(np.nanmean(slp_min)) / 100.0
    return dict(
        label=label, n_storms_total=len(sub),
        P_mean=P_mean, P_se=P_se, P_n=P_n,
        A_mean=A_mean, A_se=A_se, A_n=A_n,
        Fc_mean=Fc_mean, Fc_se=Fc_se, Fc_n=Fc_n,
        SLPmin=slp_hPa,
        Vmax=float(np.nanmean(vmax)),
        recurv_lat=float(np.nanmean(rec_lat)),
    )


def _compute_pool(
        *,
        label: str,
        comp_path: Path,
        source: str,
        recurv_csv: pd.DataFrame,
        data_config: dict[str, str],
        n_workers: int = 1,
) -> dict:
    _LOG.info("=== %s (%s) ===", label, comp_path.name)
    meta, tlat, tlon, tpres, twind, lag_h = _load_storm_meta(
        comp_path=comp_path, recurv_csv=recurv_csv)
    _LOG.info("storms in cube: %d", len(meta))

    p_key = PRECIP_KEY[source]
    lh_key = LH_LWA_KEY[source]

    in_full = np.ones_like(lag_h, dtype=bool)
    slp_min_per_storm = np.nanmin(np.where(in_full, tpres, np.nan), axis=1)
    vmax_per_storm = np.nanmax(np.where(in_full, twind, np.nan), axis=1)

    P_per_storm = _per_storm_window_mean(
        storm_meta=meta, track_lat=tlat, track_lon=tlon, lag_h=lag_h,
        var_key=p_key, win_h=WIN_PRECIP_H, halfwidth=HALF_PRECIP,
        label=f"{label}/P5deg", data_config=data_config, n_workers=n_workers)
    A_per_storm = _per_storm_window_mean(
        storm_meta=meta, track_lat=tlat, track_lon=tlon, lag_h=lag_h,
        var_key=lh_key, win_h=WIN_LH_H, halfwidth=HALF_LH,
        label=f"{label}/dot_A_L10deg", data_config=data_config,
        n_workers=n_workers)

    fc_records = []
    for _, st in meta.iterrows():
        fc_records.append({
            "ref_time": pd.Timestamp(st["recurv_time"]).to_pydatetime(),
            "recurv_lat": float(st["recurv_lat_cube"]),
            "recurv_lon": float(st["recurv_lon_cube_w"]),
        })
    Fc_per_storm = _per_storm_fc_recurv(
        records=fc_records, lag_h=lag_h, var_key=FC_KEY[source],
        win_h=WIN_FC_H, dlat_lo=FC_DLAT_LO, dlat_hi=FC_DLAT_HI,
        dlon_lo=FC_DLON_LO, dlon_hi=FC_DLON_HI,
        label=f"{label}/Fc_dwg", data_config=data_config, n_workers=n_workers)

    P_mean, P_se, P_n = _pool_mean_se(per_storm=P_per_storm)
    A_mean, A_se, A_n = _pool_mean_se(per_storm=A_per_storm * 86400.0)
    Fc_mean, Fc_se, Fc_n = _pool_mean_se(per_storm=Fc_per_storm)

    SLPmin_mean = float(np.nanmean(slp_min_per_storm))
    Vmax_mean = float(np.nanmean(vmax_per_storm))
    Phi_mean = float(np.nanmean(meta["recurv_lat_cube"].values))

    return dict(
        label=label, n_storms_total=len(meta),
        P_mean=P_mean, P_se=P_se, P_n=P_n,
        A_mean=A_mean, A_se=A_se, A_n=A_n,
        Fc_mean=Fc_mean, Fc_se=Fc_se, Fc_n=Fc_n,
        SLPmin=SLPmin_mean, Vmax=Vmax_mean, recurv_lat=Phi_mean,
    )


def _pool_wpna(*, rec_wp: dict, rec_na: dict) -> dict:
    nP_wp, nP_na = rec_wp["P_n"], rec_na["P_n"]
    nA_wp, nA_na = rec_wp["A_n"], rec_na["A_n"]
    n_total = rec_wp["n_storms_total"] + rec_na["n_storms_total"]

    def _w(a: float, na: int, b: float, nb: int) -> float:
        if na + nb == 0:
            return float("nan")
        return (a * na + b * nb) / (na + nb)

    nF_wp, nF_na = rec_wp.get("Fc_n", 0), rec_na.get("Fc_n", 0)
    return dict(
        label="ERA5_WPNA",
        n_storms_total=n_total,
        P_mean=_w(rec_wp["P_mean"], nP_wp, rec_na["P_mean"], nP_na),
        P_se=float("nan"),
        P_n=nP_wp + nP_na,
        A_mean=_w(rec_wp["A_mean"], nA_wp, rec_na["A_mean"], nA_na),
        A_se=float("nan"),
        A_n=nA_wp + nA_na,
        Fc_mean=_w(rec_wp.get("Fc_mean", float("nan")), nF_wp,
                   rec_na.get("Fc_mean", float("nan")), nF_na),
        Fc_se=float("nan"),
        Fc_n=nF_wp + nF_na,
        SLPmin=_w(rec_wp["SLPmin"], rec_wp["n_storms_total"],
                  rec_na["SLPmin"], rec_na["n_storms_total"]),
        Vmax=_w(rec_wp["Vmax"], rec_wp["n_storms_total"],
                rec_na["Vmax"], rec_na["n_storms_total"]),
        recurv_lat=_w(rec_wp["recurv_lat"], rec_wp["n_storms_total"],
                      rec_na["recurv_lat"], rec_na["n_storms_total"]),
    )


def _format_table(*, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    cols = ["label", "n_storms_total", "SLPmin", "Vmax", "recurv_lat",
            "P_mean", "P_se", "P_n",
            "A_mean", "A_se", "A_n"]
    if "Fc_mean" in df.columns:
        cols += ["Fc_mean", "Fc_se", "Fc_n"]
    df = df[cols]
    return df


def main(
        data_config: Annotated[Path, typer.Option(
            help="JSON file mapping data keys to directories")],
        composites_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_recurvature_*.nc cubes")],
        tracks_csv: Annotated[Path, typer.Option(
            help="recurving_nh_tracks.csv with recurv_time per storm")],
        output_directory: Annotated[Path, typer.Option()],
        basins: Annotated[Optional[list[str]], typer.Option(
            help="ERA5 pool labels")] = None,
        mpas_pools: Annotated[Optional[list[str]], typer.Option(
            help="MPAS pool labels (e.g. MPAS_current_WP MPAS_future_NA)")] = None,
        et_track_directory: Annotated[Optional[Path], typer.Option(
            help="Directory with traj_et_mpas_avg_*_{scenario}.dat")] = None,
        all_tracks_root: Annotated[Optional[Path], typer.Option(
            help="Directory walked for ettrack_{scenario}_*.txt")] = None,
        workers: Annotated[int, typer.Option(
            help="Parallel processes for per-storm extraction")] = max(
                1, multiprocessing.cpu_count() // 2),
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())

    cfg = data_registry.load_data_config(path=data_config)
    data_registry.register_all(data_config=cfg)

    basin_labels = basins or ["ERA5_WP", "ERA5_NA", "ERA5_WPNA",
                              "ERA5_RW", "ERA5_noRW"]
    mpas_labels = mpas_pools or []

    output_directory.mkdir(parents=True, exist_ok=True)
    recurv_csv = pd.read_csv(tracks_csv)

    rows: list[dict] = []
    rec_wp = rec_na = None
    for label in basin_labels:
        if label == "ERA5_WPNA":
            continue
        spec = ERA5_POOLS.get(label)
        if spec is None:
            print(f"  (skipping {label}: no spec)")
            continue
        comp_file, source = spec
        comp_path = composites_directory / comp_file
        rec = _compute_pool(
            label=label, comp_path=comp_path, source=source,
            recurv_csv=recurv_csv, data_config=cfg, n_workers=workers)
        rows.append(rec)
        if label == "ERA5_WP":
            rec_wp = rec
        elif label == "ERA5_NA":
            rec_na = rec

    if "ERA5_WPNA" in basin_labels and rec_wp is not None and rec_na is not None:
        rows.append(_pool_wpna(rec_wp=rec_wp, rec_na=rec_na))

    mpas_recs = {}
    for label in mpas_labels:
        if label not in MPAS_POOLS:
            print(f"  (skipping {label}: no MPAS spec)")
            continue
        _, scenario, basin = MPAS_POOLS[label]
        rec = _compute_pool_mpas(
            label=label, scenario=scenario, basin=basin, data_config=cfg,
            et_track_directory=et_track_directory,
            all_tracks_root=all_tracks_root, n_workers=workers)
        rows.append(rec)
        mpas_recs[label] = rec

    for scen in ("current", "future"):
        wp = mpas_recs.get(f"MPAS_{scen}_WP")
        na = mpas_recs.get(f"MPAS_{scen}_NA")
        if wp is not None and na is not None:
            pooled = _pool_wpna(rec_wp=wp, rec_na=na)
            pooled["label"] = f"MPAS_{scen}_WPNA"
            rows.append(pooled)

    df = _format_table(rows=rows)
    out_path = output_directory / "table_cases_track_following.csv"
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nWrote {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    typer.run(main)
