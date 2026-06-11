"""IBTrACS processing: recurvature detection (Burroughs & Brand 1973), ET onset,
track database build/save/load, and interpolation onto the composite lag grid."""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
import pandas as pd

import downstream_et_lwa.composite_config as composite_config

_LOG = logging.getLogger(__name__)

_TROPICAL_NATURES = {"TS", "HU", "TY", "TC", "ST", "NR", "DS", "SS", "TD"}


def _decode_char_array(*, arr: Any) -> str:
    if hasattr(arr, "tobytes"):
        return arr.tobytes().decode("utf-8", errors="replace").strip("\x00").strip()
    return str(arr)


def _decode_char_matrix(*, mat: Any) -> list[str]:
    return [_decode_char_array(arr=mat[i]) for i in range(mat.shape[0])]


def load_ibtracs(*, path: Path) -> dict[str, Any]:
    with nc.Dataset(str(path), "r") as ds:
        n_storms = ds.dimensions["storm"].size
        n_times = ds.dimensions["date_time"].size

        sid = _decode_char_matrix(mat=ds["sid"][:])
        name = _decode_char_matrix(mat=ds["name"][:])

        basin_raw = ds["basin"][:]
        b0 = np.char.decode(basin_raw[:, :, 0].astype("S1"), "ascii", errors="replace")
        b1 = np.char.decode(basin_raw[:, :, 1].astype("S1"), "ascii", errors="replace")
        basin = np.char.strip(np.char.add(b0, b1))

        season = np.array(ds["season"][:])
        lat = np.array(ds["lat"][:], dtype=np.float64)
        lon = np.array(ds["lon"][:], dtype=np.float64)

        time_var = ds["time"]
        time_units = time_var.units
        time_cal = getattr(time_var, "calendar", "standard")
        time_num = np.array(time_var[:])

        nature_raw = ds["nature"][:]
        n0 = np.char.decode(nature_raw[:, :, 0].astype("S1"), "ascii", errors="replace")
        n1 = np.char.decode(nature_raw[:, :, 1].astype("S1"), "ascii", errors="replace")
        nature = np.char.strip(np.char.add(n0, n1))

        wmo_wind = np.array(ds["wmo_wind"][:], dtype=np.float64)
        wmo_pres = np.array(ds["wmo_pres"][:], dtype=np.float64)
        numobs = np.array(ds["numobs"][:])

        track_type = _decode_char_matrix(mat=ds["track_type"][:])

    lat[lat < -900] = np.nan
    lon[lon < -900] = np.nan
    wmo_wind[wmo_wind < 0] = np.nan
    wmo_pres[wmo_pres < 0] = np.nan

    return {
        "sid": sid,
        "name": name,
        "basin": basin,
        "season": season,
        "lat": lat,
        "lon": lon,
        "time_num": time_num,
        "time_units": time_units,
        "time_cal": time_cal,
        "nature": nature,
        "wmo_wind": wmo_wind,
        "wmo_pres": wmo_pres,
        "numobs": numobs,
        "track_type": track_type,
        "n_storms": n_storms,
        "n_times": n_times,
    }


def _num2date(*, time_num: np.ndarray, units: str, calendar: str) -> list[datetime.datetime | None]:
    dates = nc.num2date(time_num, units, calendar)
    if hasattr(dates, "__iter__"):
        out: list[datetime.datetime | None] = []
        for d in dates:
            if d is None or (hasattr(d, "year") and d.year < 1800):
                out.append(None)
            else:
                try:
                    out.append(datetime.datetime(d.year, d.month, d.day, d.hour, d.minute))
                except Exception:
                    _LOG.exception("Failed converting time value %s", d)
                    out.append(None)
        return out
    return [dates]


def detect_recurvature(
        *,
        lat_track: np.ndarray,
        lon_track: np.ndarray,
        nature_track: np.ndarray,
        min_lat: float | None = None,
) -> int | None:
    min_lat = min_lat or composite_config.MIN_RECURV_LAT
    n = len(lat_track)

    def _dlon(i1: int, i2: int) -> float:
        d = lon_track[i2] - lon_track[i1]
        if abs(d) > 180:
            d -= np.sign(d) * 360
        return float(d)

    for i in range(1, n - 2):
        if any(np.isnan(lat_track[j]) or np.isnan(lon_track[j])
               for j in range(i - 1, min(i + 3, n))):
            continue
        if lat_track[i] < 0 or lat_track[i] < min_lat:
            continue

        was_tropical = any(
            nature_track[j] in _TROPICAL_NATURES
            for j in range(max(0, i - 4), i + 1)
            if nature_track[j] not in ("", " ")
        )
        if not was_tropical:
            continue

        dlon_before = _dlon(i - 1, i)
        dlon_after1 = _dlon(i, i + 1)
        dlon_after2 = (_dlon(i + 1, i + 2)
                       if i + 2 < n and not np.isnan(lon_track[i + 2]) else dlon_after1)

        if dlon_before <= 0 and dlon_after1 >= 0 and dlon_after2 >= 0:
            return i

    return None


def find_et_onset(*, nature_track: np.ndarray) -> int | None:
    for i, nat in enumerate(nature_track):
        if nat == "ET":
            return i
    return None


def build_track_database(
        *,
        ibtracs_data: dict[str, Any],
        year_start: int = composite_config.YEAR_START,
        year_end: int = composite_config.YEAR_END,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    data = ibtracs_data
    storms = []
    full_tracks: dict[str, pd.DataFrame] = {}

    _LOG.info("Processing %d storms...", data["n_storms"])

    for s in range(data["n_storms"]):
        season = int(data["season"][s])
        if season < year_start or season > year_end:
            continue

        nobs = int(data["numobs"][s])
        if nobs < 4:
            continue

        track_type = data["track_type"][s]
        if "main" not in track_type.lower() and "PROVISIONAL" not in track_type:
            if "MAIN" not in track_type:
                continue

        lat_full = data["lat"][s, :nobs]
        lon_full = data["lon"][s, :nobs]
        nature_full = data["nature"][s, :nobs]
        basin_full = data["basin"][s, :nobs]

        if not np.any(lat_full[np.isfinite(lat_full)] > 0):
            continue

        first_basin = ""
        for b in basin_full:
            if b in composite_config.NH_BASINS:
                first_basin = b
                break
        if not first_basin:
            continue

        recurv_idx = detect_recurvature(
            lat_track=lat_full, lon_track=lon_full, nature_track=nature_full)
        if recurv_idx is None:
            continue

        et_idx = find_et_onset(nature_track=nature_full)
        if et_idx is None:
            continue
        if et_idx <= recurv_idx:
            continue

        times = _num2date(
            time_num=data["time_num"][s, :nobs],
            units=data["time_units"],
            calendar=data["time_cal"],
        )

        recurv_time = times[recurv_idx]
        et_time = times[et_idx]
        if recurv_time is None or et_time is None:
            continue

        wind_full = data["wmo_wind"][s, :nobs]
        pres_full = data["wmo_pres"][s, :nobs]

        track_times = []
        track_lats = []
        track_lons = []
        track_winds = []
        track_pres_arr = []
        track_natures = []
        track_hours_from_recurv = []
        track_hours_from_et = []

        for t in range(nobs):
            tt = times[t]
            if tt is None:
                continue
            dt_recurv = (tt - recurv_time).total_seconds() / 3600
            dt_et = (tt - et_time).total_seconds() / 3600

            if dt_recurv < -composite_config.HOURS_BEFORE or dt_recurv > composite_config.HOURS_AFTER:
                if dt_et < -composite_config.HOURS_BEFORE or dt_et > composite_config.HOURS_AFTER:
                    continue

            track_times.append(tt)
            track_lats.append(lat_full[t])
            track_lons.append(lon_full[t])
            track_winds.append(wind_full[t])
            track_pres_arr.append(pres_full[t])
            track_natures.append(nature_full[t])
            track_hours_from_recurv.append(dt_recurv)
            track_hours_from_et.append(dt_et)

        if len(track_times) < 4:
            continue

        sid = data["sid"][s]
        storm_name = data["name"][s]
        max_wind = np.nanmax(wind_full) if np.any(np.isfinite(wind_full)) else np.nan
        min_pres = np.nanmin(pres_full) if np.any(np.isfinite(pres_full)) else np.nan

        storms.append({
            "storm_id": sid,
            "name": storm_name,
            "basin": first_basin,
            "season": season,
            "recurv_time": recurv_time,
            "recurv_lat": lat_full[recurv_idx],
            "recurv_lon": lon_full[recurv_idx],
            "et_time": et_time,
            "et_lat": lat_full[et_idx],
            "et_lon": lon_full[et_idx],
            "max_wind": max_wind,
            "min_pres": min_pres,
            "n_track_points": len(track_times),
        })

        full_tracks[sid] = pd.DataFrame({
            "time": track_times,
            "lat": track_lats,
            "lon": track_lons,
            "wind": track_winds,
            "pres": track_pres_arr,
            "nature": track_natures,
            "hours_from_recurv": track_hours_from_recurv,
            "hours_from_et": track_hours_from_et,
        })

    storms_df = pd.DataFrame(storms)
    _LOG.info("Found %d recurving NH TCs (%d-%d)", len(storms_df), year_start, year_end)
    if len(storms_df) > 0:
        for basin in storms_df["basin"].unique():
            _LOG.info("  %s: %d storms", basin, (storms_df["basin"] == basin).sum())

    return storms_df, full_tracks


def build_et_nh_track_database(
        *,
        ibtracs_data: dict[str, Any],
        year_start: int = composite_config.YEAR_START,
        year_end: int = composite_config.YEAR_END,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    data = ibtracs_data
    storms = []
    full_tracks: dict[str, pd.DataFrame] = {}

    _LOG.info("Processing %d storms (ET-only)...", data["n_storms"])

    for s in range(data["n_storms"]):
        season = int(data["season"][s])
        if season < year_start or season > year_end:
            continue

        nobs = int(data["numobs"][s])
        if nobs < 4:
            continue

        track_type = data["track_type"][s]
        if "main" not in track_type.lower() and "PROVISIONAL" not in track_type:
            if "MAIN" not in track_type:
                continue

        lat_full = data["lat"][s, :nobs]
        lon_full = data["lon"][s, :nobs]
        nature_full = data["nature"][s, :nobs]
        basin_full = data["basin"][s, :nobs]

        if not np.any(lat_full[np.isfinite(lat_full)] > 0):
            continue

        first_basin = ""
        for b in basin_full:
            if b in composite_config.NH_BASINS:
                first_basin = b
                break
        if not first_basin:
            continue

        et_idx = find_et_onset(nature_track=nature_full)
        if et_idx is None:
            continue

        times = _num2date(
            time_num=data["time_num"][s, :nobs],
            units=data["time_units"],
            calendar=data["time_cal"],
        )
        et_time = times[et_idx]
        if et_time is None:
            continue

        wind_full = data["wmo_wind"][s, :nobs]
        pres_full = data["wmo_pres"][s, :nobs]

        track_times = []
        track_lats = []
        track_lons = []
        track_winds = []
        track_pres_arr = []
        track_natures = []
        track_hours_from_recurv = []
        track_hours_from_et = []

        for t in range(nobs):
            tt = times[t]
            if tt is None:
                continue
            dt_et = (tt - et_time).total_seconds() / 3600
            if dt_et < -composite_config.HOURS_BEFORE or dt_et > composite_config.HOURS_AFTER:
                continue

            track_times.append(tt)
            track_lats.append(lat_full[t])
            track_lons.append(lon_full[t])
            track_winds.append(wind_full[t])
            track_pres_arr.append(pres_full[t])
            track_natures.append(nature_full[t])
            track_hours_from_recurv.append(np.nan)
            track_hours_from_et.append(dt_et)

        if len(track_times) < 4:
            continue

        sid = data["sid"][s]
        storm_name = data["name"][s]
        max_wind = np.nanmax(wind_full) if np.any(np.isfinite(wind_full)) else np.nan
        min_pres = np.nanmin(pres_full) if np.any(np.isfinite(pres_full)) else np.nan

        storms.append({
            "storm_id": sid,
            "name": storm_name,
            "basin": first_basin,
            "season": season,
            "recurv_time": pd.NaT,
            "recurv_lat": np.nan,
            "recurv_lon": np.nan,
            "et_time": et_time,
            "et_lat": lat_full[et_idx],
            "et_lon": lon_full[et_idx],
            "max_wind": max_wind,
            "min_pres": min_pres,
            "n_track_points": len(track_times),
        })

        full_tracks[sid] = pd.DataFrame({
            "time": track_times,
            "lat": track_lats,
            "lon": track_lons,
            "wind": track_winds,
            "pres": track_pres_arr,
            "nature": track_natures,
            "hours_from_recurv": track_hours_from_recurv,
            "hours_from_et": track_hours_from_et,
        })

    storms_df = pd.DataFrame(storms)
    _LOG.info("Found %d NH ET TCs (%d-%d)", len(storms_df), year_start, year_end)
    if len(storms_df) > 0:
        for basin in storms_df["basin"].unique():
            _LOG.info("  %s: %d storms", basin, (storms_df["basin"] == basin).sum())

    return storms_df, full_tracks


def save_et_nh_track_database(
        *,
        storms_df: pd.DataFrame,
        full_tracks: dict[str, pd.DataFrame],
        output_directory: Path,
) -> None:
    os.makedirs(output_directory, exist_ok=True)

    storms_path = os.path.join(output_directory, "et_nh_tracks.csv")
    storms_df.to_csv(storms_path, index=False)
    _LOG.info("Saved storm metadata: %s (%d storms)", storms_path, len(storms_df))

    tracks_dir = os.path.join(output_directory, "individual_et")
    os.makedirs(tracks_dir, exist_ok=True)
    for sid, track_df in full_tracks.items():
        safe_sid = sid.replace("/", "_").replace(" ", "_")
        track_df.to_csv(os.path.join(tracks_dir, f"{safe_sid}.csv"), index=False)

    _LOG.info("Saved individual tracks: %s/", tracks_dir)


def load_et_nh_track_database(
        *,
        tracks_directory: Path,
        build_if_missing: bool = False,
        ibtracs_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    storms_path = os.path.join(tracks_directory, "et_nh_tracks.csv")
    ind_dir = os.path.join(tracks_directory, "individual_et")

    if not os.path.isfile(storms_path):
        if not build_if_missing or ibtracs_path is None:
            raise FileNotFoundError(
                f"Missing {storms_path}; run build_et_nh_track_database first."
            )
        storms_df, full_tracks = build_et_nh_track_database(
            ibtracs_data=load_ibtracs(path=ibtracs_path))
        save_et_nh_track_database(
            storms_df=storms_df, full_tracks=full_tracks,
            output_directory=tracks_directory)
        return storms_df, full_tracks

    storms_df = pd.read_csv(
        storms_path,
        parse_dates=["recurv_time", "et_time"],
        keep_default_na=False,
        na_values=[""],
    )

    full_tracks = {}
    for sid in storms_df["storm_id"]:
        safe_sid = sid.replace("/", "_").replace(" ", "_")
        path = os.path.join(ind_dir, f"{safe_sid}.csv")
        if os.path.exists(path):
            full_tracks[sid] = pd.read_csv(path, parse_dates=["time"])

    return storms_df, full_tracks


def interpolate_track_to_lags(
        *,
        track_df: pd.DataFrame,
        reference: str = "recurvature",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if reference == "recurvature":
        hours_col = "hours_from_recurv"
    else:
        hours_col = "hours_from_et"

    out_lat = np.full(composite_config.N_LAGS, np.nan)
    out_lon = np.full(composite_config.N_LAGS, np.nan)
    out_wind = np.full(composite_config.N_LAGS, np.nan)
    out_pres = np.full(composite_config.N_LAGS, np.nan)

    t_track = track_df[hours_col].values
    sort_idx = np.argsort(t_track)
    t_sorted = t_track[sort_idx]

    for i, lag_h in enumerate(composite_config.LAG_HOURS):
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


def save_track_database(
        *,
        storms_df: pd.DataFrame,
        full_tracks: dict[str, pd.DataFrame],
        output_directory: Path,
) -> None:
    os.makedirs(output_directory, exist_ok=True)

    storms_path = os.path.join(output_directory, "recurving_nh_tracks.csv")
    storms_df.to_csv(storms_path, index=False)
    _LOG.info("Saved storm metadata: %s (%d storms)", storms_path, len(storms_df))

    tracks_dir = os.path.join(output_directory, "individual")
    os.makedirs(tracks_dir, exist_ok=True)
    for sid, track_df in full_tracks.items():
        safe_sid = sid.replace("/", "_").replace(" ", "_")
        track_df.to_csv(os.path.join(tracks_dir, f"{safe_sid}.csv"), index=False)

    _LOG.info("Saved individual tracks: %s/", tracks_dir)


def load_track_database(
        *,
        tracks_directory: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    storms_path = os.path.join(tracks_directory, "recurving_nh_tracks.csv")
    storms_df = pd.read_csv(storms_path, parse_dates=["recurv_time", "et_time"],
                            keep_default_na=False, na_values=[""])

    full_tracks = {}
    ind_dir = os.path.join(tracks_directory, "individual")
    for sid in storms_df["storm_id"]:
        safe_sid = sid.replace("/", "_").replace(" ", "_")
        path = os.path.join(ind_dir, f"{safe_sid}.csv")
        if os.path.exists(path):
            full_tracks[sid] = pd.read_csv(path, parse_dates=["time"])

    return storms_df, full_tracks
