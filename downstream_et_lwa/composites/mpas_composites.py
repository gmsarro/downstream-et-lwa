"""MPAS current/future budget composites: ExTraTrack parsing, geometric
recurvature + Hart (2003) CPS ET detection, MPAS data-source registration
(BARO_N terms, derived Term I/II/III, Fc, RWB, precip, LH/non-QG sources,
QGPV 10 km), data-window filtering, and the composite build driver."""

from __future__ import annotations

import datetime
import glob
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import netCDF4 as nc
import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.composites.engine as engine
import downstream_et_lwa.composites.io as composite_io
import downstream_et_lwa.data_registry as data_registry
import downstream_et_lwa.grid_utils as grid_utils

_LOG = logging.getLogger(__name__)

MPAS_BUDGET_SUFFIXES = {
    "lwa": ("LWAb_N", "lwa"),
    "ua1": ("ua1_N", "ua1"),
    "ua2": ("ua2_N", "ua2"),
    "ep1": ("ep1_N", "ep1"),
    "ep2a": ("ep2_N", "ep2"),
    "ep3a": ("ep3_N", "ep3"),
    "ep4": ("ep4_N", "ep4"),
    "Ub": ("Ub_N", "u"),
    "Urefb": ("Urefb_N", "uref"),
}

BUDGET_VARS = [
    "lwa", "ua1", "ua2", "ep1", "ep2a", "ep3a", "ep4",
    "budget_termI", "budget_termII", "budget_termIII",
]

MPAS_COMPOSITE_DIAGNOSTIC_KEYS = (
    "mpas_cc_Fc",
    "mpas_rwb_awb",
    "mpas_rwb_cwb",
    "mpas_precip",
    "mpas_lh_lwa",
    "mpas_nonqg_lwa",
    "mpas_qgpv_10km",
)

MIN_TRACK_COUNT = 3

MPAS_VALID_YEARS: tuple[int, ...] = (
    1988, 1989,
    1992, 1993, 1994, 1995,
    1997, 1998,
    2001, 2002,
    2005, 2006,
    2010, 2011, 2012, 2013, 2014, 2015, 2016,
)

MPAS_FINAL_VALID_DATE: datetime.datetime = datetime.datetime(2016, 5, 14, 18)

TRACK_COLS = ["lon", "lat", "pres", "wind",
              "v5", "v6", "v7", "v8", "v9",
              "year", "month", "day", "hour"]


def filter_mpas_storms_within_data(
        *,
        storms_df: pd.DataFrame,
        ref_col: str = "recurv_time",
        hours_before: int | None = None,
        hours_after: int | None = None,
        valid_years: tuple[int, ...] = MPAS_VALID_YEARS,
        last_valid_dt: datetime.datetime = MPAS_FINAL_VALID_DATE,
) -> pd.DataFrame:
    if storms_df is None or len(storms_df) == 0:
        return storms_df
    ha = composite_config.HOURS_AFTER if hours_after is None else int(hours_after)
    valid_set = set(int(y) for y in valid_years)

    ref_dt = pd.to_datetime(storms_df[ref_col])
    keep = pd.Series(True, index=storms_df.index)

    yr_mask = ref_dt.dt.year.isin(valid_set)
    keep &= yr_mask
    n_after_year = int(keep.sum())

    last_ts = pd.Timestamp(last_valid_dt)
    end_ts = ref_dt + pd.Timedelta(hours=ha)
    in_range = end_ts <= last_ts
    keep &= in_range
    n_after_window = int(keep.sum())

    n_in = int(len(storms_df))
    n_dropped_year = n_in - n_after_year
    n_dropped_window = n_after_year - n_after_window
    if n_dropped_year or n_dropped_window:
        _LOG.info(
            "MPAS storm filter: %d -> %d (dropped %d for year not in MPAS run "
            "set, %d for [ref, ref+%dh] beyond %s)",
            n_in, n_after_window, n_dropped_year, n_dropped_window, ha, last_ts)
    else:
        _LOG.info("MPAS storm filter: %d kept (all storms within MPAS data support).",
                  n_in)
    return storms_df.loc[keep].copy()


def resolve_mpas_baro_month_nc(
        *,
        budget_root: str,
        scenario: str,
        year: int,
        month: int,
        suffix_stem: str,
) -> str | None:
    stem = suffix_stem[:-3] if suffix_stem.endswith(".nc") else suffix_stem
    fname = f"{year}_{month:02d}_{stem}.nc"
    pat = os.path.join(budget_root, scenario, "*", "BARO_N", fname)
    matches = sorted(glob.glob(pat))
    if not matches:
        return None
    if len(matches) > 1:
        _LOG.warning("%d BARO_N matches for %s %s; using %s",
                     len(matches), scenario, fname, matches[0])
    return matches[0]


def _mpas_file_finder(
        *,
        budget_root: str,
        scenario: str,
        suffix: str,
) -> Callable[[int, int], str | None]:
    stem = suffix[:-3] if suffix.endswith(".nc") else suffix

    def finder(year: int, month: int) -> str | None:
        return resolve_mpas_baro_month_nc(
            budget_root=budget_root, scenario=scenario,
            year=year, month=month, suffix_stem=stem)

    return finder


def _mpas_derived_loader(*, prefix: str, term: str) -> Callable[..., np.ndarray | None]:
    def loader(target_dt: datetime.datetime,
               cache: dict | None = None) -> np.ndarray | None:
        keys = ["ua1", "ua2", "ep1", "ep2a", "ep3a", "ep4"]
        fields = {}
        for k in keys:
            src = data_registry.get(key=f"{prefix}_{k}")
            if src is None:
                continue
            raw = data_registry.load_snapshot(
                source=src, target_dt=target_dt, cache=cache)
            if raw is not None:
                if isinstance(raw, np.ndarray) and raw.shape == (
                        composite_config.NLAT, composite_config.NLON):
                    deseamed = grid_utils.deseam_longitude(field=raw)
                    if deseamed is not None:
                        raw = deseamed
                fields[k] = raw
        if "ua1" not in fields:
            return None
        if term == "termI":
            return grid_utils.compute_budget_termI_global(
                ua1=fields["ua1"], ua2=fields.get("ua2"), ep1=fields.get("ep1"))
        if term == "termII":
            return grid_utils.compute_budget_termII_global(
                ep2a=fields.get("ep2a"), ep3a=fields.get("ep3a"))
        if term == "termIII":
            return fields.get("ep4")
        return None
    return loader


def register_mpas_map_diagnostics(
        *,
        scenario: str,
        data_config: Mapping[str, str],
) -> None:
    prefix = f"mpas_{scenario}"

    cc_dir = data_config.get("mpas_cc", "")
    cc_path = os.path.join(cc_dir, f"mpas_{scenario}_monthly_clim_params.npz")

    def cc_loader(target_dt: datetime.datetime,
                  cache: dict | None = None) -> np.ndarray | None:
        if not os.path.isfile(cc_path):
            return None
        c = cache if cache is not None else {}
        if cc_path not in c:
            c[cc_path] = np.load(cc_path)
        npz = c[cc_path]
        mi = int(target_dt.month) - 1
        return np.asarray(npz["Fc"][mi, :, :], dtype=np.float64)

    key_cc = f"{prefix}_mpas_cc_Fc"
    if key_cc not in data_registry.REGISTRY:
        data_registry.register(source=data_registry.DataSource(
            key=key_cc,
            long_name=(f"MPAS {scenario} carrying capacity Fc "
                       f"(monthly clim NPZ; BN25 stationary A0)"),
            source=f"mpas_{scenario}",
            category="carrying_capacity",
            file_finder=lambda y, m: "derived",
            nc_var="", time_encoding="derived",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="m^2/s^2",
            is_derived=True,
            derived_loader=cc_loader,
        ))

    rwb_root = data_config.get("mpas_rwb", "")

    def _rwb_loader(ncvar: str) -> Callable[..., np.ndarray | None]:
        def loader(target_dt: datetime.datetime,
                   cache: dict | None = None) -> np.ndarray | None:
            path = os.path.join(
                rwb_root, scenario, "masks",
                f"rwb_masks_{target_dt.year}_{target_dt.month:02d}.nc",
            )
            if not os.path.isfile(path):
                return None
            c = cache if cache is not None else {}
            if path not in c:
                c[path] = nc.Dataset(path, "r")
            ds = c[path]
            tv = ds.variables["time"]
            t0 = nc.num2date(tv[0], tv.units, only_use_python_datetimes=True)
            dh = (target_dt.replace(tzinfo=None) - t0).total_seconds() / 3600.0
            ti = int(round(dh / 6.0))
            ti = max(0, min(int(tv.shape[0]) - 1, ti))
            return np.asarray(ds.variables[ncvar][ti, :, :], dtype=np.float64)

        return loader

    for ncvar, suffix in (("rwb_mask_awb", "awb"), ("rwb_mask_cwb", "cwb")):
        key = f"{prefix}_mpas_rwb_{suffix}"
        if key in data_registry.REGISTRY:
            continue
        data_registry.register(source=data_registry.DataSource(
            key=key,
            long_name=f"MPAS {scenario} RWB {suffix.upper()} mask",
            source=f"mpas_{scenario}",
            category="rwb",
            file_finder=lambda y, m: "derived",
            nc_var="", time_encoding="derived",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="1",
            is_derived=True,
            derived_loader=_rwb_loader(ncvar),
        ))

    subset_root = data_config.get("mpas_subset", "")

    def precip_loader(target_dt: datetime.datetime,
                      cache: dict | None = None) -> np.ndarray | None:
        y = int(target_dt.year)
        m = int(target_dt.month)
        d = int(target_dt.day)
        subdir = os.path.join(subset_root, scenario)
        fname = (
            f"mpas.subset.nh.selvar.{scenario}.{y:04d}-{m:02d}-{d:02d}_00:00:00.nc"
        )
        path = os.path.join(subdir, fname)
        if not os.path.isfile(path):
            return None
        c = cache if cache is not None else {}
        if path not in c:
            c[path] = nc.Dataset(path, "r")
        ds = c[path]
        rainc = np.asarray(ds.variables["rainc"][:], dtype=np.float64)
        rainnc = np.asarray(ds.variables["rainnc"][:], dtype=np.float64)
        total = rainc + rainnc
        nt = total.shape[0]
        if nt < 2:
            return None
        di = np.empty_like(total)
        dt_h = 6.0
        di[0] = (total[1] - total[0]) / dt_h
        di[-1] = (total[-1] - total[-2]) / dt_h
        if nt > 2:
            di[1:-1] = (total[2:] - total[:-2]) / (2.0 * dt_h)
        tv = ds.variables["time"]
        t0 = nc.num2date(tv[0], tv.units, only_use_python_datetimes=True)
        dh = (target_dt.replace(tzinfo=None) - t0).total_seconds() / 3600.0
        ti = int(round(dh / 6.0))
        ti = max(0, min(nt - 1, ti))
        rate = di[ti]
        rate = np.maximum(rate, 0.0)
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        out = grid_utils.regrid_curvilinear_to_1deg_nh(
            data=rate, lat_1d=lat, lon_1d=lon)
        if out is not None:
            out = np.where(np.isfinite(out), np.maximum(out, 0.0), out)
        return out

    key_pr = f"{prefix}_mpas_precip"
    if key_pr not in data_registry.REGISTRY:
        data_registry.register(source=data_registry.DataSource(
            key=key_pr,
            long_name=f"MPAS {scenario} precipitation rate (subset, 1 deg NH)",
            source=f"mpas_{scenario}",
            category="precipitation",
            file_finder=lambda y, m: "derived",
            nc_var="", time_encoding="derived",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="mm/hr",
            is_derived=True,
            derived_loader=precip_loader,
        ))

    lh_root = data_config.get("mpas_lh_lwa", "")

    def _mpas_lh_finder(scen: str = scenario) -> Callable[[int, int], str | None]:
        def finder(year: int, month: int) -> str | None:
            path = os.path.join(
                lh_root, scen, "final", "BARO_N",
                f"{int(year)}_{int(month):02d}_LWAlh_N.nc")
            return path if os.path.isfile(path) else None
        return finder

    key_lh = f"{prefix}_mpas_lh_lwa"
    if key_lh not in data_registry.REGISTRY:
        data_registry.register(source=data_registry.DataSource(
            key=key_lh,
            long_name=(f"MPAS {scenario} latent-heating LWA source "
                       f"(barotropic, Huang-Nakamura)"),
            source=f"mpas_{scenario}",
            category="lh_lwa",
            file_finder=_mpas_lh_finder(),
            nc_var="lwa",
            time_encoding="flat",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="m/s/s",
        ))

    nonqg_root = data_config.get("mpas_nonqg", "")

    def _mpas_nonqg_finder(scen: str = scenario) -> Callable[[int, int], str | None]:
        def finder(year: int, month: int) -> str | None:
            path = os.path.join(
                nonqg_root, scen,
                f"{int(year)}_{int(month):02d}_AOUTbaro_N.nc")
            return path if os.path.isfile(path) else None
        return finder

    key_nonqg = f"{prefix}_mpas_nonqg_lwa"
    if key_nonqg not in data_registry.REGISTRY:
        data_registry.register(source=data_registry.DataSource(
            key=key_nonqg,
            long_name=(f"MPAS {scenario} non-QG (ageostrophic) LWA source "
                       f"(barotropic, Huang-Nakamura)"),
            source=f"mpas_{scenario}",
            category="nonqg_lwa",
            file_finder=_mpas_nonqg_finder(),
            nc_var="aout_baro",
            time_encoding="flat",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="m/s/s",
        ))

    def _mpas_qgpv_finder(scen: str = scenario) -> Callable[[int, int], str | None]:
        def finder(year: int, month: int) -> str | None:
            path = os.path.join(
                rwb_root, scen, f"{int(year)}_{int(month):02d}_qgpv.nc")
            return path if os.path.isfile(path) else None
        return finder

    key_qgpv = f"{prefix}_mpas_qgpv_10km"
    if key_qgpv not in data_registry.REGISTRY:
        data_registry.register(source=data_registry.DataSource(
            key=key_qgpv,
            long_name=f"MPAS {scenario} QGPV at ~10 km",
            source=f"mpas_{scenario}",
            category="qgpv",
            file_finder=_mpas_qgpv_finder(),
            nc_var="qgpv",
            time_encoding="flat",
            native_grid=grid_utils.GRID_1DEG_GLOBAL,
            units="PVU",
            level_index=0,
        ))


def register_mpas_sources(
        *,
        scenario: str,
        data_config: Mapping[str, str],
) -> None:
    prefix = f"mpas_{scenario}"
    budget_root = data_config.get("mpas_budget", "")
    for var_key, (suffix, nc_var) in MPAS_BUDGET_SUFFIXES.items():
        key = f"{prefix}_{var_key}"
        if key in data_registry.REGISTRY:
            continue
        data_registry.register(source=data_registry.DataSource(
            key=key,
            long_name=f"MPAS/{scenario} {var_key} (barotropic)",
            source=f"mpas_{scenario}", category="lwa_budget",
            file_finder=_mpas_file_finder(
                budget_root=budget_root, scenario=scenario, suffix=suffix),
            nc_var=nc_var,
            time_encoding="time_x_month",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="m/s" if var_key in ("Ub",) else "m^2/s" if var_key == "lwa" else "m^2/s^2",
        ))
    for term, ln in [("termI", "zonal flux convergence"),
                     ("termII", "meridional flux"),
                     ("termIII", "non-conservative (ep4)")]:
        key = f"{prefix}_budget_{term}"
        if key in data_registry.REGISTRY:
            continue
        data_registry.register(source=data_registry.DataSource(
            key=key,
            long_name=f"MPAS/{scenario} LWA budget {ln}",
            source=f"mpas_{scenario}", category="lwa_budget_derived",
            file_finder=lambda y, m: "derived",
            nc_var="", time_encoding="derived",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="m/s/s",
            is_derived=True,
            derived_loader=_mpas_derived_loader(prefix=prefix, term=term),
        ))

    register_mpas_map_diagnostics(scenario=scenario, data_config=data_config)


def parse_et_dat(*, path: Path) -> list[tuple[str, pd.DataFrame]]:
    blocks = []
    cur: dict[str, Any] = {"meta": None, "rows": []}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("start"):
                if cur["rows"]:
                    blocks.append(cur)
                cur = {"meta": s, "rows": []}
            else:
                parts = s.split()
                if len(parts) < len(TRACK_COLS):
                    continue
                try:
                    cur["rows"].append([float(x) for x in parts[:len(TRACK_COLS)]])
                except ValueError:
                    continue
    if cur["rows"]:
        blocks.append(cur)

    storms = []
    for b in blocks:
        df = pd.DataFrame(b["rows"], columns=TRACK_COLS)
        df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        df = df.sort_values("time").reset_index(drop=True)
        df.loc[df["pres"] < 1100, "pres"] = df.loc[df["pres"] < 1100, "pres"] * 100
        m = re.match(
            r"start\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
            b["meta"])
        track_id = m.group(6) if m else "0000"
        start_dt = df["time"].iloc[0]
        storm_id = f"{start_dt:%Y%m%d%H}_{track_id}"
        storms.append((storm_id, df))
    return storms


def detect_recurvature_mpas(*, df: pd.DataFrame, min_lat: float = 15.0) -> int | None:
    lon = df["lon"].values
    lat = df["lat"].values
    n = len(lon)

    def dlon(i1: int, i2: int) -> float:
        d = lon[i2] - lon[i1]
        if abs(d) > 180:
            d -= np.sign(d) * 360
        return float(d)

    for i in range(1, n - 2):
        if lat[i] < min_lat:
            continue
        d_before = dlon(i - 1, i)
        d_after1 = dlon(i, i + 1)
        d_after2 = dlon(i + 1, i + 2) if i + 2 < n else d_after1
        if d_before <= 0 and d_after1 >= 0 and d_after2 >= 0:
            return i
    return None


def detect_et_mpas(*, df: pd.DataFrame, min_lat: float = 25.0,
                   persist: int = 2) -> int | None:
    if "v9" not in df.columns:
        return None
    v9 = df["v9"].to_numpy(dtype=float)
    lat = df["lat"].to_numpy(dtype=float)
    n = len(v9)
    if n < persist:
        return None
    cold = (v9 < 0) & (lat >= min_lat)
    run = 0
    start = None
    for i in range(n):
        if cold[i]:
            if start is None:
                start = i
            run += 1
            if run >= persist:
                return start
        else:
            run = 0
            start = None
    return None


def basin_of(*, lon: float, lat: float) -> str | None:
    lon = lon % 360
    if lat < 0:
        return None
    if 100 <= lon < 180:
        return "WP"
    if 180 <= lon < 260:
        return "EP"
    if 260 <= lon < 360 or lon < 20:
        return "NA"
    if 40 <= lon < 100:
        return "NI"
    return None


def discover_mpas_traj_paths(*, et_track_dir: Path, scenario: str) -> list[str]:
    pat = os.path.join(str(et_track_dir), f"traj_et_mpas_avg_*_{scenario}.dat")
    return sorted(glob.glob(pat))


def discover_ettrack_txt_files(*, root: Path | None, scenario: str) -> list[str]:
    if not root or not os.path.isdir(root):
        return []
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.startswith(f"ettrack_{scenario}_") and f.endswith(".txt"):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def parse_ettrack_txt(
        *,
        path: str,
        root_for_id: str,
) -> tuple[str | None, pd.DataFrame | None]:
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < len(TRACK_COLS):
                continue
            try:
                rows.append([float(x) for x in parts[:len(TRACK_COLS)]])
            except ValueError:
                continue
    if len(rows) < 4:
        return None, None
    df = pd.DataFrame(rows, columns=TRACK_COLS)
    df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values("time").reset_index(drop=True)
    df.loc[df["pres"] < 1100, "pres"] = df.loc[df["pres"] < 1100, "pres"] * 100
    try:
        rel = os.path.relpath(path, root_for_id)
    except ValueError:
        rel = os.path.basename(path)
    sid = rel.replace(os.sep, "_")
    if sid.lower().endswith(".txt"):
        sid = sid[:-4]
    return sid, df


def _append_one_mpas_storm(
        *,
        scenario: str,
        sid: str,
        df: pd.DataFrame,
        file_tag: str,
        full_tracks: dict[str, pd.DataFrame],
        rows: list[dict],
) -> None:
    if len(df) < 4:
        return
    r_idx = detect_recurvature_mpas(df=df)
    if r_idx is None:
        return

    recurv_time = df["time"].iloc[r_idx]
    recurv_lat = float(df["lat"].iloc[r_idx])
    recurv_lon = float(df["lon"].iloc[r_idx])
    basin = basin_of(lon=recurv_lon, lat=recurv_lat)
    if basin is None:
        return

    storm_id = f"MPAS_{scenario}_{sid}"
    if storm_id in full_tracks:
        storm_id = f"{storm_id}_{file_tag}"
    season = int(recurv_time.year)
    max_wind = float(df["wind"].max(skipna=True))
    min_pres = float(df["pres"].min(skipna=True))

    hours_from_recurv = (df["time"] - recurv_time).dt.total_seconds() / 3600.0

    et_idx = detect_et_mpas(df=df)
    if et_idx is None or et_idx < r_idx:
        return
    et_time = df["time"].iloc[et_idx]
    et_lat = float(df["lat"].iloc[et_idx])
    et_lon = float(df["lon"].iloc[et_idx])
    nature_label = "ET"

    hours_from_et = (df["time"] - et_time).dt.total_seconds() / 3600.0

    keep = ((hours_from_recurv >= -composite_config.HOURS_BEFORE)
            & (hours_from_recurv <= composite_config.HOURS_AFTER))
    if keep.sum() < 4:
        return
    sub = df.loc[keep].copy()
    sub["nature"] = nature_label
    sub["hours_from_recurv"] = hours_from_recurv.loc[keep]
    sub["hours_from_et"] = hours_from_et.loc[keep]
    sub = sub[["time", "lat", "lon", "wind", "pres", "nature",
               "hours_from_recurv", "hours_from_et"]]

    full_tracks[storm_id] = sub.reset_index(drop=True)
    rows.append({
        "storm_id": storm_id,
        "name": f"mpas{scenario[0].upper()}{sid}"[:200],
        "basin": basin,
        "season": season,
        "recurv_time": recurv_time,
        "recurv_lat": recurv_lat,
        "recurv_lon": recurv_lon,
        "et_time": et_time,
        "et_lat": et_lat,
        "et_lon": et_lon,
        "max_wind": max_wind,
        "min_pres": min_pres,
        "n_track_points": len(sub),
    })


def _append_one_mpas_storm_et_only(
        *,
        scenario: str,
        sid: str,
        df: pd.DataFrame,
        file_tag: str,
        full_tracks: dict[str, pd.DataFrame],
        rows: list[dict],
) -> None:
    if len(df) < 4:
        return
    et_idx = detect_et_mpas(df=df)
    if et_idx is None:
        return

    et_time = df["time"].iloc[et_idx]
    et_lat = float(df["lat"].iloc[et_idx])
    et_lon = float(df["lon"].iloc[et_idx])
    basin = basin_of(lon=et_lon, lat=et_lat)
    if basin is None:
        return

    storm_id = f"MPAS_{scenario}_{sid}"
    if storm_id in full_tracks:
        storm_id = f"{storm_id}_{file_tag}"
    season = int(et_time.year)
    max_wind = float(df["wind"].max(skipna=True))
    min_pres = float(df["pres"].min(skipna=True))

    hours_from_et = (df["time"] - et_time).dt.total_seconds() / 3600.0
    keep = ((hours_from_et >= -composite_config.HOURS_BEFORE)
            & (hours_from_et <= composite_config.HOURS_AFTER))
    if keep.sum() < 4:
        return
    sub = df.loc[keep].copy()
    sub["nature"] = "ET"
    sub["hours_from_recurv"] = np.nan
    sub["hours_from_et"] = hours_from_et.loc[keep]
    sub = sub[["time", "lat", "lon", "wind", "pres", "nature",
               "hours_from_recurv", "hours_from_et"]]

    full_tracks[storm_id] = sub.reset_index(drop=True)
    rows.append({
        "storm_id": storm_id,
        "name": f"mpas{scenario[0].upper()}{sid}"[:200],
        "basin": basin,
        "season": season,
        "recurv_time": pd.NaT,
        "recurv_lat": np.nan,
        "recurv_lon": np.nan,
        "et_time": et_time,
        "et_lat": et_lat,
        "et_lon": et_lon,
        "max_wind": max_wind,
        "min_pres": min_pres,
        "n_track_points": len(sub),
    })


def build_mpas_tracks(
        *,
        scenario: str,
        track_path: Path | None = None,
        track_paths: list[Path] | None = None,
        et_track_dir: Path | None = None,
        all_tracks_root: Path | None = None,
        et_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict] = []
    full_tracks: dict[str, pd.DataFrame] = {}

    def append(sid: str, df: pd.DataFrame, file_tag: str) -> None:
        if et_only:
            _append_one_mpas_storm_et_only(
                scenario=scenario, sid=sid, df=df, file_tag=file_tag,
                full_tracks=full_tracks, rows=rows)
        else:
            _append_one_mpas_storm(
                scenario=scenario, sid=sid, df=df, file_tag=file_tag,
                full_tracks=full_tracks, rows=rows)

    force_dat = track_path is not None or track_paths is not None

    txt_root = None
    if not force_dat and all_tracks_root and os.path.isdir(all_tracks_root):
        txt_root = all_tracks_root

    if txt_root:
        txts = discover_ettrack_txt_files(root=txt_root, scenario=scenario)
        if txts:
            _LOG.info("MPAS/%s ettrack .txt: %d file(s) under %s",
                      scenario, len(txts), txt_root)
            for path in txts:
                sid, df = parse_ettrack_txt(path=path, root_for_id=str(txt_root))
                if df is None or sid is None:
                    continue
                file_tag = os.path.splitext(os.path.basename(path))[0]
                append(sid, df, file_tag)
            return pd.DataFrame(rows), full_tracks
        _LOG.info("MPAS/%s: no ettrack .txt in %s, using .dat",
                  scenario, txt_root)

    if track_paths is not None:
        paths = [str(p) for p in track_paths if p]
    elif track_path is not None:
        paths = [str(track_path)]
    elif et_track_dir is not None:
        paths = discover_mpas_traj_paths(
            et_track_dir=et_track_dir, scenario=scenario)
        if not paths:
            legacy = os.path.join(
                str(et_track_dir), f"traj_et_mpas_avg_2013_{scenario}.dat")
            if os.path.isfile(legacy):
                paths = [legacy]
    else:
        paths = []
    if paths:
        _LOG.info("MPAS/%s track file(s): %s", scenario, paths)

    for path in paths:
        if not os.path.isfile(path):
            _LOG.info("skip missing %s", path)
            continue
        storms = parse_et_dat(path=Path(path))
        file_tag = os.path.splitext(os.path.basename(path))[0]
        for sid, df in storms:
            append(sid, df, file_tag)

    storms_df = pd.DataFrame(rows)
    return storms_df, full_tracks


def load_mpas_composite(*, path: Path, prefix: str) -> dict[str, Any]:
    with nc.Dataset(str(path), "r") as ds:
        d: dict[str, Any] = {}
        for v in ds.variables:
            if v.endswith("_mean"):
                k = v[:-5]
                if k.startswith(prefix + "_"):
                    k = k[len(prefix) + 1:]
                d[k] = np.array(ds[v][:])
            elif v.endswith("_sumsq"):
                k = v[:-6]
                if k.startswith(prefix + "_"):
                    k = k[len(prefix) + 1:]
                d[f"{k}__sumsq"] = np.array(ds[v][:])
            elif v.endswith("_count_field"):
                k = v[:-12]
                if k.startswith(prefix + "_"):
                    k = k[len(prefix) + 1:]
                d[f"{k}__count_field"] = np.array(ds[v][:])

        d["_lat"] = np.array(ds["rel_lat"][:])
        d["_lon"] = np.array(ds["rel_lon"][:])
        d["_lag_hours"] = np.array(ds["lag_hours"][:])
        d["_n_storms"] = ds.dimensions["storm"].size

        if "track_rel_lat" in ds.variables:
            trl = ds["track_rel_lat"][:]
            trn = ds["track_rel_lon"][:]
            valid = np.sum(np.isfinite(trl), axis=0)
            d["_mean_rel_lat"] = np.where(valid >= MIN_TRACK_COUNT,
                                          np.nanmean(trl, axis=0), np.nan)
            d["_mean_rel_lon"] = np.where(valid >= MIN_TRACK_COUNT,
                                          np.nanmean(trn, axis=0), np.nan)

        if "recurv_lat" in ds.variables:
            d["_mean_abs_lat"] = float(np.nanmean(ds["recurv_lat"][:]))
            d["_mean_abs_lon"] = float(np.nanmean(ds["recurv_lon"][:]))

    return d


def run_scenario_build(
        *,
        scenario: str,
        basins: list[str],
        workers: int,
        data_config: Mapping[str, str],
        output_directory: Path,
        et_track_dir: Path | None = None,
        all_tracks_root: Path | None = None,
        with_map_diagnostics: bool = True,
) -> bool:
    os.makedirs(output_directory, exist_ok=True)

    prefix = f"mpas_{scenario}"
    register_mpas_sources(scenario=scenario, data_config=data_config)

    var_keys = [f"{prefix}_{v}" for v in BUDGET_VARS]
    if with_map_diagnostics:
        var_keys = list(var_keys) + [
            f"{prefix}_{vk}" for vk in MPAS_COMPOSITE_DIAGNOSTIC_KEYS]

    print(f"\n=== Building tracks for MPAS/{scenario} ===", flush=True)
    storms_df, full_tracks = build_mpas_tracks(
        scenario=scenario,
        et_track_dir=et_track_dir,
        all_tracks_root=all_tracks_root,
    )
    if len(storms_df) == 0:
        print("  No storms found after filtering, aborting.", flush=True)
        return False

    storms_df = filter_mpas_storms_within_data(
        storms_df=storms_df, ref_col="recurv_time")
    kept_ids = set(storms_df["storm_id"].tolist())
    full_tracks = {sid: tr for sid, tr in full_tracks.items()
                   if sid in kept_ids}
    if len(storms_df) == 0:
        print("  No MPAS storms left after data-window filter, aborting.",
              flush=True)
        return False
    print(f"  {len(storms_df)} storms within MPAS data window", flush=True)
    print(storms_df["basin"].value_counts().to_string(), flush=True)

    print(f"\n=== Composites for MPAS/{scenario} ({basins}) ===", flush=True)
    premian = tuple(float(x)
                    for x in composite_config.COMPOSITE_PREMEAN_VOLUME_SIGMA_3D)
    accum = engine.build_composites_parallel(
        storms_df=storms_df, full_tracks=full_tracks,
        var_keys=var_keys, basins=basins,
        reference="recurvature", n_workers=workers,
        premean_volume_sigma_3d=premian,
    )
    composite_io.save_composites(
        accum=accum, reference=f"recurvature_mpas_{scenario}",
        output_directory=output_directory,
        premean_volume_sigma_3d=premian)
    return True


def main(
        data_config: Annotated[Path, typer.Option(
            help="JSON file mapping data-source keys to root directories "
                 "(needs mpas_budget, mpas_cc, mpas_rwb, mpas_subset, "
                 "mpas_lh_lwa, mpas_nonqg)")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for composite NetCDF output")],
        scenario: Annotated[Optional[list[str]], typer.Option(
            help="current and/or future")] = None,
        basins: Annotated[Optional[list[str]], typer.Option()] = None,
        workers: Annotated[int, typer.Option()] = 8,
        et_track_directory: Annotated[Optional[Path], typer.Option(
            help="Directory with traj_et_mpas_avg_*_{scenario}.dat")] = None,
        all_tracks_root: Annotated[Optional[Path], typer.Option(
            help="Directory walked for ettrack_{scenario}_*.txt")] = None,
        no_map_diagnostics: Annotated[bool, typer.Option(
            help="Do not composite map extras (Fc, RWB, precip, QGPV)")] = False,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())

    cfg = data_registry.load_data_config(path=data_config)
    data_registry.register_all(data_config=cfg)

    if scenario is None:
        scenario = ["current", "future"]
    if basins is None:
        basins = ["WP", "NA"]

    for scen in scenario:
        ok = run_scenario_build(
            scenario=scen, basins=list(basins), workers=workers,
            data_config=cfg, output_directory=output_directory,
            et_track_dir=et_track_directory,
            all_tracks_root=all_tracks_root,
            with_map_diagnostics=not no_map_diagnostics,
        )
        if ok:
            print(f"Done: MPAS/{scen} composites in {output_directory}",
                  flush=True)


if __name__ == "__main__":
    typer.run(main)
