"""Build a per-storm (time, lat, lon) catalog of LN24-decomposed LWA budget forcing."""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Callable

import netCDF4 as nc
import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

import downstream_et_lwa.lh_removal.advection_kernel as advection_kernel

_LOG = logging.getLogger(__name__)

DEFAULT_TC_RADIUS_DEG = 10.0

BARO_VARS = {
    "lwa": ("LWAb_N", "lwa"),
    "ua1": ("ua1_N", "ua1"),
    "ua2": ("ua2_N", "ua2"),
    "ep1": ("ep1_N", "ep1"),
    "ep2a": ("ep2a_N", "ep2a"),
    "ep3a": ("ep3a_N", "ep3a"),
    "ep4": ("ep4_N", "ep4"),
}

DT_INPUT_SEC = 6 * 3600
SAFE_A_FLOOR = 1e-3

MonthLoader = Callable[[int, int], tuple[np.ndarray, np.ndarray]]


def _days_in_month(*, year: int, month: int) -> int:
    if month == 12:
        return (datetime.datetime(year + 1, 1, 1) - datetime.datetime(year, 12, 1)).days
    return (datetime.datetime(year, month + 1, 1) - datetime.datetime(year, month, 1)).days


def _month_times(*, year: int, month: int) -> np.ndarray:
    t0 = datetime.datetime(year, month, 1)
    n_t = _days_in_month(year=year, month=month) * 4
    return np.array([(t0 + datetime.timedelta(hours=6 * i))
                     for i in range(n_t)], dtype="datetime64[s]")


def _load_baro_month(*, var: str, year: int, month: int,
                     baro_directory: Path) -> tuple[np.ndarray, np.ndarray]:
    suf, ncvar = BARO_VARS[var]
    path = baro_directory / f"{year}_{suf}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    n_t = _days_in_month(year=year, month=month) * 4
    with nc.Dataset(path, "r") as d:
        arr = np.asarray(d[ncvar][:n_t, month - 1, :, :], dtype=np.float64)
    return _month_times(year=year, month=month), arr


def _load_lh_month(*, year: int, month: int,
                   lh_source_directory: Path) -> tuple[np.ndarray, np.ndarray]:
    path = lh_source_directory / f"{year}_{month:02d}_LWAb_N.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    with nc.Dataset(path, "r") as d:
        arr = np.asarray(d["lwa"][:], dtype=np.float64)
    n_t = _days_in_month(year=year, month=month) * 4
    if arr.shape[0] != n_t:
        raise ValueError(f"LH file {path} has Nt={arr.shape[0]}, expected {n_t}")
    return _month_times(year=year, month=month), arr


def _make_baro_loader(*, var: str, baro_directory: Path) -> MonthLoader:
    def _loader(year: int, month: int) -> tuple[np.ndarray, np.ndarray]:
        return _load_baro_month(var=var, year=year, month=month,
                                baro_directory=baro_directory)

    return _loader


def _make_lh_loader(*, lh_source_directory: Path) -> MonthLoader:
    def _loader(year: int, month: int) -> tuple[np.ndarray, np.ndarray]:
        return _load_lh_month(year=year, month=month,
                              lh_source_directory=lh_source_directory)

    return _loader


def _load_window(*, t_start: datetime.datetime, t_end: datetime.datetime,
                 loader: MonthLoader) -> tuple[np.ndarray, np.ndarray]:
    months = []
    cur = datetime.datetime(t_start.year, t_start.month, 1)
    while cur <= t_end:
        months.append((cur.year, cur.month))
        ny, nm = (cur.year + 1, 1) if cur.month == 12 else (cur.year, cur.month + 1)
        cur = datetime.datetime(ny, nm, 1)

    times_chunks, data_chunks = [], []
    for (y, m) in months:
        t_m, d_m = loader(y, m)
        times_chunks.append(t_m)
        data_chunks.append(d_m)
    times = np.concatenate(times_chunks)
    data = np.concatenate(data_chunks, axis=0)

    t_start_np = np.datetime64(t_start)
    t_end_np = np.datetime64(t_end)
    mask = (times >= t_start_np) & (times <= t_end_np)
    return times[mask], data[mask]


def _zonal_grad(*, field: np.ndarray) -> np.ndarray:
    g = np.empty_like(field)
    g[..., :, 1:-1] = (field[..., :, 2:] - field[..., :, :-2]) / 2.0
    g[..., :, 0] = (field[..., :, 1] - field[..., :, -1]) / 2.0
    g[..., :, -1] = (field[..., :, 0] - field[..., :, -2]) / 2.0
    denom = (advection_kernel.A_EARTH
             * np.where(np.abs(advection_kernel.COSPHI) < 1e-3, np.nan,
                        advection_kernel.COSPHI)
             * advection_kernel.DLAMBDA)
    return g / denom[:, None]


def diagnose_budget(*, fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    A = fields["lwa"]
    F_lambda = fields["ua1"] + fields["ua2"] + fields["ep1"]
    termI = -_zonal_grad(field=F_lambda)

    cosphi = np.where(np.abs(advection_kernel.COSPHI) < 1e-3, np.nan,
                      advection_kernel.COSPHI)
    denom_y = (2.0 * advection_kernel.A_EARTH * cosphi
               * advection_kernel.DLAMBDA)[:, None]
    termII = (fields["ep2a"] - fields["ep3a"]) / denom_y

    termIII = fields["ep4"]

    tend = np.full_like(A, np.nan)
    tend[1:-1] = (A[2:] - A[:-2]) / (2.0 * DT_INPUT_SEC)
    tend[0] = (A[1] - A[0]) / DT_INPUT_SEC
    tend[-1] = (A[-1] - A[-2]) / DT_INPUT_SEC

    R = tend - termI - termII - termIII
    return dict(
        F_lambda=F_lambda, termI=termI, termII=termII, termIII=termIII,
        tend=tend, R=R,
    )


def closure_metric(*, diag: dict[str, np.ndarray]) -> float:
    t = diag["tend"][1:-1]
    rhs = (diag["termI"][1:-1] + diag["termII"][1:-1]
           + diag["termIII"][1:-1] + diag["R"][1:-1])
    err = t - rhs
    rms_err = float(np.sqrt(np.nanmean(err ** 2)))
    rms_t = float(np.sqrt(np.nanmean(t ** 2)))
    return rms_err / max(rms_t, 1e-30)


def ln24_decompose(
    *,
    A: np.ndarray,
    F_lambda: np.ndarray,
    ep2a: np.ndarray,
    ep3a: np.ndarray,
    R: np.ndarray,
    LH: np.ndarray,
    j_south: int = 6,
    j_north: int = 86,
    cy_clip: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A_safe = np.where(np.abs(A) < SAFE_A_FLOOR,
                      np.sign(A) * SAFE_A_FLOOR + (A == 0) * SAFE_A_FLOOR,
                      A)
    c_x = F_lambda / A_safe
    c_x = np.clip(c_x, -cy_clip, cy_clip)

    A_jp = np.empty_like(A_safe)
    A_jm = np.empty_like(A_safe)
    A_jp[..., :-1, :] = A_safe[..., 1:, :]
    A_jp[..., -1, :] = A_safe[..., -1, :]
    A_jm[..., 1:, :] = A_safe[..., :-1, :]
    A_jm[..., 0, :] = A_safe[..., 0, :]

    cosp = np.cos(np.deg2rad(advection_kernel.LATS + 1.0))[:, None]
    cosm = np.cos(np.deg2rad(advection_kernel.LATS - 1.0))[:, None]

    cyp = np.zeros_like(A_safe)
    cym = np.zeros_like(A_safe)
    cyp[..., j_south:j_north + 1, :] = (
        -ep2a[..., j_south:j_north + 1, :]
        / (A_jp[..., j_south:j_north + 1, :] * cosp[j_south:j_north + 1, :])
    )
    cym[..., j_south:j_north + 1, :] = (
        -ep3a[..., j_south:j_north + 1, :]
        / (A_jm[..., j_south:j_north + 1, :] * cosm[j_south:j_north + 1, :])
    )
    cyp = np.clip(cyp, -cy_clip, cy_clip)
    cym = np.clip(cym, -cy_clip, cy_clip)

    gamma = np.where(R < 0.0, R, 0.0) / A_safe
    S_other = np.maximum(R, 0.0) - np.maximum(LH, 0.0)
    S_other = np.maximum(S_other, 0.0)
    return c_x, cyp, cym, gamma, S_other


def _load_indiv_track(*, storm_id: str,
                      individual_tracks_directory: Path) -> pd.DataFrame:
    p = individual_tracks_directory / f"{storm_id}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    tr = pd.read_csv(p, parse_dates=["time"])
    return tr.sort_values("time").reset_index(drop=True)


def _interp_storm_center_at(*, times: np.ndarray, tr: pd.DataFrame
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype="datetime64[ns]")
    tr_t = tr["time"].values.astype("datetime64[ns]")
    tr_lat = tr["lat"].to_numpy(dtype=np.float64)
    tr_lon = tr["lon"].to_numpy(dtype=np.float64)
    t_secs = times.astype("int64") / 1e9
    tr_secs = tr_t.astype("int64") / 1e9

    inside = (times >= tr_t[0]) & (times <= tr_t[-1])
    lat_out = np.interp(t_secs, tr_secs, tr_lat,
                        left=tr_lat[0], right=tr_lat[-1])
    ang = np.deg2rad(tr_lon)
    sin_l = np.interp(t_secs, tr_secs, np.sin(ang),
                      left=np.sin(ang[0]), right=np.sin(ang[-1]))
    cos_l = np.interp(t_secs, tr_secs, np.cos(ang),
                      left=np.cos(ang[0]), right=np.cos(ang[-1]))
    lon_out = (np.rad2deg(np.arctan2(sin_l, cos_l)) + 360.0) % 360.0
    return lat_out, lon_out, inside


def _great_circle_arc_deg(*, lat1: float, lon1: float, lat2_grid: np.ndarray,
                          lon2_grid: np.ndarray) -> np.ndarray:
    phi1 = np.deg2rad(lat1)
    lam1 = np.deg2rad(lon1)
    phi2 = np.deg2rad(lat2_grid)[:, None]
    lam2 = np.deg2rad(lon2_grid)[None, :]
    dlam = lam2 - lam1
    cos_arc = (np.sin(phi1) * np.sin(phi2)
               + np.cos(phi1) * np.cos(phi2) * np.cos(dlam))
    cos_arc = np.clip(cos_arc, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_arc))


def build_tc_mask(*, times_out_s: np.ndarray, t0_window: pd.Timestamp,
                  storm_id: str, individual_tracks_directory: Path,
                  R_deg: float = DEFAULT_TC_RADIUS_DEG) -> np.ndarray:
    tr = _load_indiv_track(storm_id=storm_id,
                           individual_tracks_directory=individual_tracks_directory)
    times_dt = pd.to_datetime(t0_window) + pd.to_timedelta(times_out_s, unit="s")
    times_np = times_dt.values.astype("datetime64[ns]")
    lat_t, lon_t, inside = _interp_storm_center_at(times=times_np, tr=tr)

    Nt = len(times_out_s)
    mask = np.zeros((Nt, advection_kernel.NLAT, advection_kernel.NLON),
                    dtype=np.float32)
    for k in range(Nt):
        if not inside[k]:
            continue
        d = _great_circle_arc_deg(lat1=lat_t[k], lon1=lon_t[k],
                                  lat2_grid=advection_kernel.LATS,
                                  lon2_grid=advection_kernel.LONS)
        mask[k] = (d <= R_deg).astype(np.float32)
    return mask


def linear_resample(*, times_in_s: np.ndarray, field: np.ndarray,
                    times_out_s: np.ndarray) -> np.ndarray:
    field_clean = np.where(np.isfinite(field), field, 0.0).astype(np.float64)
    times_in_s = np.asarray(times_in_s, dtype=np.float64)
    times_out_s = np.asarray(times_out_s, dtype=np.float64)
    Nt_in = times_in_s.size
    j = np.clip(np.searchsorted(times_in_s, times_out_s, side="right") - 1,
                0, Nt_in - 2)
    t0 = times_in_s[j]
    t1 = times_in_s[j + 1]
    w = (times_out_s - t0) / np.maximum(t1 - t0, 1e-9)
    w = np.clip(w, 0.0, 1.0)
    w = w[:, None, None]
    return (1.0 - w) * field_clean[j] + w * field_clean[j + 1]


def _snap_6h(*, value: datetime.datetime, up: bool) -> datetime.datetime:
    h = (value.hour // 6) * 6
    snapped = value.replace(hour=h, minute=0, second=0, microsecond=0)
    if up and snapped < value:
        snapped += datetime.timedelta(hours=6)
    return snapped


def build_event_catalog(
    *,
    storm_id: str,
    tracks_file: Path,
    individual_tracks_directory: Path,
    baro_directory: Path,
    lh_source_directory: Path,
    output_directory: Path,
    t_pre_days: float = 1.0,
    t_post_days: float = 8.0,
    dt_int_min: float = 30.0,
    tc_radius_deg: float = DEFAULT_TC_RADIUS_DEG,
    overwrite: bool = False,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    out_path = output_directory / f"event_{storm_id}.nc"
    if out_path.exists() and not overwrite:
        _LOG.info("Skipping %s; exists (use --overwrite to rebuild)", out_path)
        return out_path

    df = pd.read_csv(
        tracks_file, parse_dates=["recurv_time", "et_time"],
        keep_default_na=False, na_values=[""],
    )
    rows = df[df["storm_id"] == storm_id]
    if len(rows) == 0:
        raise KeyError(f"storm_id {storm_id} not in {tracks_file}")
    row = rows.iloc[0]
    _LOG.info("storm_id=%s basin=%s recurv_time=%s et_time=%s",
              storm_id, row["basin"], row["recurv_time"], row["et_time"])

    t_recurv: pd.Timestamp = row["recurv_time"]
    t_et: pd.Timestamp = row["et_time"]
    t_start_dt = (t_recurv - datetime.timedelta(days=t_pre_days)).to_pydatetime()
    t_end_dt = (t_et + datetime.timedelta(days=t_post_days)).to_pydatetime()

    t_start_dt = _snap_6h(value=t_start_dt, up=False)
    t_end_dt = _snap_6h(value=t_end_dt, up=True)
    _LOG.info("Window: %s -> %s (%.2f days)", t_start_dt, t_end_dt,
              (t_end_dt - t_start_dt).total_seconds() / 86400)

    fields = {}
    times = np.array([], dtype="datetime64[s]")
    for v in BARO_VARS:
        times, arr = _load_window(
            t_start=t_start_dt, t_end=t_end_dt,
            loader=_make_baro_loader(var=v, baro_directory=baro_directory),
        )
        fields[v] = arr
    times_6h = times

    lh_times, lh = _load_window(
        t_start=t_start_dt, t_end=t_end_dt,
        loader=_make_lh_loader(lh_source_directory=lh_source_directory),
    )
    if not np.array_equal(lh_times, times_6h):
        raise RuntimeError("LH and BARO_N times mismatch")

    diag = diagnose_budget(fields=fields)
    closure = closure_metric(diag=diag)
    _LOG.info("Native-grid closure |dA/dt - sum|2 / |dA/dt|2 = %.2e", closure)

    c_x, cyp, cym, gamma, S_other = ln24_decompose(
        A=fields["lwa"],
        F_lambda=diag["F_lambda"],
        ep2a=fields["ep2a"], ep3a=fields["ep3a"],
        R=diag["R"], LH=lh,
    )

    t0_s = pd.Timestamp(times_6h[0]).timestamp()
    times_in_s = np.array(
        [(pd.Timestamp(t).timestamp() - t0_s) for t in times_6h],
        dtype=np.float64,
    )
    dt_int = float(dt_int_min) * 60.0
    n_out = int(np.round(times_in_s[-1] / dt_int)) + 1
    times_out_s = np.arange(n_out, dtype=np.float64) * dt_int

    _LOG.info("Resampling %d 6-h snapshots -> %d steps at dt=%g min",
              len(times_in_s), n_out, dt_int_min)

    A_obs_int = linear_resample(times_in_s=times_in_s, field=fields["lwa"],
                                times_out_s=times_out_s)
    c_x_int = linear_resample(times_in_s=times_in_s, field=c_x,
                              times_out_s=times_out_s)
    cyp_int = linear_resample(times_in_s=times_in_s, field=cyp,
                              times_out_s=times_out_s)
    cym_int = linear_resample(times_in_s=times_in_s, field=cym,
                              times_out_s=times_out_s)
    gamma_int = linear_resample(times_in_s=times_in_s, field=gamma,
                                times_out_s=times_out_s)
    eps4_int = linear_resample(times_in_s=times_in_s, field=fields["ep4"],
                               times_out_s=times_out_s)
    S_other_int = linear_resample(times_in_s=times_in_s, field=S_other,
                                  times_out_s=times_out_s)
    LH_int = linear_resample(times_in_s=times_in_s, field=lh,
                             times_out_s=times_out_s)

    _LOG.info("Building TC mask, R = %g deg", tc_radius_deg)
    tc_mask = build_tc_mask(
        times_out_s=times_out_s,
        t0_window=pd.Timestamp(times_6h[0]),
        storm_id=storm_id,
        individual_tracks_directory=individual_tracks_directory,
        R_deg=tc_radius_deg,
    )
    n_inside = int((tc_mask > 0).any(axis=(1, 2)).sum())
    _LOG.info("Mask non-empty for %d/%d time steps (%.1f h of TC presence)",
              n_inside, n_out, n_inside * dt_int / 3600)

    tmp = out_path.with_suffix(".nc.tmp")
    with nc.Dataset(tmp, "w", format="NETCDF4") as o:
        o.createDimension("time", n_out)
        o.createDimension("time_native", len(times_in_s))
        o.createDimension("lat", advection_kernel.NLAT)
        o.createDimension("lon", advection_kernel.NLON)

        vt = o.createVariable("time", "f8", ("time",))
        vt.units = f"seconds since {pd.Timestamp(times_6h[0]).isoformat()}"
        vt[:] = times_out_s
        vtn = o.createVariable("time_native", "f8", ("time_native",))
        vtn.units = vt.units
        vtn[:] = times_in_s
        vlat = o.createVariable("lat", "f4", ("lat",))
        vlat.units = "degrees_north"
        vlat[:] = advection_kernel.LATS.astype(np.float32)
        vlon = o.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"
        vlon[:] = advection_kernel.LONS.astype(np.float32)

        def _mk(*, name: str, arr: np.ndarray, units: str, longn: str) -> None:
            v = o.createVariable(name, "f4", ("time", "lat", "lon"),
                                 zlib=True, complevel=4, shuffle=True)
            v.units = units
            v.long_name = longn
            v[:] = arr.astype(np.float32)

        _mk(name="A_obs", arr=A_obs_int, units="m s-1",
            longn="Observed barotropic LWA")
        _mk(name="c_x", arr=c_x_int, units="m s-1",
            longn="Zonal transport velocity F_lambda/A")
        _mk(name="cyp", arr=cyp_int, units="m s-1",
            longn="LN24 northward transport velocity")
        _mk(name="cym", arr=cym_int, units="m s-1",
            longn="LN24 southward transport velocity")
        _mk(name="gamma", arr=gamma_int, units="s-1",
            longn="Damping rate min(R,0)/A")
        _mk(name="eps4", arr=eps4_int, units="m s-2",
            longn="Term III: ep4 (dissipation channel)")
        _mk(name="S_other", arr=S_other_int, units="m s-2",
            longn="Non-condensational positive source max(R,0)-max(LH,0)")
        _mk(name="LH", arr=LH_int, units="m s-2",
            longn="ERA5 LH-LWA source (lh_lwa)")

        vmask = o.createVariable(
            "tc_mask", "i1", ("time", "lat", "lon"),
            zlib=True, complevel=6, shuffle=True,
            fill_value=0,
        )
        vmask.units = "1"
        vmask.long_name = (f"TC disk mask, 1 inside great-circle radius "
                           f"{tc_radius_deg:g} deg of storm centre, 0 outside.")
        vmask.tc_radius_deg = float(tc_radius_deg)
        vmask[:] = tc_mask.astype(np.int8)

        o.storm_id = storm_id
        o.basin = str(row["basin"])
        o.recurv_time = pd.Timestamp(t_recurv).isoformat()
        o.et_time = pd.Timestamp(t_et).isoformat()
        o.window_start = t_start_dt.isoformat()
        o.window_end = t_end_dt.isoformat()
        o.dt_int_min = float(dt_int_min)
        o.tc_radius_deg = float(tc_radius_deg)
        o.closure_native = closure
        o.note = ("Catalog of LN24-decomposed forcing for 2-D LWA "
                  "reconstruction.")

    os.replace(tmp, out_path)
    _LOG.info("Wrote %s", out_path)
    return out_path


def main(
    storm_id: Annotated[str, typer.Option(help="Storm id from the recurving tracks CSV.")],
    tracks_file: Annotated[Path, typer.Option(help="Recurving NH tracks CSV.")],
    individual_tracks_directory: Annotated[Path, typer.Option(
        help="Directory of per-storm track CSVs (<storm_id>.csv).")],
    baro_directory: Annotated[Path, typer.Option(
        help="ERA5 BARO_N archive with yearly files per variable.")],
    lh_source_directory: Annotated[Path, typer.Option(
        help="Monthly latent-heating LWA tendency NetCDFs (YYYY_MM_LWAb_N.nc).")],
    output_directory: Annotated[Path, typer.Option(
        help="Directory for the event_<storm_id>.nc catalog.")],
    t_pre_days: Annotated[float, typer.Option()] = 1.0,
    t_post_days: Annotated[float, typer.Option()] = 8.0,
    dt_int_min: Annotated[float, typer.Option(
        help="Output integration time-step (minutes).")] = 30.0,
    tc_radius_deg: Annotated[float, typer.Option(
        help="Great-circle radius (deg) of the TC mask used to localise "
             "LH / R+ removal in the forward integration.")] = DEFAULT_TC_RADIUS_DEG,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    out = build_event_catalog(
        storm_id=storm_id,
        tracks_file=tracks_file,
        individual_tracks_directory=individual_tracks_directory,
        baro_directory=baro_directory,
        lh_source_directory=lh_source_directory,
        output_directory=output_directory,
        t_pre_days=t_pre_days,
        t_post_days=t_post_days,
        dt_int_min=dt_int_min,
        tc_radius_deg=tc_radius_deg,
        overwrite=overwrite,
    )
    print(out)


if __name__ == "__main__":
    typer.run(main)
