"""Extract cos(lat)-weighted 20-80N meridional-mean 1-D strips of the LWA budget.

Budget terms from a BARO_N archive: tendency (centred 6-hour finite
difference), Term I = -d(ua1+ua2+ep1)/dx, Term II = (ep2a-ep3a) or (ep2-ep3)
divergence, Term III = ep4, and the residual, plus optional heating-source and
non-QG source strips.  Writes one NetCDF per month named
budget_strips_{source}_{year}_{month:02d}.nc.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime
import logging
import os
from pathlib import Path
from typing import Annotated

import netCDF4
import numpy as np
import typer

import downstream_et_lwa.constants as constants

_LOG = logging.getLogger(__name__)

NLAT = 91
NLON = 360
LATS = np.linspace(0, 90, NLAT)
LONS = np.linspace(0, 359, NLON)
COSPHI = np.cos(np.deg2rad(LATS))

DLAMBDA = np.deg2rad(1.0)
DT_SEC = 6 * 3600
SEC_PER_DAY = 86400.0

LAT_MIN = 20.0
LAT_MAX = 80.0

_BUDGET_VARS = ("lwa", "ua1", "ua2", "ep1", "ep2a", "ep3a", "ep4")


@dataclasses.dataclass(frozen=True)
class StripConfig:
    source_name: str
    baro_directory: Path
    baro_filename_template: str
    baro_time_encoding: str
    varmap: dict[str, str]
    source_terms: dict[str, tuple[Path, str, str]]
    output_directory: Path


def _days_in_month(*, year: int, month: int) -> int:
    if month == 12:
        return (datetime.datetime(year + 1, 1, 1) - datetime.datetime(year, 12, 1)).days
    return (datetime.datetime(year, month + 1, 1) - datetime.datetime(year, month, 1)).days


def _month_times(*, year: int, month: int) -> np.ndarray:
    t0 = datetime.datetime(year, month, 1)
    n_t = _days_in_month(year=year, month=month) * 4
    return np.array(
        [(t0 + datetime.timedelta(hours=6 * i)).timestamp() for i in range(n_t)],
        dtype=np.float64,
    )


def _parse_vm(*, vm: str) -> tuple[str, str]:
    suf, nc_var = vm.split("|")
    return suf, nc_var


def _baro_path(*, cfg: StripConfig, var_suffix: str, year: int, month: int) -> Path:
    return cfg.baro_directory / cfg.baro_filename_template.format(
        year=year, month=month, var=var_suffix
    )


def _read_month_baro(*, cfg: StripConfig, var: str, year: int, month: int) -> np.ndarray | None:
    suf, nc_var = _parse_vm(vm=cfg.varmap[var])
    path = _baro_path(cfg=cfg, var_suffix=suf, year=year, month=month)
    if not path.exists():
        return None
    n_t = _days_in_month(year=year, month=month) * 4
    with netCDF4.Dataset(path, "r") as d:
        v = d[nc_var]
        if cfg.baro_time_encoding == "time_x_month":
            return np.asarray(v[:n_t, month - 1, :, :], dtype=np.float32)
        return np.asarray(v[:n_t, :, :], dtype=np.float32)


def _last_baro_snapshot_prev(
    *, cfg: StripConfig, var: str, year: int, month: int
) -> np.ndarray | None:
    py, pm = (year - 1, 12) if month == 1 else (year, month - 1)
    suf, nc_var = _parse_vm(vm=cfg.varmap[var])
    path = _baro_path(cfg=cfg, var_suffix=suf, year=py, month=pm)
    if not path.exists():
        return None
    n_prev = _days_in_month(year=py, month=pm) * 4
    try:
        with netCDF4.Dataset(path, "r") as d:
            v = d[nc_var]
            if cfg.baro_time_encoding == "time_x_month":
                return np.asarray(v[n_prev - 1, pm - 1, :, :], dtype=np.float32)
            return np.asarray(v[n_prev - 1, :, :], dtype=np.float32)
    except Exception:
        _LOG.exception("Failed reading previous-month snapshot %s", path)
        return None


def _first_baro_snapshot_next(
    *, cfg: StripConfig, var: str, year: int, month: int
) -> np.ndarray | None:
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    suf, nc_var = _parse_vm(vm=cfg.varmap[var])
    path = _baro_path(cfg=cfg, var_suffix=suf, year=ny, month=nm)
    if not path.exists():
        return None
    try:
        with netCDF4.Dataset(path, "r") as d:
            v = d[nc_var]
            if cfg.baro_time_encoding == "time_x_month":
                return np.asarray(v[0, nm - 1, :, :], dtype=np.float32)
            return np.asarray(v[0, :, :], dtype=np.float32)
    except Exception:
        _LOG.exception("Failed reading next-month snapshot %s", path)
        return None


def _read_source_term(
    *, cfg: StripConfig, key: str, year: int, month: int, n_t_expected: int
) -> np.ndarray | None:
    directory, template, nc_var = cfg.source_terms[key]
    path = directory / template.format(year=year, month=month)
    if not path.exists():
        return None
    try:
        with netCDF4.Dataset(path, "r") as d:
            arr = np.asarray(d[nc_var][:], dtype=np.float32)
    except Exception:
        _LOG.exception("Failed reading source term %s from %s", key, path)
        return None
    if arr.shape[0] != n_t_expected:
        if arr.shape[0] > n_t_expected:
            arr = arr[:n_t_expected]
        else:
            pad = np.full(
                (n_t_expected - arr.shape[0], NLAT, NLON), np.nan, dtype=np.float32
            )
            arr = np.concatenate([arr, pad], axis=0)
    return arr


def _zonal_grad(*, field_lat_lon: np.ndarray) -> np.ndarray:
    dx_denom = constants.EARTH_RADIUS_M * COSPHI * DLAMBDA
    dx_denom = np.where(np.abs(dx_denom) < 1e-3, np.nan, dx_denom)
    f = field_lat_lon
    grad = np.empty_like(f)
    grad[..., :, 1:-1] = (f[..., :, 2:] - f[..., :, :-2]) / 2.0
    grad[..., :, 0] = (f[..., :, 1] - f[..., :, -1]) / 2.0
    grad[..., :, -1] = (f[..., :, 0] - f[..., :, -2]) / 2.0
    return grad / dx_denom[:, np.newaxis]


def _merid_mean(*, arr_t_lat_lon: np.ndarray) -> np.ndarray:
    sel = (LATS >= LAT_MIN) & (LATS <= LAT_MAX)
    w = COSPHI[sel]
    return (arr_t_lat_lon[:, sel, :] * w[None, :, None]).sum(axis=1) / w.sum()


def process_month(*, cfg: StripConfig, year: int, month: int, overwrite: bool = False) -> str:
    out_path = cfg.output_directory / f"budget_strips_{cfg.source_name}_{year}_{month:02d}.nc"
    if out_path.exists() and not overwrite:
        return f"[skip] exists: {out_path.name}"
    cfg.output_directory.mkdir(parents=True, exist_ok=True)

    fields = {}
    for v in _BUDGET_VARS:
        arr = _read_month_baro(cfg=cfg, var=v, year=year, month=month)
        if arr is None:
            return f"[err]  {cfg.source_name} {year}-{month:02d}  missing {v}"
        fields[v] = arr
    n_t = fields["lwa"].shape[0]

    lwa = fields["lwa"]
    tendency = np.full_like(lwa, np.nan, dtype=np.float32)
    tendency[1:-1] = (lwa[2:] - lwa[:-2]) / (2.0 * DT_SEC)
    lwa_prev = _last_baro_snapshot_prev(cfg=cfg, var="lwa", year=year, month=month)
    lwa_next = _first_baro_snapshot_next(cfg=cfg, var="lwa", year=year, month=month)
    tendency[0] = (
        (lwa[1] - lwa_prev) / (2.0 * DT_SEC)
        if lwa_prev is not None
        else (lwa[1] - lwa[0]) / DT_SEC
    )
    tendency[-1] = (
        (lwa_next - lwa[-2]) / (2.0 * DT_SEC)
        if lwa_next is not None
        else (lwa[-1] - lwa[-2]) / DT_SEC
    )

    f_lambda = fields["ua1"] + fields["ua2"] + fields["ep1"]
    term_i = -_zonal_grad(field_lat_lon=f_lambda)

    denom = 2.0 * constants.EARTH_RADIUS_M * COSPHI * DLAMBDA
    denom = np.where(np.abs(denom) < 1e-3, np.nan, denom)
    term_ii = (fields["ep2a"] - fields["ep3a"]) / denom[:, np.newaxis]

    term_iii = fields["ep4"].astype(np.float32)

    residual = tendency - term_i - term_ii - term_iii

    src_fields = {}
    for key in cfg.source_terms:
        arr_src = _read_source_term(cfg=cfg, key=key, year=year, month=month, n_t_expected=n_t)
        if arr_src is not None:
            src_fields[key] = arr_src

    strips = {
        "lwa_raw": _merid_mean(arr_t_lat_lon=lwa).astype(np.float32),
        "tendency": _merid_mean(arr_t_lat_lon=tendency).astype(np.float32) * SEC_PER_DAY,
        "termI": _merid_mean(arr_t_lat_lon=term_i).astype(np.float32) * SEC_PER_DAY,
        "termII": _merid_mean(arr_t_lat_lon=term_ii).astype(np.float32) * SEC_PER_DAY,
        "termIII": _merid_mean(arr_t_lat_lon=term_iii).astype(np.float32) * SEC_PER_DAY,
        "residual": _merid_mean(arr_t_lat_lon=residual).astype(np.float32) * SEC_PER_DAY,
    }
    for key, arr_src in src_fields.items():
        strips[key] = _merid_mean(arr_t_lat_lon=arr_src).astype(np.float32) * SEC_PER_DAY

    times_s = _month_times(year=year, month=month)
    tmp_path = out_path.with_suffix(".nc.tmp")
    with netCDF4.Dataset(tmp_path, "w", format="NETCDF4") as out:
        out.createDimension("time", n_t)
        out.createDimension("lon", NLON)
        vt = out.createVariable("time", "f8", ("time",))
        vt.units = "seconds since 1970-01-01"
        vt[:] = times_s
        vlon = out.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"
        vlon[:] = LONS.astype(np.float32)

        def _mk(name: str, arr: np.ndarray, longn: str, units: str) -> None:
            v = out.createVariable(
                name, "f4", ("time", "lon"), zlib=True, complevel=4, shuffle=True
            )
            v.long_name = longn
            v.units = units
            v[:] = arr

        _mk(
            "lwa_raw",
            strips["lwa_raw"],
            f"{cfg.source_name.upper()} raw LWA, cos-lat mean 20-80N",
            "m s-1",
        )
        for key, ln in [
            ("tendency", "dLWA/dt"),
            ("termI", "Term I: -d(ua1+ua2+ep1)/dx"),
            ("termII", "Term II: meridional flux divergence"),
            ("termIII", "Term III: ep4 (dissipation)"),
            ("residual", "Residual = tendency - (I+II+III)"),
        ]:
            _mk(
                key,
                strips[key],
                f"{cfg.source_name.upper()} {ln}, cos-lat mean 20-80N",
                "m s-1 day-1",
            )
        for key in src_fields:
            _mk(
                key,
                strips[key],
                f"{cfg.source_name.upper()} LWA source term {key}, cos-lat mean 20-80N",
                "m s-1 day-1",
            )

        out.source_label = cfg.source_name
        out.baro_dir = str(cfg.baro_directory)
        out.lat_min = np.float32(LAT_MIN)
        out.lat_max = np.float32(LAT_MAX)
        out.method = (
            "Budget per NHN framework; centred finite-difference "
            "tendency; residual closes the budget."
        )

    os.replace(tmp_path, out_path)
    return f"[ok]   {out_path.name}  n_times={n_t}  sources={list(src_fields.keys())}"


def _build_config(
    *,
    source_name: str,
    baro_directory: Path,
    baro_filename_template: str,
    baro_time_encoding: str,
    momentum_flux_style: str,
    heating_directory: Path | None,
    heating_terms: list[str],
    nonqg_directory: Path | None,
    nonqg_filename_template: str,
    nonqg_variable: str,
    output_directory: Path,
) -> StripConfig:
    varmap = {
        "lwa": "LWAb_N|lwa",
        "ua1": "ua1_N|ua1",
        "ua2": "ua2_N|ua2",
        "ep1": "ep1_N|ep1",
        "ep4": "ep4_N|ep4",
    }
    if momentum_flux_style == "split":
        varmap["ep2a"] = "ep2a_N|ep2a"
        varmap["ep3a"] = "ep3a_N|ep3a"
    else:
        varmap["ep2a"] = "ep2_N|ep2"
        varmap["ep3a"] = "ep3_N|ep3"

    source_terms: dict[str, tuple[Path, str, str]] = {}
    if heating_directory is not None:
        for entry in heating_terms:
            name, template, nc_var = entry.split("|")
            source_terms[name] = (heating_directory, template, nc_var)
    if nonqg_directory is not None:
        source_terms["nonqg_lwa"] = (nonqg_directory, nonqg_filename_template, nonqg_variable)

    return StripConfig(
        source_name=source_name,
        baro_directory=baro_directory,
        baro_filename_template=baro_filename_template,
        baro_time_encoding=baro_time_encoding,
        varmap=varmap,
        source_terms=source_terms,
        output_directory=output_directory,
    )


def main(
    baro_directory: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    source_name: Annotated[str, typer.Option()] = "era5",
    baro_filename_template: Annotated[str, typer.Option()] = "{year}_{month:02d}_{var}.nc",
    baro_time_encoding: Annotated[str, typer.Option()] = "time",
    momentum_flux_style: Annotated[str, typer.Option()] = "split",
    heating_directory: Annotated[Path | None, typer.Option()] = None,
    heating_term: Annotated[list[str] | None, typer.Option()] = None,
    nonqg_directory: Annotated[Path | None, typer.Option()] = None,
    nonqg_filename_template: Annotated[str, typer.Option()] = "{year}_{month:02d}_AOUTbaro_N.nc",
    nonqg_variable: Annotated[str, typer.Option()] = "aout_baro",
    year_start: Annotated[int, typer.Option()] = 2000,
    year_end: Annotated[int, typer.Option()] = 2022,
    months: Annotated[list[int] | None, typer.Option()] = None,
    n_workers: Annotated[int, typer.Option()] = 8,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if momentum_flux_style not in ("split", "plain"):
        raise typer.BadParameter("momentum_flux_style must be 'split' or 'plain'")
    if baro_time_encoding not in ("time", "time_x_month"):
        raise typer.BadParameter("baro_time_encoding must be 'time' or 'time_x_month'")
    months_list = months if months else list(range(1, 13))
    heating_terms = (
        heating_term if heating_term else ["lh_lwa|{year}_{month:02d}_LWAb_N.nc|lwa"]
    )

    cfg = _build_config(
        source_name=source_name,
        baro_directory=baro_directory,
        baro_filename_template=baro_filename_template,
        baro_time_encoding=baro_time_encoding,
        momentum_flux_style=momentum_flux_style,
        heating_directory=heating_directory,
        heating_terms=heating_terms,
        nonqg_directory=nonqg_directory,
        nonqg_filename_template=nonqg_filename_template,
        nonqg_variable=nonqg_variable,
        output_directory=output_directory,
    )

    jobs = [(y, m) for y in range(year_start, year_end + 1) for m in months_list]
    print(
        f"[{source_name}] processing {len(jobs)} (year, month) jobs "
        f"with {n_workers} workers."
    )
    print(f"Output dir: {output_directory}")

    if n_workers <= 1:
        for y, m in jobs:
            print(process_month(cfg=cfg, year=y, month=m, overwrite=overwrite), flush=True)
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {
            ex.submit(process_month, cfg=cfg, year=y, month=m, overwrite=overwrite): (y, m)
            for y, m in jobs
        }
        for fut in concurrent.futures.as_completed(futs):
            print(fut.result(), flush=True)


if __name__ == "__main__":
    typer.run(main)
