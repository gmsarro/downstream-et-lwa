"""Convert per-month MPAS QGPV Fortran binaries to compact ~10 km single-level
NetCDF files matching the ERA5 {Y}_{MM}_qgpv.nc schema (qgpv(time, level, lat, lon))."""

from __future__ import annotations

import datetime
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import xarray as xr
from typing_extensions import Annotated

_LOG = logging.getLogger(__name__)

IMAX = 360
JMAX = 181
KMAX = 97
KMAX_TRUNC = 41

DZ = 0.5
HEIGHTS = np.arange(0.5, 0.5 + KMAX * DZ, DZ)[:KMAX]
HEIGHTS_TRUNC = HEIGHTS[:KMAX_TRUNC]

DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}

HEIGHT_LEVEL_IDX = 20
LAT_FULL = np.arange(-90.0, 91.0, 1.0)
LON_FULL = np.arange(0.0, 360.0, 1.0)


def days_in_month(*, year: int, month: int) -> int:
    days = DAYS_IN_MONTH[month]
    if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
        days = 29
    return days


def read_fortran_qgpv(
        *,
        filepath: str,
        year: int,
        month: int,
        truncate_height: bool = True,
) -> tuple[np.ndarray, list[datetime.datetime]]:
    n_days = days_in_month(year=year, month=month)
    n_times = n_days * 4

    kmax = KMAX_TRUNC if truncate_height else KMAX

    qgpv = np.zeros((n_times, kmax, JMAX, IMAX), dtype=np.float32)

    with open(filepath, "rb") as f:
        for t in range(n_times):
            marker1 = np.fromfile(f, dtype="<i4", count=1)
            if len(marker1) == 0:
                _LOG.warning("EOF at timestep %d in %s", t, filepath)
                break

            data = np.fromfile(f, dtype="<f4", count=IMAX * JMAX * KMAX)

            if len(data) != IMAX * JMAX * KMAX:
                _LOG.warning("Incomplete record at timestep %d in %s", t, filepath)
                break

            data = data.reshape((IMAX, JMAX, KMAX), order="F")
            qgpv[t, :, :, :] = np.transpose(data[:, :, :kmax], (2, 1, 0))

            np.fromfile(f, dtype="<i4", count=1)

    start_date = datetime.datetime(year, month, 1, 0, 0)
    times = [start_date + datetime.timedelta(hours=6 * t) for t in range(n_times)]

    return qgpv, times


def _process_one(args: tuple) -> tuple[str, str]:
    mode, year, month, work_root, out_root = args
    tag = f"{year}_{month:02d}"
    src = Path(work_root) / mode / "work" / str(year) / f"{tag}_QGPV"
    if not src.is_file():
        return tag, "missing"
    out_dir = Path(out_root) / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc = out_dir / f"{tag}_qgpv.nc"
    if out_nc.is_file():
        return tag, "skip-exists"
    qgpv, times = read_fortran_qgpv(
        filepath=str(src), year=year, month=month, truncate_height=True)
    qgpv_10km = qgpv[:, HEIGHT_LEVEL_IDX:HEIGHT_LEVEL_IDX + 1, :, :].astype(np.float32)
    z_km = float(HEIGHTS[HEIGHT_LEVEL_IDX])
    ds = xr.Dataset(
        data_vars={
            "qgpv": (
                ("time", "level", "lat", "lon"),
                qgpv_10km,
                {
                    "long_name": "MPAS quasi-geostrophic potential vorticity at ~10 km",
                    "units": "PVU",
                    "height_km": z_km,
                    "source_binary": str(src),
                },
            ),
        },
        coords={
            "time": (
                "time",
                np.array([np.datetime64(t) for t in times], dtype="datetime64[ns]"),
            ),
            "level": ("level", np.array([z_km], dtype=np.float32),
                      {"long_name": "height", "units": "km"}),
            "lat": ("lat", LAT_FULL.astype(np.float32),
                    {"long_name": "latitude", "units": "degrees_north"}),
            "lon": ("lon", LON_FULL.astype(np.float32),
                    {"long_name": "longitude", "units": "degrees_east"}),
        },
        attrs={
            "title": "MPAS QGPV at ~10 km (single level extracted from monthly binary)",
            "scenario": mode,
            "year": int(year),
            "month": int(month),
            "height_km": z_km,
            "source_dir": str(src.parent),
        },
    )
    enc = {"qgpv": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    ds.to_netcdf(out_nc, encoding=enc)
    return tag, f"ok ({qgpv_10km.shape[0]} ts -> {out_nc.name})"


def main(
        work_root: Annotated[Path, typer.Option(
            help="Contains {mode}/work/{Y}/{Y}_{MM}_QGPV binaries")],
        output_directory: Annotated[Path, typer.Option(
            help="QGPV NetCDFs written under "
                 "{output_directory}/{mode}/{Y}_{MM}_qgpv.nc")],
        mode: Annotated[str, typer.Option(
            help="current, future, or both")] = "both",
        workers: Annotated[int, typer.Option()] = 8,
        year_list: Annotated[Optional[str], typer.Option(
            help="Comma-separated subset of years (default: all under work/)")] = None,
        month_list: Annotated[Optional[str], typer.Option(
            help="Comma-separated subset of months (default: all 1-12)")] = None,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())

    modes = ["current", "future"] if mode == "both" else [mode]
    months = ([int(m) for m in month_list.split(",")]
              if month_list else list(range(1, 13)))

    tasks = []
    for md in modes:
        scenario_root = Path(work_root) / md / "work"
        if not scenario_root.is_dir():
            print(f"  [skip] {scenario_root} not found")
            continue
        if year_list:
            years = [int(y) for y in year_list.split(",")]
        else:
            years = sorted(int(p.name) for p in scenario_root.iterdir()
                           if p.is_dir() and p.name.isdigit())
        for y in years:
            for m in months:
                tag = f"{y}_{m:02d}"
                src = Path(work_root) / md / "work" / str(y) / f"{tag}_QGPV"
                if src.is_file():
                    tasks.append((md, y, m, str(work_root), str(output_directory)))

    print(f"Processing {len(tasks)} month files with {workers} workers...")
    with mp.Pool(workers) as pool:
        for i, (tag, status) in enumerate(
                pool.imap_unordered(_process_one, tasks), start=1):
            print(f"  [{i}/{len(tasks)}] {tag}: {status}", flush=True)


if __name__ == "__main__":
    typer.run(main)
