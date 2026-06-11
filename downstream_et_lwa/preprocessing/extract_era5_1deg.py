"""Extract ERA5 u, v, t at 250/500/850 hPa from 0.25-degree monthly files and regrid to 1-degree NH."""
from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing
from pathlib import Path
from typing import Annotated

import netCDF4
import numpy as np
import scipy.interpolate
import typer

_LOG = logging.getLogger(__name__)

TARGET_LEVELS: dict[int, int] = {250: 20, 500: 15, 850: 6}
TARGET_VARS = ("u", "v", "t")
TGT_LAT = np.linspace(0, 90, 91)
TGT_LON = np.linspace(0, 359, 360)
INPUT_FILENAME_PATTERN = "{month:02d}_{year}.6hrly.nc"
OUTPUT_FILENAME_PATTERN = "{month:02d}_{year}.nc"


def _regrid_slice(*, src_data: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray) -> np.ndarray:
    interpolator = scipy.interpolate.RegularGridInterpolator(
        (src_lat, src_lon), src_data,
        method="linear", bounds_error=False, fill_value=np.nan)
    mesh = np.meshgrid(TGT_LAT, TGT_LON, indexing="ij")
    return interpolator((mesh[0], mesh[1])).astype(np.float32)


def _process_month(*, year: int, month: int, input_directory: Path,
                   output_directory: Path) -> str:
    src_path = input_directory / INPUT_FILENAME_PATTERN.format(month=month, year=year)
    if not src_path.exists():
        return f"SKIP {year}-{month:02d}: no source file"

    out_path = output_directory / OUTPUT_FILENAME_PATTERN.format(month=month, year=year)
    if out_path.exists():
        return f"SKIP {year}-{month:02d}: already extracted"

    try:
        with netCDF4.Dataset(src_path, "r") as ds_in:
            src_lat = np.array(ds_in["latitude"][:], dtype=np.float64)
            src_lon = np.array(ds_in["longitude"][:], dtype=np.float64)
            ntimes = ds_in.dimensions["valid_time"].size

            time_var = ds_in["valid_time"]
            time_vals = np.array(time_var[:])
            time_units = time_var.units if hasattr(time_var, "units") else "seconds since 1970-01-01"

            with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ds_out:
                ds_out.createDimension("time", ntimes)
                ds_out.createDimension("lat", len(TGT_LAT))
                ds_out.createDimension("lon", len(TGT_LON))

                v_time = ds_out.createVariable("time", "f8", ("time",))
                v_time[:] = time_vals
                v_time.units = time_units

                v_lat = ds_out.createVariable("lat", "f4", ("lat",))
                v_lat[:] = TGT_LAT
                v_lat.units = "degrees_north"

                v_lon = ds_out.createVariable("lon", "f4", ("lon",))
                v_lon[:] = TGT_LON
                v_lon.units = "degrees_east"

                for var_name in TARGET_VARS:
                    src_var = ds_in[var_name]
                    for plev_hpa, plev_idx in TARGET_LEVELS.items():
                        out_name = f"{var_name}{plev_hpa}"
                        v_out = ds_out.createVariable(
                            out_name, "f4", ("time", "lat", "lon"),
                            zlib=True, complevel=4)
                        v_out.units = "m/s" if var_name in ("u", "v") else "K"
                        v_out.long_name = f"{var_name} at {plev_hpa} hPa (1-deg NH)"

                        for ti in range(ntimes):
                            raw = np.array(src_var[ti, plev_idx, :, :], dtype=np.float64)
                            v_out[ti, :, :] = _regrid_slice(
                                src_data=raw, src_lat=src_lat, src_lon=src_lon)

        size_mb = out_path.stat().st_size / 1e6
        return f"OK   {year}-{month:02d}: {ntimes} times, {size_mb:.1f} MB"

    except Exception as exc:
        _LOG.exception("Extraction failed for %04d-%02d", year, month)
        if out_path.exists():
            out_path.unlink()
        return f"FAIL {year}-{month:02d}: {exc}"


def main(
    input_directory: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    year_start: Annotated[int, typer.Option()] = 2000,
    year_end: Annotated[int, typer.Option()] = 2024,
    workers: Annotated[int, typer.Option()] = 20,
) -> None:
    logging.basicConfig(level=logging.INFO)
    output_directory.mkdir(parents=True, exist_ok=True)

    jobs = [(year, month) for year in range(year_start, year_end + 1)
            for month in range(1, 13)]
    print(f"Extracting {len(jobs)} month-files, {workers} workers", flush=True)

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork")) as pool:
        futures = [
            pool.submit(_process_month, year=year, month=month,
                        input_directory=input_directory,
                        output_directory=output_directory)
            for year, month in jobs
        ]
        for future in futures:
            print(f"  {future.result()}", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    typer.run(main)
