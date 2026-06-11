"""Reproject one month of MPAS raw output to ERA5-style 1-degree, 37-pressure-level u, v, w, t, z files."""
from __future__ import annotations

import calendar
import gc
import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
import xarray as xr

import downstream_et_lwa.constants as constants

_LOG = logging.getLogger(__name__)

NEW_LAT = np.arange(-90, 91, 1, dtype=np.float64)
NEW_LON = np.arange(0, 360, 1, dtype=np.float64)
ERA5_LEVELS = np.array([
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700,
    650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175,
    150, 125, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1
])
ERA5_LEVS_SORTED = np.sort(ERA5_LEVELS).astype(np.float64)
NPAD = 5
INPUT_FILENAME_PATTERN = "mpas.subset.nh.selvar.{mode}.{year}-{month:02d}-*.nc"

MPAS_RENAME = {
    "uzonal": "u", "umeridional": "v", "w": "w",
    "temperature": "t", "height": "z",
}
MPAS_ORIG = {v: k for k, v in MPAS_RENAME.items()}
LAT_DESC_IDX = np.argsort(NEW_LAT)[::-1]


def _interpolate_one_var_one_file(*, file_path: Path, varname: str) -> tuple[np.ndarray, np.ndarray]:
    orig_name = MPAS_ORIG[varname]
    with xr.open_dataset(file_path) as ds:
        lev = ds["lev"].values
        if lev.max() > 1200:
            lev = lev / 100.0
        lon = np.mod(ds["lon"].values.astype(np.float64), 360.0)
        lat = ds["lat"].values
        time_vals = ds["time"].values.copy()
        raw = ds[orig_name].values
    gc.collect()

    da = xr.DataArray(
        raw, dims=["time", "lev", "lat", "lon"],
        coords={"time": time_vals, "lev": lev, "lat": lat, "lon": lon},
    )
    del raw
    da = da.sortby("lon")

    if varname == "z":
        da = da * constants.GRAVITY

    lon_sorted = da["lon"].values
    left = da.isel(lon=slice(-NPAD, None)).assign_coords(lon=lon_sorted[-NPAD:] - 360.0)
    right = da.isel(lon=slice(None, NPAD)).assign_coords(lon=lon_sorted[:NPAD] + 360.0)
    da = xr.concat([left, da, right], dim="lon")

    da = da.interp(lat=NEW_LAT, lon=NEW_LON, method="linear")
    da = da.bfill(dim="lat").ffill(dim="lat")
    da.loc[dict(lat=90)] = da.sel(lat=89)
    da = da.interp(lev=ERA5_LEVS_SORTED, method="linear")

    vals = da.values.copy()
    del da
    gc.collect()

    for k in range(vals.shape[1]):
        if np.all(np.isnan(vals[:, k, :, :])):
            for k2 in range(k + 1, vals.shape[1]):
                if not np.all(np.isnan(vals[:, k2, :, :])):
                    vals[:, k, :, :] = vals[:, k2, :, :]
                    break
    np.nan_to_num(vals, copy=False, nan=0.0)

    vals = vals[:, :, LAT_DESC_IDX, :]
    vals[:, :, :, 0] = vals[:, :, :, 1]

    return vals, time_vals


def _process_month(*, mode: str, input_directory: Path, year: int, month: int,
                   output_directory: Path) -> bool:
    output_directory.mkdir(parents=True, exist_ok=True)
    out_tag = f"{year}_{month:02d}"

    all_exist = all(
        (output_directory / f"{out_tag}_{v}.nc").is_file()
        for v in ("u", "v", "w", "t", "z")
    )
    if all_exist:
        _LOG.info("[SKIP] All outputs exist for %s", out_tag)
        return True

    pattern = INPUT_FILENAME_PATTERN.format(mode=mode, year=year, month=month)
    file_list = sorted(input_directory.glob(pattern))
    if not file_list:
        _LOG.info("[SKIP] No raw files for %04d-%02d in %s", year, month, input_directory)
        return False

    n_files = len(file_list)
    n_days = calendar.monthrange(year, month)[1]
    n_full_days = n_days
    if n_files < n_days:
        n_full_days = n_files
        _LOG.info("PARTIAL MONTH: %d/%d days available, writing %d timesteps",
                  n_files, n_days, n_files * 4)
    full_times = pd.date_range(
        f"{year}-{month:02d}-01",
        f"{year}-{month:02d}-{n_full_days} 18:00", freq="6h"
    )

    lat_desc = NEW_LAT[::-1].copy()
    saved_t: np.ndarray | None = None

    for varname in ("u", "v", "t", "z", "w"):
        out_file = output_directory / f"{out_tag}_{varname}.nc"
        if out_file.is_file():
            if varname == "t":
                with xr.open_dataset(out_file) as ds_existing:
                    saved_t = ds_existing[varname].values.astype(np.float64)
            _LOG.info("[SKIP] %s exists", out_file)
            continue

        _LOG.info("Variable '%s': processing %d files", varname, n_files)

        all_vals: list[np.ndarray] = []
        all_times: list[np.datetime64] = []
        for idx, file_path in enumerate(file_list):
            if (idx + 1) % 5 == 0 or idx == 0 or idx == n_files - 1:
                _LOG.info("[%d/%d]", idx + 1, n_files)
            vals, times = _interpolate_one_var_one_file(file_path=file_path, varname=varname)
            all_vals.append(vals)
            all_times.extend(times)
            del vals
            gc.collect()

        data = np.concatenate(all_vals, axis=0)
        del all_vals
        gc.collect()

        time_idx = pd.DatetimeIndex(all_times)
        if getattr(time_idx, "tz", None) is not None:
            time_idx = time_idx.tz_localize(None)
        tgt_times = pd.DatetimeIndex(full_times)
        da = xr.DataArray(
            data, dims=["time", "lev", "lat", "lon"],
            coords={
                "time": time_idx,
                "lev": ERA5_LEVS_SORTED,
                "lat": lat_desc,
                "lon": NEW_LON,
            },
            name=varname,
        )
        da = da.groupby("time").mean(skipna=True)
        da = da.sortby("time")
        da = da.interp(time=tgt_times, method="linear")
        da = da.interpolate_na(dim="time", method="linear")
        da = da.ffill(dim="time").bfill(dim="time")

        if varname == "t":
            saved_t = da.values.astype(np.float64)

        if varname == "w":
            _LOG.info("Converting w to omega")
            w_f64 = da.values.astype(np.float64)
            lev_pa = ERA5_LEVS_SORTED.astype(np.float64) * 100.0
            if saved_t is None:
                t_file = output_directory / f"{out_tag}_t.nc"
                with xr.open_dataset(t_file) as ds_t:
                    saved_t = ds_t["t"].values.astype(np.float64)
            lev_bc = lev_pa[np.newaxis, :, np.newaxis, np.newaxis]
            omega = -w_f64 * constants.GRAVITY * lev_bc / (constants.DRY_GAS_CONSTANT * saved_t)
            omega = np.where(saved_t != 0, omega, 0.0)
            da = xr.DataArray(
                omega, dims=["time", "lev", "lat", "lon"],
                coords=da.coords, name="w",
            )
            del w_f64, omega
            gc.collect()
        else:
            da = da.astype(np.float32)

        da.attrs["scale_factor"] = 1.0
        da.attrs["add_offset"] = 0.0
        da.to_netcdf(out_file)
        _LOG.info("Saved %s", out_file)
        del da, data
        gc.collect()

    del saved_t
    gc.collect()
    _LOG.info("Done %04d-%02d", year, month)
    return True


def main(
    mode: Annotated[str, typer.Option()],
    input_directory: Annotated[Path, typer.Option()],
    year: Annotated[int, typer.Option()],
    month: Annotated[int, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
) -> None:
    logging.basicConfig(level=logging.INFO)
    if mode not in ("current", "future"):
        raise typer.BadParameter("mode must be one of: current, future")
    print(f"=== Reprojecting {mode} {year}-{month:02d} ===", flush=True)
    print(f"  Raw dir:    {input_directory}", flush=True)
    print(f"  Output dir: {output_directory}", flush=True)
    success = _process_month(mode=mode, input_directory=input_directory,
                             year=year, month=month,
                             output_directory=output_directory)
    if not success:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
