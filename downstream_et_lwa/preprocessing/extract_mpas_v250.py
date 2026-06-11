"""Regrid MPAS NH-subset 250-hPa meridional wind to 1-degree NH monthly files, dropping sentinel timesteps."""
from __future__ import annotations

import concurrent.futures
import datetime
import logging
from pathlib import Path
from typing import Annotated

import netCDF4
import numpy as np
import scipy.interpolate
import typer

_LOG = logging.getLogger(__name__)

TGT_LAT = np.linspace(0, 90, 91)
TGT_LON = np.linspace(0, 359, 360)
V_NAME_IN = "umeridional"
V_NAME_OUT = "v250"
PLEV_IDX_DEFAULT = 20
INPUT_FILENAME_PATTERN = "mpas.subset.nh.selvar.{scenario}.{year:04d}-{month:02d}-*.nc"
OUTPUT_FILENAME_PATTERN = "{month:02d}_{year}.nc"

_DT_LO = np.datetime64("1950-01-01")
_DT_HI = np.datetime64("2200-12-31")


def _regrid_slice(*, src_data: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray) -> np.ndarray:
    interpolator = scipy.interpolate.RegularGridInterpolator(
        (src_lat, src_lon), src_data,
        method="linear", bounds_error=False, fill_value=np.nan)
    mesh = np.meshgrid(TGT_LAT, TGT_LON, indexing="ij")
    return interpolator((mesh[0], mesh[1])).astype(np.float32)


def _process_one_month(*, scenario: str, year: int, month: int,
                       input_directory: Path, output_directory: Path,
                       plev_index: int, overwrite: bool) -> str:
    scenario_dir = input_directory / scenario
    if not scenario_dir.is_dir():
        return f"[skip] {scenario}: no dir {scenario_dir}"

    out_dir = output_directory / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_FILENAME_PATTERN.format(month=month, year=year)
    if out_path.exists() and not overwrite:
        return f"[skip] exists {out_path.name}"

    pattern = INPUT_FILENAME_PATTERN.format(scenario=scenario, year=year, month=month)
    paths = sorted(scenario_dir.glob(pattern))
    if not paths:
        return f"[skip] no files {year}-{month:02d} {scenario}"

    v_list: list[np.ndarray] = []
    t_list: list[float] = []
    src_lat: np.ndarray | None = None
    src_lon: np.ndarray | None = None

    dropped: list[tuple[str, int, list[str]]] = []
    for path in paths:
        with netCDF4.Dataset(path, "r") as ds:
            if src_lat is None:
                src_lat = np.asarray(ds["lat"][:], dtype=np.float64)
                src_lon = np.asarray(ds["lon"][:], dtype=np.float64)
            traw = np.asarray(ds["time"][:], dtype=np.float64)
            tunits = ds["time"].units
            v4 = np.asarray(ds[V_NAME_IN][:, plev_index, :, :], dtype=np.float64)

        assert src_lat is not None and src_lon is not None

        tdt = np.asarray(netCDF4.num2date(
            traw, tunits, only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        ))
        fn_date: np.datetime64 | None
        try:
            name_tokens = path.name.split(".")
            date_token = name_tokens[5].split("_")[0]
            fn_date = np.datetime64(date_token)
        except Exception:
            _LOG.exception("Could not parse date from filename %s", path.name)
            fn_date = None

        good = np.ones(len(traw), dtype=bool)
        for ti, t in enumerate(tdt):
            tn = np.datetime64(t)
            if (traw[ti] == 0.0
                    or not np.isfinite(traw[ti])
                    or np.isnat(tn)
                    or tn < _DT_LO or tn > _DT_HI
                    or (fn_date is not None
                        and abs((tn.astype("datetime64[D]") - fn_date)
                                .astype(int)) > 1)):
                good[ti] = False
            elif np.nanmean(np.abs(v4[ti])) < 0.5:
                good[ti] = False

        if not good.all():
            dropped.append((path.name, int((~good).sum()),
                            [str(np.datetime64(t)) for i, t in enumerate(tdt)
                             if not good[i]]))

        for ti in np.where(good)[0]:
            t = tdt[ti]
            if t.tzinfo is None:
                t = t.replace(tzinfo=datetime.timezone.utc)
            v1 = _regrid_slice(src_data=v4[ti], src_lat=src_lat, src_lon=src_lon)
            v_list.append(v1)
            t_list.append(t.timestamp())

    if dropped:
        summary = "; ".join("{}:{} bad ({})".format(name, count, ",".join(values))
                            for name, count, values in dropped)
        _LOG.info("[filter] %s %04d-%02d dropped sentinels: %s",
                  scenario, year, month, summary)

    if not v_list:
        return f"[skip] empty {year}-{month:02d}"

    v_stack = np.stack(v_list, axis=0).astype(np.float32)
    times = np.array(t_list, dtype=np.float64)
    order = np.argsort(times)
    times = times[order]
    v_stack = v_stack[order]
    nt = v_stack.shape[0]

    tmp_path = out_path.with_suffix(".nc.tmp")
    with netCDF4.Dataset(tmp_path, "w", format="NETCDF4") as ds_out:
        ds_out.createDimension("time", nt)
        ds_out.createDimension("lat", len(TGT_LAT))
        ds_out.createDimension("lon", len(TGT_LON))

        vt = ds_out.createVariable("time", "f8", ("time",))
        vt.units = "seconds since 1970-01-01 00:00:00"
        vt.calendar = "standard"
        vt[:] = times

        vlat = ds_out.createVariable("lat", "f4", ("lat",))
        vlat.units = "degrees_north"
        vlat[:] = TGT_LAT.astype(np.float32)

        vlon = ds_out.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"
        vlon[:] = TGT_LON.astype(np.float32)

        vv = ds_out.createVariable(
            V_NAME_OUT, "f4", ("time", "lat", "lon"),
            zlib=True, complevel=4)
        vv.units = "m/s"
        vv.long_name = "Meridional wind at 250 hPa (1-deg NH, from MPAS combined)"
        vv[:] = v_stack

        ds_out.source = f"MPAS combined {scenario}: {paths[0]} ... ({len(paths)} days)"
        ds_out.history = datetime.datetime.now(datetime.timezone.utc).strftime(
            "Created %Y-%m-%dT%H:%MZ extract_mpas_v250.py")

    tmp_path.replace(out_path)
    size_mb = out_path.stat().st_size / 1e6
    return f"[ok] {out_path.name} nt={nt} days={len(paths)} {size_mb:.2f} MB"


def main(
    input_directory: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    scenario: Annotated[str, typer.Option()] = "both",
    year_start: Annotated[int, typer.Option()] = 1988,
    year_end: Annotated[int, typer.Option()] = 2016,
    months: Annotated[list[int] | None, typer.Option()] = None,
    plev_index: Annotated[int, typer.Option()] = PLEV_IDX_DEFAULT,
    workers: Annotated[int, typer.Option()] = 8,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    logging.basicConfig(level=logging.INFO)
    if scenario not in ("current", "future", "both"):
        raise typer.BadParameter("scenario must be one of: current, future, both")

    month_values = months if months else list(range(1, 13))
    scenarios = ["current", "future"] if scenario == "both" else [scenario]
    jobs = [(scen, year, month)
            for scen in scenarios
            for year in range(year_start, year_end + 1)
            for month in month_values]

    print(f"{len(jobs)} month jobs, {workers} workers")
    print(f"out: {output_directory}/<scenario>/MM_YYYY.nc")

    if workers <= 1:
        for scen, year, month in jobs:
            print(_process_one_month(
                scenario=scen, year=year, month=month,
                input_directory=input_directory,
                output_directory=output_directory,
                plev_index=plev_index, overwrite=overwrite))
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_process_one_month,
                            scenario=scen, year=year, month=month,
                            input_directory=input_directory,
                            output_directory=output_directory,
                            plev_index=plev_index, overwrite=overwrite)
            for scen, year, month in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    typer.run(main)
