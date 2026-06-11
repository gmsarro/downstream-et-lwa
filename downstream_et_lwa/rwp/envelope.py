"""Zimin/Quinting-Jones Hilbert-transform RWP envelope, threshold, and mask from 250-hPa meridional wind."""
from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Annotated

import netCDF4
import numpy as np
import typer

_LOG = logging.getLogger(__name__)

NLAT = 91
NLON = 360
LATS = np.linspace(0, 90, NLAT)
LONS = np.linspace(0, 359, NLON)
COSPHI = np.cos(np.deg2rad(LATS))

KMIN_DEFAULT = 5
KMAX_DEFAULT = 15
TAU_STAR_DEFAULT = 3.2
INPUT_FILENAME_PATTERN = "{month:02d}_{year}.nc"
OUTPUT_FILENAME_PATTERN = "rwp_envelope_{year}_{month:02d}.nc"


def compute_envelope(*, v_field: np.ndarray, kmin: int = KMIN_DEFAULT,
                     kmax: int = KMAX_DEFAULT) -> np.ndarray:
    nlon = v_field.shape[-1]
    v_clean = np.where(np.isfinite(v_field), v_field, 0.0).astype(np.float64)

    fft_v = np.fft.fft(v_clean, axis=-1)

    keep = np.zeros(nlon, dtype=np.complex128)
    keep[kmin:kmax + 1] = 1.0

    filtered = fft_v * keep
    analytic = np.fft.ifft(2.0 * filtered, axis=-1)
    return np.abs(analytic).astype(np.float32)


def hemispheric_mean(*, field: np.ndarray) -> np.ndarray:
    weights = COSPHI[:, None]
    w_sum = weights.sum() * NLON
    return (field * weights).sum(axis=(-2, -1)) / w_sum


def _process_month(*, year: int, month: int, input_directory: Path,
                   output_directory: Path, kmin: int = KMIN_DEFAULT,
                   kmax: int = KMAX_DEFAULT, tau_star: float = TAU_STAR_DEFAULT,
                   overwrite: bool = False) -> str:
    in_path = input_directory / INPUT_FILENAME_PATTERN.format(month=month, year=year)
    out_path = output_directory / OUTPUT_FILENAME_PATTERN.format(year=year, month=month)

    if not in_path.exists():
        return f"[skip] missing input: {in_path}"
    if out_path.exists() and not overwrite:
        return f"[skip] exists: {out_path.name}"

    output_directory.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(in_path, "r") as ds:
        v = ds.variables["v250"][:]
        t_var = ds.variables["time"]
        times = t_var[:]
        time_units = t_var.units

    v = np.asarray(v, dtype=np.float32)
    nt = v.shape[0]

    env = compute_envelope(v_field=v, kmin=kmin, kmax=kmax)
    hem = hemispheric_mean(field=env)

    tau = (tau_star * hem)[:, None, None]
    mask = (env > tau).astype(np.uint8)
    env_thr = np.where(mask.astype(bool), env, 0.0).astype(np.float32)

    tmp_path = out_path.with_suffix(".nc.tmp")
    with netCDF4.Dataset(tmp_path, "w", format="NETCDF4") as out:
        out.createDimension("time", nt)
        out.createDimension("lat", NLAT)
        out.createDimension("lon", NLON)

        vt = out.createVariable("time", "f8", ("time",))
        vt.units = time_units
        vt[:] = times

        vlat = out.createVariable("lat", "f4", ("lat",))
        vlat.units = "degrees_north"
        vlat[:] = LATS.astype(np.float32)

        vlon = out.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"
        vlon[:] = LONS.astype(np.float32)

        ve = out.createVariable(
            "envelope", "f4", ("time", "lat", "lon"),
            zlib=True, complevel=4, shuffle=True,
            chunksizes=(1, NLAT, NLON))
        ve.long_name = f"RWP envelope (Hilbert transform of v250, wavenumbers {kmin}-{kmax})"
        ve.units = "m s-1"
        ve[:] = env

        vet = out.createVariable(
            "envelope_thr", "f4", ("time", "lat", "lon"),
            zlib=True, complevel=4, shuffle=True,
            chunksizes=(1, NLAT, NLON))
        vet.long_name = f"Thresholded RWP envelope (tau = {tau_star}*<E>_hem)"
        vet.units = "m s-1"
        vet[:] = env_thr

        vm = out.createVariable(
            "mask", "u1", ("time", "lat", "lon"),
            zlib=True, complevel=4, shuffle=True,
            chunksizes=(1, NLAT, NLON))
        vm.long_name = "Binary RWP mask (1 where envelope > tau)"
        vm.units = "1"
        vm[:] = mask

        vh = out.createVariable("hem_mean", "f4", ("time",))
        vh.long_name = "Cos(lat)-weighted hemispheric mean of E"
        vh.units = "m s-1"
        vh[:] = hem.astype(np.float32)

        out.source = f"v250 from {in_path}"
        out.method = ("Zimin et al. 2003 / Quinting & Jones 2016: "
                      "FFT along lat circles, keep k={}-{}, Hilbert "
                      "envelope, threshold tau=tau_star*<E>_hem".format(kmin, kmax))
        out.kmin = np.int32(kmin)
        out.kmax = np.int32(kmax)
        out.tau_star = np.float32(tau_star)

    tmp_path.replace(out_path)
    return f"[ok]   {out_path.name}  n_times={nt}"


def main(
    input_directory: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    year_start: Annotated[int, typer.Option()] = 2000,
    year_end: Annotated[int, typer.Option()] = 2023,
    months: Annotated[list[int] | None, typer.Option()] = None,
    kmin: Annotated[int, typer.Option()] = KMIN_DEFAULT,
    kmax: Annotated[int, typer.Option()] = KMAX_DEFAULT,
    tau_star: Annotated[float, typer.Option()] = TAU_STAR_DEFAULT,
    workers: Annotated[int, typer.Option()] = 8,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    logging.basicConfig(level=logging.INFO)
    month_values = months if months else list(range(1, 13))
    jobs = [(year, month) for year in range(year_start, year_end + 1)
            for month in month_values]

    print(f"Processing {len(jobs)} (year, month) jobs with {workers} workers...")
    print(f"Input dir: {input_directory}")
    print(f"Output dir: {output_directory}")
    print(f"kmin={kmin}, kmax={kmax}, tau_star={tau_star}")

    output_directory.mkdir(parents=True, exist_ok=True)

    if workers <= 1:
        for year, month in jobs:
            print(_process_month(year=year, month=month,
                                 input_directory=input_directory,
                                 output_directory=output_directory,
                                 kmin=kmin, kmax=kmax,
                                 tau_star=tau_star, overwrite=overwrite))
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_process_month, year=year, month=month,
                            input_directory=input_directory,
                            output_directory=output_directory,
                            kmin=kmin, kmax=kmax,
                            tau_star=tau_star, overwrite=overwrite)
            for year, month in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    typer.run(main)
