"""Convert archived MERRA2 temperature tendencies (K/s) to dtheta/dt on pressure levels.

Reads daily M2T3NPTDT-style tendency files and matching ASM files (for surface
pressure) from {directory}/{year}/{month:02d}/, applies dtheta/dt =
(p0/p)^kappa * dT/dt, fills below-surface points with iterative
Poisson/Laplace relaxation, and writes one monthly NetCDF
{year}_{month:02d}_{variable}_dtheta_dt.nc under {output_directory}/{year}/.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Annotated

import netCDF4
import numpy as np
import scipy.ndimage
import typer

import downstream_et_lwa.constants as constants

_LOG = logging.getLogger(__name__)

_P0_HPA = constants.REFERENCE_PRESSURE_PA / 100.0


def poisson_fill_level(
    *, field_2d: np.ndarray, mask: np.ndarray, max_iter: int = 500, tol: float = 1e-6
) -> np.ndarray:
    if not mask.any():
        return field_2d.copy()

    filled = field_2d.copy()
    filled[mask] = 0.0

    for _ in range(max_iter):
        lap = scipy.ndimage.laplace(filled)
        update = lap * 0.25
        delta = update[mask]
        filled[mask] += delta
        if np.max(np.abs(delta)) < tol:
            break

    return filled


def poisson_fill_3d(*, data_3d: np.ndarray, plev: np.ndarray, ps_2d: np.ndarray) -> np.ndarray:
    nlev, nlat, nlon = data_3d.shape
    filled = data_3d.copy()

    for k in range(nlev):
        p = plev[k]
        mask_fill = np.abs(data_3d[k]) > 1e10
        mask_underground = ps_2d < p
        mask = mask_fill | mask_underground

        if mask.any():
            slab = filled[k].copy()
            slab[mask] = np.nan
            valid = slab[~mask]
            if len(valid) > 0:
                slab[mask] = np.nanmean(valid)
            filled[k] = poisson_fill_level(field_2d=slab, mask=mask)

    return filled


def process_day(
    *, asm_path: str, tdt_path: str, var_name: str
) -> tuple[np.ndarray, int]:
    with netCDF4.Dataset(asm_path) as fa, netCDF4.Dataset(tdt_path) as ft:
        ps = np.array(fa.variables["PS"][:])
        ps_hpa = ps / 100.0

        plev = np.array(ft.variables["lev"][:])
        dtdt = np.array(ft.variables[var_name][:])

    ntime = dtdt.shape[0]

    exner = (_P0_HPA / plev) ** constants.KAPPA_TWO_SEVENTHS

    dtheta_dt_out = np.zeros_like(dtdt)
    n_filled_total = 0

    for t in range(ntime):
        dtheta = dtdt[t] * exner[:, None, None]
        dtheta_filled = poisson_fill_3d(data_3d=dtheta, plev=plev, ps_2d=ps_hpa[t])

        n_filled = int(np.sum(np.abs(dtdt[t]) > 1e10))
        n_filled_total += n_filled

        dtheta_dt_out[t] = dtheta_filled

    return dtheta_dt_out, n_filled_total


def main(
    year: Annotated[int, typer.Option()],
    month: Annotated[int, typer.Option()],
    tendency_variable: Annotated[str, typer.Option()],
    tendency_directory: Annotated[Path, typer.Option()],
    asm_directory: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    file_glob: Annotated[str, typer.Option()] = "MERRA2*.nc4",
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mm = f"{month:02d}"
    tag = f"{year}_{mm}"
    out_dir = output_directory / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}_{tendency_variable}_dtheta_dt.nc"

    if out_path.exists():
        print(f"Output already exists: {out_path}")
        return

    tdt_dir = tendency_directory / str(year) / mm
    asm_dir = asm_directory / str(year) / mm

    tdt_files = sorted(glob.glob(str(tdt_dir / file_glob)))
    asm_files = sorted(glob.glob(str(asm_dir / file_glob)))

    if len(tdt_files) != len(asm_files):
        print(f"Mismatch: {len(tdt_files)} TDT files vs {len(asm_files)} ASM files")
        raise typer.Exit(code=1)

    print(f"Processing {tendency_variable} {tag} for {len(tdt_files)} days")

    with netCDF4.Dataset(tdt_files[0]) as f0:
        plev = np.array(f0.variables["lev"][:])
        lat = np.array(f0.variables["lat"][:])
        lon = np.array(f0.variables["lon"][:])

    all_data_list = []
    all_times_list = []
    time_units = ""
    total_filled = 0

    for i, (tdt_f, asm_f) in enumerate(zip(tdt_files, asm_files)):
        day = os.path.basename(tdt_f).split(".")[-2]
        _LOG.info("  Day %s (%d/%d)", day, i + 1, len(tdt_files))

        with netCDF4.Dataset(tdt_f) as ft:
            times = np.array(ft.variables["time"][:])
            time_units = ft.variables["time"].units

        dtheta, nf = process_day(asm_path=asm_f, tdt_path=tdt_f, var_name=tendency_variable)
        total_filled += nf
        all_data_list.append(dtheta)
        all_times_list.append(times)

    all_data = np.concatenate(all_data_list, axis=0)
    all_times = np.concatenate(all_times_list, axis=0)

    _LOG.info("Total underground points Poisson-filled: %d", total_filled)
    _LOG.info(
        "Output shape: %s, range: [%.4g, %.4g]",
        all_data.shape,
        all_data.min(),
        all_data.max(),
    )

    print(f"Writing {out_path} ...")
    with netCDF4.Dataset(out_path, "w") as out:
        out.createDimension("time", len(all_times))
        out.createDimension("lev", len(plev))
        out.createDimension("lat", len(lat))
        out.createDimension("lon", len(lon))

        t_var = out.createVariable("time", "f8", ("time",))
        t_var[:] = all_times
        t_var.units = time_units

        l_var = out.createVariable("lev", "f8", ("lev",))
        l_var[:] = plev
        l_var.units = "hPa"

        la_var = out.createVariable("lat", "f8", ("lat",))
        la_var[:] = lat
        la_var.units = "degrees_north"

        lo_var = out.createVariable("lon", "f8", ("lon",))
        lo_var[:] = lon
        lo_var.units = "degrees_east"

        d_var = out.createVariable(
            tendency_variable, "f4", ("time", "lev", "lat", "lon"), zlib=True, complevel=4
        )
        d_var[:] = all_data.astype(np.float32)
        d_var.units = "K/s"
        d_var.long_name = (
            f"Potential temperature tendency from {tendency_variable} (Poisson-filled)"
        )

        out.history = (
            "dT/dt to dtheta/dt conversion with (p0/p)^kappa, "
            f"kappa={constants.KAPPA_TWO_SEVENTHS:.4f}. "
            "Underground points filled using iterative Poisson/Laplace relaxation."
        )

    fsize = os.path.getsize(out_path) / 1e6
    print(f"Done: {out_path} ({fsize:.1f} MB)")


if __name__ == "__main__":
    typer.run(main)
