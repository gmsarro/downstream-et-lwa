"""Convert Dq/Dt to a latent-heating potential-temperature tendency.

theta_dot_L = -(Lv/cp) * (p0/p)^kappa * Dq/Dt on the native pressure grid,
following the sandro_like ERA5 preprocessing: the conversion is applied to the
full Dq/Dt field by default (pass --condensation-only to zero grid points with
Dq/Dt >= 0), and NaN/underground points are filled with iterative
Poisson/Laplace relaxation using a conservative surface-pressure estimate of
max(plev) + 50 hPa.
"""

from __future__ import annotations

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


def _pick_name(*, dataset: netCDF4.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.variables:
            return name
    raise KeyError(f"None of {candidates} found in input file")


def poisson_fill_level(
    *, field_2d: np.ndarray, mask: np.ndarray, max_iter: int = 500, tol: float = 1e-6
) -> np.ndarray:
    if not mask.any():
        return field_2d.copy()

    filled = field_2d.copy()
    valid = filled[~mask]
    if len(valid) > 0:
        filled[mask] = np.nanmean(valid)
    else:
        filled[mask] = 0.0

    for _ in range(max_iter):
        lap = scipy.ndimage.laplace(filled)
        delta = lap[mask] * 0.25
        filled[mask] += delta
        if np.max(np.abs(delta)) < tol:
            break

    return filled


def poisson_fill_3d(
    *, data_3d: np.ndarray, plev_hpa: np.ndarray, ps_hpa: np.ndarray
) -> np.ndarray:
    nlev, nlat, nlon = data_3d.shape
    filled = data_3d.copy()

    for k in range(nlev):
        p = plev_hpa[k]
        mask = np.isnan(filled[k]) | (np.abs(filled[k]) > 1e10) | (ps_hpa < p)
        if mask.any():
            slab = filled[k].copy()
            slab[mask] = np.nan
            valid = slab[~mask]
            if len(valid) > 0:
                slab[mask] = np.nanmean(valid)
            filled[k] = poisson_fill_level(field_2d=slab, mask=mask)

    return filled


def main(
    dqdt_file: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    dqdt_variable: Annotated[str, typer.Option()] = "DqDt",
    output_variable: Annotated[str, typer.Option()] = "DQDT",
    output_filename: Annotated[str | None, typer.Option()] = None,
    condensation_only: Annotated[bool, typer.Option()] = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_directory.mkdir(parents=True, exist_ok=True)
    out_path = output_directory / (
        output_filename if output_filename else f"{dqdt_file.stem}_dtheta_dt.nc"
    )
    if out_path.exists():
        print(f"Output already exists: {out_path}")
        return
    print(f"Opening Dq/Dt file: {dqdt_file}")

    with netCDF4.Dataset(dqdt_file) as ds:
        t_name = _pick_name(dataset=ds, candidates=("valid_time", "time", "Time"))
        p_name = _pick_name(dataset=ds, candidates=("pressure_level", "level", "lev", "plev"))
        lat_name = "latitude" if "latitude" in ds.variables else "lat"
        lon_name = "longitude" if "longitude" in ds.variables else "lon"

        time_vals = np.array(ds.variables[t_name][:], dtype=np.float64)
        time_units = ds.variables[t_name].units
        time_cal = getattr(ds.variables[t_name], "calendar", "standard")

        p_raw = np.array(ds.variables[p_name][:], dtype=np.float64)
        lat = np.array(ds.variables[lat_name][:], dtype=np.float64)
        lon = np.array(ds.variables[lon_name][:], dtype=np.float64)

        if np.max(p_raw) < 2000:
            p_hpa = p_raw
        else:
            p_hpa = p_raw / 100.0

        nt = len(time_vals)
        nlev = len(p_hpa)
        nlat = len(lat)
        nlon = len(lon)
        _LOG.info("Grid: %d times, %d levels, %d lat, %d lon", nt, nlev, nlat, nlon)

        exner = (_P0_HPA / p_hpa) ** constants.KAPPA_TWO_SEVENTHS
        ps_hpa_approx = np.full((nlat, nlon), np.max(p_hpa) + 50.0, dtype=np.float64)

        tmp_path = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
        total_filled = 0
        with netCDF4.Dataset(tmp_path, "w", format="NETCDF4") as ncout:
            ncout.createDimension("time", nt)
            ncout.createDimension("lev", nlev)
            ncout.createDimension("lat", nlat)
            ncout.createDimension("lon", nlon)

            t_out = ncout.createVariable("time", "f8", ("time",))
            t_out.units = time_units
            t_out.calendar = time_cal
            t_out[:] = time_vals

            p_out = ncout.createVariable("lev", "f8", ("lev",))
            p_out.units = "hPa"
            p_out[:] = p_hpa

            la_out = ncout.createVariable("lat", "f8", ("lat",))
            la_out.units = "degrees_north"
            la_out[:] = lat

            lo_out = ncout.createVariable("lon", "f8", ("lon",))
            lo_out.units = "degrees_east"
            lo_out[:] = lon

            d_var = ncout.createVariable(
                output_variable, "f4", ("time", "lev", "lat", "lon"), zlib=True, complevel=4
            )
            d_var.units = "K/s"
            d_var.long_name = (
                "Potential temperature tendency from latent heating "
                "(DqDt-based, Poisson-filled)"
            )

            for ti in range(nt):
                if ti % 10 == 0 or ti == nt - 1:
                    _LOG.info("Timestep %d/%d", ti + 1, nt)
                dqdt = np.array(ds.variables[dqdt_variable][ti], dtype=np.float64)
                if condensation_only:
                    dqdt = np.where(dqdt < 0.0, dqdt, 0.0)
                dtheta_3d = (
                    -(constants.LATENT_HEAT_VAPORIZATION / constants.SPECIFIC_HEAT_PRESSURE)
                    * exner[:, np.newaxis, np.newaxis]
                    * dqdt
                )
                total_filled += int(np.isnan(dtheta_3d).sum())
                dtheta_filled = poisson_fill_3d(
                    data_3d=dtheta_3d, plev_hpa=p_hpa, ps_hpa=ps_hpa_approx
                )
                d_var[ti] = dtheta_filled.astype(np.float32)

            ncout.history = (
                "dtheta/dt = -(Lv/cp)*(p0/p)^kappa*DqDt. "
                f"Lv={constants.LATENT_HEAT_VAPORIZATION:.0f}, "
                f"cp={constants.SPECIFIC_HEAT_PRESSURE:.0f}, "
                f"kappa={constants.KAPPA_TWO_SEVENTHS:.4f}, "
                f"condensation_only={condensation_only}. "
                "Underground/NaN points Poisson-filled. "
                f"Source: {dqdt_file}"
            )

    os.replace(tmp_path, out_path)
    _LOG.info("NaN/underground points filled: %d", total_filled)
    print(f"Done: {out_path}")


if __name__ == "__main__":
    typer.run(main)
