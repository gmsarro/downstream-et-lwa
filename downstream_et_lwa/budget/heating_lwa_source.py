"""Compute the barotropic LWA tendency from a heating rate with falwa QGFieldNHN22.

Generic driver for any dtheta/dt source (ERA5 latent heating from Dq/Dt,
MERRA2 DTDT* decomposition, MPAS latent heating): for each timestep it reads
u, v, t on pressure levels and the preprocessed heating rate, runs
interpolate_fields, compute_reference_states,
compute_ncforce_from_heating_rate, and compute_lwa_and_barotropic_fluxes, and
stores ncforce_baro in one monthly NetCDF.
"""

from __future__ import annotations

import gc
import glob
import logging
import os
from pathlib import Path
from typing import Annotated

import netCDF4
import numpy as np
import typer

import downstream_et_lwa.constants as constants
import falwa.oopinterface

_LOG = logging.getLogger(__name__)

_FILL_THRESH = 1e10


def _pick_name(*, dataset: netCDF4.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.variables:
            return name
    raise KeyError(f"None of {candidates} found in input file")


def _fix_fill_values(*, arr_3d: np.ndarray) -> np.ndarray:
    arr = arr_3d.copy()
    arr[np.abs(arr) > _FILL_THRESH] = np.nan
    for k in range(arr.shape[0]):
        mask = np.isnan(arr[k])
        if mask.any():
            valid = arr[k][~mask]
            arr[k][mask] = np.nanmean(valid) if len(valid) > 0 else 0.0
    return arr


def main(
    year: Annotated[int, typer.Option()],
    month: Annotated[int, typer.Option()],
    uvt_directory: Annotated[Path, typer.Option()],
    heating_directory: Annotated[Path, typer.Option()],
    heating_variable: Annotated[str, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    uvt_filename_glob: Annotated[str, typer.Option()] = "{month:02d}_{year}.6hrly.nc",
    heating_filename_template: Annotated[
        str, typer.Option()
    ] = "{year}/{year}_{month:02d}_{variable}_dtheta_dt.nc",
    output_filename_template: Annotated[
        str, typer.Option()
    ] = "{year}_{month:02d}_ncforce_baro_{variable}.nc",
    u_variable: Annotated[str, typer.Option()] = "u",
    v_variable: Annotated[str, typer.Option()] = "v",
    t_variable: Annotated[str, typer.Option()] = "t",
    eq_boundary_index: Annotated[int, typer.Option()] = 5,
    kmax: Annotated[int, typer.Option()] = 49,
    dz: Annotated[float, typer.Option()] = 1000.0,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tag = f"{year}_{month:02d}"
    out_path = output_directory / output_filename_template.format(
        year=year, month=month, variable=heating_variable
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"Output already exists: {out_path}")
        return

    heating_path = heating_directory / heating_filename_template.format(
        year=year, month=month, variable=heating_variable
    )
    if not heating_path.exists():
        print(f"Heating file not found: {heating_path}")
        raise typer.Exit(code=1)

    uvt_pattern = str(uvt_directory / uvt_filename_glob.format(year=year, month=month))
    uvt_files = sorted(glob.glob(uvt_pattern))
    if not uvt_files:
        print(f"No u,v,t files match: {uvt_pattern}")
        raise typer.Exit(code=1)

    print(f"Processing {heating_variable} {tag}: {len(uvt_files)} input file(s)")

    with netCDF4.Dataset(uvt_files[0]) as ds0:
        t_name = _pick_name(dataset=ds0, candidates=("valid_time", "time", "Time"))
        p_name = _pick_name(dataset=ds0, candidates=("pressure_level", "level", "lev", "plev"))
        lat_name = _pick_name(dataset=ds0, candidates=("latitude", "lat"))
        lon_name = _pick_name(dataset=ds0, candidates=("longitude", "lon"))
        plev_raw = np.array(ds0.variables[p_name][:], dtype=np.float64)
        lat_raw = np.array(ds0.variables[lat_name][:], dtype=np.float64)
        xlon = np.array(ds0.variables[lon_name][:], dtype=np.float64)

    lat_ascending = bool(lat_raw[0] < lat_raw[-1])
    if not lat_ascending:
        lat_raw = lat_raw[::-1]
    plev_descending = bool(plev_raw[0] > plev_raw[-1])
    if not plev_descending:
        plev_raw = plev_raw[::-1]

    ylat = np.linspace(lat_raw[0], lat_raw[-1], len(lat_raw))
    nlat = len(ylat)
    nlon = len(xlon)

    hmax = -constants.SCALE_HEIGHT_M * np.log(plev_raw[-1] / 1000.0)
    kmax_used = min(kmax, int(hmax // dz) + 1)
    _LOG.info("Vertical extent %.0f m -> kmax=%d (dz=%.0f m)", hmax, kmax_used, dz)

    n_uvt = 0
    for f in uvt_files:
        with netCDF4.Dataset(f) as ds:
            n_uvt += len(ds.variables[t_name][:])

    with netCDF4.Dataset(heating_path) as fd:
        ht_name = _pick_name(dataset=fd, candidates=("valid_time", "time", "Time"))
        hp_name = _pick_name(dataset=fd, candidates=("pressure_level", "level", "lev", "plev"))
        hlat_name = _pick_name(dataset=fd, candidates=("latitude", "lat"))
        n_heating = len(fd.variables[ht_name][:])
        time_units = fd.variables[ht_name].units
        heating_lat = np.array(fd.variables[hlat_name][:])
        heating_plev = np.array(fd.variables[hp_name][:])
    heating_lat_ascending = bool(heating_lat[0] < heating_lat[-1])
    heating_plev_descending = bool(heating_plev[0] > heating_plev[-1])

    nt = min(n_uvt, n_heating)
    _LOG.info("u,v,t: %d timesteps, heating: %d timesteps -> processing %d",
              n_uvt, n_heating, nt)

    ncforce_baro_all = np.zeros((nt, nlat, nlon), dtype=np.float32)
    time_vals = np.zeros(nt, dtype=np.float64)

    with netCDF4.Dataset(heating_path) as fd:
        time_vals[:] = np.array(fd.variables[ht_name][:nt], dtype=np.float64)

        global_t = 0
        for uvt_file in uvt_files:
            if global_t >= nt:
                break
            with netCDF4.Dataset(uvt_file) as ds:
                nt_file = len(ds.variables[t_name][:])
                for ti in range(nt_file):
                    if global_t >= nt:
                        break
                    if global_t % 10 == 0 or global_t == nt - 1:
                        _LOG.info("Timestep %d/%d (%s)", global_t + 1, nt,
                                  os.path.basename(uvt_file))
                    uu = np.array(ds.variables[u_variable][ti], dtype=np.float64)
                    vv = np.array(ds.variables[v_variable][ti], dtype=np.float64)
                    tt = np.array(ds.variables[t_variable][ti], dtype=np.float64)
                    if not lat_ascending:
                        uu = uu[:, ::-1, :]
                        vv = vv[:, ::-1, :]
                        tt = tt[:, ::-1, :]
                    if not plev_descending:
                        uu = uu[::-1, :, :]
                        vv = vv[::-1, :, :]
                        tt = tt[::-1, :, :]
                    uu = _fix_fill_values(arr_3d=uu)
                    vv = _fix_fill_values(arr_3d=vv)
                    tt = _fix_fill_values(arr_3d=tt)

                    dtheta = np.array(
                        fd.variables[heating_variable][global_t], dtype=np.float64
                    )
                    if not heating_lat_ascending:
                        dtheta = dtheta[:, ::-1, :]
                    if not heating_plev_descending:
                        dtheta = dtheta[::-1, :, :]

                    try:
                        qgfield = falwa.oopinterface.QGFieldNHN22(
                            xlon,
                            ylat,
                            plev_raw,
                            uu,
                            vv,
                            tt,
                            kmax=kmax_used,
                            dz=dz,
                            northern_hemisphere_results_only=False,
                            eq_boundary_index=eq_boundary_index,
                        )
                        qgfield.interpolate_fields(return_named_tuple=False)
                        qgfield.compute_reference_states(return_named_tuple=False)
                        ncforce = qgfield.compute_ncforce_from_heating_rate(
                            heating_rate=dtheta
                        )
                        qgfield.compute_lwa_and_barotropic_fluxes(ncforce=ncforce)
                        ncforce_baro_all[global_t] = qgfield.ncforce_baro
                        del qgfield
                    except Exception:
                        _LOG.exception("Timestep %d failed; filling with zeros", global_t)
                        ncforce_baro_all[global_t] = 0.0

                    global_t += 1
            gc.collect()

    print(f"Writing output: {out_path}")
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with netCDF4.Dataset(tmp_path, "w") as out:
        out.createDimension("time", nt)
        out.createDimension("latitude", nlat)
        out.createDimension("longitude", nlon)

        t_var = out.createVariable("time", "f8", ("time",))
        t_var[:] = time_vals
        t_var.units = time_units

        la_var = out.createVariable("latitude", "f8", ("latitude",))
        la_var[:] = ylat
        la_var.units = "degrees_north"

        lo_var = out.createVariable("longitude", "f8", ("longitude",))
        lo_var[:] = xlon
        lo_var.units = "degrees_east"

        lwa_var = out.createVariable(
            "lwa", "f4", ("time", "latitude", "longitude"), zlib=True, complevel=4
        )
        lwa_var[:] = ncforce_baro_all
        lwa_var.units = "m/s"
        lwa_var.long_name = (
            f"Barotropic ncforce from {heating_variable} "
            "(falwa QGFieldNHN22, Lubis et al. 2025)"
        )

        out.history = (
            "falwa compute_ncforce_from_heating_rate + "
            "compute_lwa_and_barotropic_fluxes. "
            f"Input heating rate: {heating_variable}. "
            f"eq_boundary_index={eq_boundary_index}."
        )

    os.replace(tmp_path, out_path)
    fsize = os.path.getsize(out_path) / 1e6
    print(f"Done: {out_path} ({fsize:.1f} MB)")


if __name__ == "__main__":
    typer.run(main)
