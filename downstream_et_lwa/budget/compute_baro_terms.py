"""Compute monthly barotropic LWA budget terms (BARO_N layout) with falwa QGFieldNHN22.

Reads monthly NetCDF files of u, v, t on pressure levels, runs the published
falwa pipeline (interpolate_fields, compute_reference_states,
compute_lwa_and_barotropic_fluxes) per timestep, and writes one NetCDF per
budget term named {year}_{month:02d}_{suffix}.nc with suffixes LWAb_N (lwa),
Ub_N (u), Urefb_N (uref), ua1_N (ua1), ua2_N (ua2), ep1_N (ep1), ep25_N
(ep25), ep4_N (ep4), ep2a_N (ep2a), and ep3a_N (ep3a).  Output dimensions are
(time, latitude, longitude) with latitude spanning 0-90N on the input grid
spacing (91 points at 1 degree) and longitude matching the input (360 points
at 1 degree for the canonical archive); Urefb_N is (time, latitude).  ep2a and
ep3a are the northward/southward meridional eddy momentum fluxes of
era4000n_e23a.f90, recomputed from the falwa interpolated u, v and reference
state zonal wind on the pseudo-height grid.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import netCDF4
import numpy as np
import typer

import downstream_et_lwa.constants as constants
import falwa.oopinterface

_LOG = logging.getLogger(__name__)

_FILL_THRESH = 1e10

_SUFFIX_VARIABLES = {
    "LWAb_N": ("lwa", "m s-1", "barotropic local wave activity (lwa_baro)"),
    "Ub_N": ("u", "m s-1", "barotropic zonal wind (u_baro)"),
    "Urefb_N": ("uref", "m s-1", "barotropic reference-state zonal wind"),
    "ua1_N": ("ua1", "m2 s-2", "zonal advective flux F1 (adv_flux_f1)"),
    "ua2_N": ("ua2", "m2 s-2", "zonal advective flux F2 (adv_flux_f2)"),
    "ep1_N": ("ep1", "m2 s-2", "zonal advective flux F3 (adv_flux_f3)"),
    "ep25_N": ("ep25", "m s-2", "divergence of eddy momentum flux"),
    "ep4_N": ("ep4", "m s-2", "low-level meridional heat flux"),
    "ep2a_N": ("ep2a", "m2 s-2", "meridional eddy momentum flux one grid north"),
    "ep3a_N": ("ep3a", "m2 s-2", "meridional eddy momentum flux one grid south"),
}


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


def _split_momentum_fluxes(
    *,
    qgfield: falwa.oopinterface.QGFieldNHN22,
    nhem: int,
    eq_boundary_index: int,
    dz: float,
    kmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    u_nh = qgfield.interpolated_u[:, -nhem:, :]
    v_nh = qgfield.interpolated_v[:, -nhem:, :]
    uref_full = qgfield.uref
    phi = np.deg2rad(np.asarray(qgfield.ylat_ref_states, dtype=np.float64))
    cos2 = np.cos(phi) ** 2
    flux = (u_nh - uref_full[:, :, np.newaxis]) * v_nh * cos2[np.newaxis, :, np.newaxis]
    jb = eq_boundary_index
    ep2a_3d = np.zeros_like(flux)
    ep3a_3d = np.zeros_like(flux)
    ep2a_3d[:, jb : nhem - 1, :] = flux[:, jb + 1 : nhem, :]
    ep3a_3d[:, jb + 1 : nhem - 1, :] = flux[:, jb : nhem - 2, :]
    ep3a_3d[:, jb, :] = flux[:, jb, :]
    height = np.arange(kmax, dtype=np.float64) * dz
    weights = np.exp(-height[1 : kmax - 1] / constants.SCALE_HEIGHT_M) * dz / qgfield.prefactor
    ep2a = np.sum(ep2a_3d[1 : kmax - 1] * weights[:, np.newaxis, np.newaxis], axis=0)
    ep3a = np.sum(ep3a_3d[1 : kmax - 1] * weights[:, np.newaxis, np.newaxis], axis=0)
    return ep2a, ep3a


def _uref_baro(
    *,
    qgfield: falwa.oopinterface.QGFieldNHN22,
    dz: float,
    kmax: int,
) -> np.ndarray:
    height = np.arange(kmax, dtype=np.float64) * dz
    weights = np.exp(-height[1 : kmax - 1] / constants.SCALE_HEIGHT_M) * dz / qgfield.prefactor
    return np.sum(qgfield.uref[1 : kmax - 1] * weights[:, np.newaxis], axis=0)


def _write_term(
    *,
    path: Path,
    var_name: str,
    data: np.ndarray,
    time_vals: np.ndarray,
    time_units: str,
    ylat_nh: np.ndarray,
    xlon: np.ndarray | None,
    units: str,
    long_name: str,
) -> None:
    with netCDF4.Dataset(path, "w", format="NETCDF4") as out:
        out.createDimension("time", data.shape[0])
        out.createDimension("latitude", len(ylat_nh))
        dims: tuple[str, ...] = ("time", "latitude")
        t_var = out.createVariable("time", "f8", ("time",))
        t_var[:] = time_vals
        t_var.units = time_units
        la_var = out.createVariable("latitude", "f8", ("latitude",))
        la_var[:] = ylat_nh
        la_var.units = "degrees_north"
        if xlon is not None:
            out.createDimension("longitude", len(xlon))
            lo_var = out.createVariable("longitude", "f8", ("longitude",))
            lo_var[:] = xlon
            lo_var.units = "degrees_east"
            dims = ("time", "latitude", "longitude")
        v_var = out.createVariable(var_name, "f4", dims, zlib=True, complevel=4)
        v_var[:] = data.astype(np.float32)
        v_var.units = units
        v_var.long_name = long_name
        out.history = "falwa QGFieldNHN22 barotropic LWA budget terms (downstream_et_lwa)"


def main(
    year: Annotated[int, typer.Option()],
    month: Annotated[int, typer.Option()],
    input_directory: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    input_filename_template: Annotated[str, typer.Option()] = "{month:02d}_{year}.6hrly.nc",
    u_variable: Annotated[str, typer.Option()] = "u",
    v_variable: Annotated[str, typer.Option()] = "v",
    t_variable: Annotated[str, typer.Option()] = "t",
    kmax: Annotated[int, typer.Option()] = 97,
    dz: Annotated[float, typer.Option()] = 500.0,
    eq_boundary_index: Annotated[int, typer.Option()] = 5,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    input_path = input_directory / input_filename_template.format(year=year, month=month)
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"Computing BARO_N terms for {year}-{month:02d} from {input_path}")

    with netCDF4.Dataset(input_path) as ds:
        t_name = _pick_name(dataset=ds, candidates=("valid_time", "time", "Time"))
        p_name = _pick_name(dataset=ds, candidates=("pressure_level", "level", "lev", "plev"))
        lat_name = _pick_name(dataset=ds, candidates=("latitude", "lat"))
        lon_name = _pick_name(dataset=ds, candidates=("longitude", "lon"))

        plev_raw = np.array(ds.variables[p_name][:], dtype=np.float64)
        lat_raw = np.array(ds.variables[lat_name][:], dtype=np.float64)
        xlon = np.array(ds.variables[lon_name][:], dtype=np.float64)
        time_vals = np.array(ds.variables[t_name][:], dtype=np.float64)
        time_units = ds.variables[t_name].units

        lat_ascending = bool(lat_raw[0] < lat_raw[-1])
        if not lat_ascending:
            lat_raw = lat_raw[::-1]
        plev_descending = bool(plev_raw[0] > plev_raw[-1])
        if not plev_descending:
            plev_raw = plev_raw[::-1]

        ylat = np.linspace(lat_raw[0], lat_raw[-1], len(lat_raw))
        nlat = len(ylat)
        nhem = nlat // 2 + 1
        ylat_nh = ylat[-nhem:]
        nt = len(time_vals)

        hmax = -constants.SCALE_HEIGHT_M * np.log(plev_raw[-1] / 1000.0)
        kmax_used = min(kmax, int(hmax // dz) + 1)
        _LOG.info("Vertical extent %.0f m -> kmax=%d (dz=%.0f m)", hmax, kmax_used, dz)

        results: dict[str, np.ndarray] = {
            suffix: np.zeros(
                (nt, nhem) if suffix == "Urefb_N" else (nt, nhem, len(xlon)),
                dtype=np.float32,
            )
            for suffix in _SUFFIX_VARIABLES
        }

        for ti in range(nt):
            if ti % 10 == 0 or ti == nt - 1:
                _LOG.info("Timestep %d/%d", ti + 1, nt)
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
                    northern_hemisphere_results_only=True,
                    eq_boundary_index=eq_boundary_index,
                )
                qgfield.interpolate_fields(return_named_tuple=False)
                qgfield.compute_reference_states(return_named_tuple=False)
                qgfield.compute_lwa_and_barotropic_fluxes(return_named_tuple=False)

                results["LWAb_N"][ti] = qgfield.lwa_baro
                results["Ub_N"][ti] = qgfield.u_baro
                results["Urefb_N"][ti] = _uref_baro(qgfield=qgfield, dz=dz, kmax=kmax_used)
                results["ua1_N"][ti] = qgfield.adv_flux_f1
                results["ua2_N"][ti] = qgfield.adv_flux_f2
                results["ep1_N"][ti] = qgfield.adv_flux_f3
                results["ep25_N"][ti] = qgfield.divergence_eddy_momentum_flux
                results["ep4_N"][ti] = qgfield.meridional_heat_flux
                ep2a, ep3a = _split_momentum_fluxes(
                    qgfield=qgfield,
                    nhem=nhem,
                    eq_boundary_index=eq_boundary_index,
                    dz=dz,
                    kmax=kmax_used,
                )
                results["ep2a_N"][ti] = ep2a
                results["ep3a_N"][ti] = ep3a
                del qgfield
            except Exception:
                _LOG.exception("Timestep %d failed; filling with zeros", ti)

    for suffix, (var_name, units, long_name) in _SUFFIX_VARIABLES.items():
        out_path = output_directory / f"{year}_{month:02d}_{suffix}.nc"
        _write_term(
            path=out_path,
            var_name=var_name,
            data=results[suffix],
            time_vals=time_vals,
            time_units=time_units,
            ylat_nh=ylat_nh,
            xlon=None if suffix == "Urefb_N" else xlon,
            units=units,
            long_name=long_name,
        )
        _LOG.info("Wrote %s", out_path)
    print(f"Done: {year}-{month:02d} -> {output_directory}")


if __name__ == "__main__":
    typer.run(main)
