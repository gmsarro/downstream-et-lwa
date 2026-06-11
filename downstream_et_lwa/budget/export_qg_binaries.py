"""Export falwa QGFieldNHN22 fields as the Fortran unformatted binaries of the ageo chain.

Writes, per (year, month), the sequential unformatted files QGPV, QGU, QGV,
QGT, QGZ, QVORT, and QGREF_N consumed by fortran/ageo/ageo_lwa_source.f90.
One record per timestep, 4-byte record markers, float32 payloads:
QGPV/QGU/QGV/QGZ/QVORT hold (imax, jmax, kmax) global fields on the regular
pseudo-height grid; QGT holds pt(imax, jmax, kmax), tn0(kmax), ts0(kmax),
statn(kmax), stats(kmax); QGREF_N holds qref(nd, kmax), uref(jd, kmax),
tref(jd, kmax), fawa(nd, kmax), ubar(nd, kmax), tbar(nd, kmax) with
nd = jmax//2 + 1 and jd = nd - eq_boundary_index.  QGZ requires a geopotential
(or geopotential-height) variable in the input file and is interpolated
linearly in pseudo-height; QVORT is absolute vorticity computed by centred
differences from the falwa-interpolated winds; fawa is not exposed by falwa
and is written as zeros (it is read but never used by the ageo Fortran).
"""

from __future__ import annotations

import contextlib
import logging
import struct
from pathlib import Path
from typing import Annotated, BinaryIO

import netCDF4
import numpy as np
import typer

import downstream_et_lwa.constants as constants
import falwa.oopinterface

_LOG = logging.getLogger(__name__)

_FILL_THRESH = 1e10
_BINARY_NAMES = ("QGPV", "QGU", "QGV", "QGT", "QGZ", "QVORT", "QGREF_N")


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


def _write_record(*, fh: BinaryIO, arrays: list[np.ndarray]) -> None:
    payload = b"".join(np.ascontiguousarray(a, dtype="<f4").tobytes() for a in arrays)
    marker = struct.pack("<i", len(payload))
    fh.write(marker)
    fh.write(payload)
    fh.write(marker)


def _interp_to_height(
    *, field_plev: np.ndarray, plev_hpa: np.ndarray, kmax: int, dz: float
) -> np.ndarray:
    zlev = -constants.SCALE_HEIGHT_M * np.log(plev_hpa / 1000.0)
    zgrid = np.arange(kmax, dtype=np.float64) * dz
    i1 = np.clip(np.searchsorted(zlev, zgrid), 1, len(zlev) - 1)
    i0 = i1 - 1
    w = (zgrid - zlev[i0]) / (zlev[i1] - zlev[i0])
    w = np.clip(w, 0.0, 1.0)
    return (
        field_plev[i0] * (1.0 - w)[:, np.newaxis, np.newaxis]
        + field_plev[i1] * w[:, np.newaxis, np.newaxis]
    )


def _absolute_vorticity(
    *, u3: np.ndarray, v3: np.ndarray, ylat: np.ndarray, planet_radius: float, omega: float
) -> np.ndarray:
    phi = np.deg2rad(ylat)
    dphi = phi[1] - phi[0]
    nlon = u3.shape[2]
    dlam = 2.0 * np.pi / nlon
    cosphi = np.cos(phi)
    cosphi_safe = np.where(np.abs(cosphi) < 1e-12, 1e-12, cosphi)
    dvdx = (np.roll(v3, -1, axis=2) - np.roll(v3, 1, axis=2)) / (2.0 * dlam)
    ucos = u3 * cosphi[np.newaxis, :, np.newaxis]
    ducos_dphi = np.empty_like(u3)
    ducos_dphi[:, 1:-1, :] = (ucos[:, 2:, :] - ucos[:, :-2, :]) / (2.0 * dphi)
    ducos_dphi[:, 0, :] = (ucos[:, 1, :] - ucos[:, 0, :]) / dphi
    ducos_dphi[:, -1, :] = (ucos[:, -1, :] - ucos[:, -2, :]) / dphi
    zeta = (dvdx - ducos_dphi) / (planet_radius * cosphi_safe[np.newaxis, :, np.newaxis])
    zeta[:, 0, :] = zeta[:, 1, :].mean(axis=-1, keepdims=True)
    zeta[:, -1, :] = zeta[:, -2, :].mean(axis=-1, keepdims=True)
    f_cor = 2.0 * omega * np.sin(phi)
    return zeta + f_cor[np.newaxis, :, np.newaxis]


def main(
    year: Annotated[int, typer.Option()],
    month: Annotated[int, typer.Option()],
    input_directory: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    input_filename_template: Annotated[str, typer.Option()] = "{month:02d}_{year}.6hrly.nc",
    u_variable: Annotated[str, typer.Option()] = "u",
    v_variable: Annotated[str, typer.Option()] = "v",
    t_variable: Annotated[str, typer.Option()] = "t",
    z_variable: Annotated[str | None, typer.Option()] = None,
    z_is_geopotential: Annotated[bool, typer.Option()] = True,
    kmax: Annotated[int, typer.Option()] = 97,
    dz: Annotated[float, typer.Option()] = 500.0,
    eq_boundary_index: Annotated[int, typer.Option()] = 5,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    input_path = input_directory / input_filename_template.format(year=year, month=month)
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"Exporting QG binaries for {year}-{month:02d} from {input_path}")

    names = list(_BINARY_NAMES)
    if z_variable is None:
        names.remove("QGZ")
        _LOG.warning("No z variable supplied; QGZ will not be written")

    with netCDF4.Dataset(input_path) as ds:
        t_name = _pick_name(dataset=ds, candidates=("valid_time", "time", "Time"))
        p_name = _pick_name(dataset=ds, candidates=("pressure_level", "level", "lev", "plev"))
        lat_name = _pick_name(dataset=ds, candidates=("latitude", "lat"))
        lon_name = _pick_name(dataset=ds, candidates=("longitude", "lon"))

        plev_raw = np.array(ds.variables[p_name][:], dtype=np.float64)
        lat_raw = np.array(ds.variables[lat_name][:], dtype=np.float64)
        xlon = np.array(ds.variables[lon_name][:], dtype=np.float64)
        nt = len(ds.variables[t_name][:])

        lat_ascending = bool(lat_raw[0] < lat_raw[-1])
        if not lat_ascending:
            lat_raw = lat_raw[::-1]
        plev_descending = bool(plev_raw[0] > plev_raw[-1])
        if not plev_descending:
            plev_raw = plev_raw[::-1]

        ylat = np.linspace(lat_raw[0], lat_raw[-1], len(lat_raw))
        nlat = len(ylat)
        nhem = nlat // 2 + 1
        jb = eq_boundary_index

        hmax = -constants.SCALE_HEIGHT_M * np.log(plev_raw[-1] / 1000.0)
        kmax_used = min(kmax, int(hmax // dz) + 1)
        _LOG.info("Vertical extent %.0f m -> kmax=%d (dz=%.0f m)", hmax, kmax_used, dz)

        with contextlib.ExitStack() as stack:
            handles: dict[str, BinaryIO] = {
                name: stack.enter_context(
                    open(output_directory / f"{year}_{month:02d}_{name}", "wb")
                )
                for name in names
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

                u_interp = qgfield.interpolated_u
                v_interp = qgfield.interpolated_v
                theta_interp = qgfield.interpolated_theta
                qgpv = qgfield.qgpv
                storage = qgfield._domain_average_storage

                _write_record(fh=handles["QGPV"], arrays=[qgpv])
                _write_record(fh=handles["QGU"], arrays=[u_interp])
                _write_record(fh=handles["QGV"], arrays=[v_interp])
                _write_record(
                    fh=handles["QGT"],
                    arrays=[
                        theta_interp,
                        storage.tn0,
                        storage.ts0,
                        storage.static_stability_n,
                        storage.static_stability_s,
                    ],
                )

                if z_variable is not None:
                    zz = np.array(ds.variables[z_variable][ti], dtype=np.float64)
                    if not lat_ascending:
                        zz = zz[:, ::-1, :]
                    if not plev_descending:
                        zz = zz[::-1, :, :]
                    zz = _fix_fill_values(arr_3d=zz)
                    if z_is_geopotential:
                        zz = zz / constants.GRAVITY
                    zz_interp = _interp_to_height(
                        field_plev=zz, plev_hpa=plev_raw, kmax=kmax_used, dz=dz
                    )
                    _write_record(fh=handles["QGZ"], arrays=[zz_interp])

                avort = _absolute_vorticity(
                    u3=u_interp,
                    v3=v_interp,
                    ylat=ylat,
                    planet_radius=qgfield.planet_radius,
                    omega=qgfield.omega,
                )
                _write_record(fh=handles["QVORT"], arrays=[avort])

                qref = qgfield.qref
                uref = qgfield.uref[:, jb:]
                tref = qgfield.ptref[:, jb:]
                ubar = u_interp[:, -nhem:, :].mean(axis=-1)
                tbar = theta_interp[:, -nhem:, :].mean(axis=-1)
                fawa = np.zeros((kmax_used, nhem), dtype=np.float64)
                _write_record(
                    fh=handles["QGREF_N"], arrays=[qref, uref, tref, fawa, ubar, tbar]
                )
                del qgfield

    print(f"Done: wrote {', '.join(names)} to {output_directory}")


if __name__ == "__main__":
    typer.run(main)
