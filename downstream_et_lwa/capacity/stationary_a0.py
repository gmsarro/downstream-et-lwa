"""Stationary LWA A0(month, lat, lon) from monthly-mean u, v, t via falwa QGFieldNHN22.

The original work computed A0 with the Nakamura Fortran chain
(eranew/era1000/era4004n); this module is the falwa equivalent.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Optional

import netCDF4
import numpy as np
import typer
from typing_extensions import Annotated

import downstream_et_lwa.constants as constants
import falwa.oopinterface

_LOG = logging.getLogger(__name__)

_FILL_THRESH = 1e10
_NATIVE_VARS = ("u", "v", "t")


def _pick_name(*, dataset: netCDF4.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.variables:
            return name
    raise KeyError(f"None of {candidates} found in input file")


def _resolve_native_nc(
    *, input_directory: Path, year: int, month: int, var: str
) -> Path:
    name = f"{year}_{month:02d}_{var}.nc"
    direct = input_directory / name
    if direct.exists():
        return direct
    matches = sorted(input_directory.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing {name} under {input_directory}")


def _fix_fill_values(*, arr_3d: np.ndarray) -> np.ndarray:
    arr = arr_3d.copy()
    arr[np.abs(arr) > _FILL_THRESH] = np.nan
    for k in range(arr.shape[0]):
        mask = np.isnan(arr[k])
        if mask.any():
            valid = arr[k][~mask]
            arr[k][mask] = np.nanmean(valid) if len(valid) > 0 else 0.0
    return arr


def _monthly_mean_field(*, path: Path, var: str) -> np.ndarray:
    with netCDF4.Dataset(str(path), "r") as ds:
        var_name = _pick_name(dataset=ds, candidates=(var,))
        arr = np.ma.filled(ds.variables[var_name][:], np.nan).astype(np.float64)
    zero_step = np.all(arr == 0.0, axis=(1, 2, 3))
    n_zero = int(zero_step.sum())
    if n_zero > 0:
        _LOG.info("%s: masked %d zero-padded timesteps", path.name, n_zero)
        arr[zero_step] = np.nan
    return np.nanmean(arr, axis=0)


def _read_coordinates(*, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with netCDF4.Dataset(str(path), "r") as ds:
        p_name = _pick_name(
            dataset=ds, candidates=("pressure_level", "level", "lev", "plev")
        )
        lat_name = _pick_name(dataset=ds, candidates=("latitude", "lat"))
        lon_name = _pick_name(dataset=ds, candidates=("longitude", "lon"))
        plev = np.array(ds.variables[p_name][:], dtype=np.float64)
        lat = np.array(ds.variables[lat_name][:], dtype=np.float64)
        lon = np.array(ds.variables[lon_name][:], dtype=np.float64)
    return plev, lat, lon


def compute_stationary_lwa(
    *,
    input_directory: Path,
    year: int,
    month: int,
    kmax: int,
    dz: float,
    eq_boundary_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = {
        var: _resolve_native_nc(
            input_directory=input_directory, year=year, month=month, var=var
        )
        for var in _NATIVE_VARS
    }
    plev_raw, lat_raw, xlon = _read_coordinates(path=paths["u"])
    uu = _monthly_mean_field(path=paths["u"], var="u")
    vv = _monthly_mean_field(path=paths["v"], var="v")
    tt = _monthly_mean_field(path=paths["t"], var="t")

    lat_ascending = bool(lat_raw[0] < lat_raw[-1])
    if not lat_ascending:
        lat_raw = lat_raw[::-1]
        uu = uu[:, ::-1, :]
        vv = vv[:, ::-1, :]
        tt = tt[:, ::-1, :]
    plev_descending = bool(plev_raw[0] > plev_raw[-1])
    if not plev_descending:
        plev_raw = plev_raw[::-1]
        uu = uu[::-1, :, :]
        vv = vv[::-1, :, :]
        tt = tt[::-1, :, :]

    uu = _fix_fill_values(arr_3d=uu)
    vv = _fix_fill_values(arr_3d=vv)
    tt = _fix_fill_values(arr_3d=tt)

    ylat = np.linspace(lat_raw[0], lat_raw[-1], len(lat_raw))
    nlat = len(ylat)
    nhem = nlat // 2 + 1
    ylat_nh = ylat[-nhem:]

    hmax = -constants.SCALE_HEIGHT_M * np.log(plev_raw[-1] / 1000.0)
    kmax_used = min(kmax, int(hmax // dz) + 1)
    _LOG.info(
        "%d-%02d: vertical extent %.0f m -> kmax=%d (dz=%.0f m)",
        year,
        month,
        hmax,
        kmax_used,
        dz,
    )

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
    lwa_stat = np.asarray(qgfield.lwa_baro, dtype=np.float32)
    return lwa_stat, ylat_nh, xlon


def process_month(
    *,
    input_directory: Path,
    work_directory: Path,
    year: int,
    month: int,
    kmax: int,
    dz: float,
    eq_boundary_index: int,
    force_rebuild: bool = False,
) -> tuple[int, int, Optional[str], str]:
    out_dir = work_directory / "lwa_mon"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_np = out_dir / f"{year}_{month:02d}_LWA_MON_N.npz"
    if out_np.exists() and not force_rebuild:
        return (year, month, str(out_np), "cached")

    try:
        lwa_stat, ylat_nh, xlon = compute_stationary_lwa(
            input_directory=input_directory,
            year=year,
            month=month,
            kmax=kmax,
            dz=dz,
            eq_boundary_index=eq_boundary_index,
        )
        np.savez_compressed(
            out_np, lwa_stat=lwa_stat, latitude=ylat_nh, longitude=xlon
        )
    except Exception as exc:
        _LOG.exception("Failed %d-%02d", year, month)
        return (year, month, None, f"ERROR: {exc}")

    return (year, month, str(out_np), "ok")


def aggregate_climatology(
    *,
    work_directory: Path,
    output_directory: Path,
    dataset_label: str,
    source_note: str = "",
) -> tuple[Path, np.ndarray]:
    out_dir = work_directory / "lwa_mon"
    files = sorted(out_dir.glob("*_LWA_MON_N.npz"))
    if not files:
        raise FileNotFoundError(f"No cached monthly LWA npz files under {out_dir}")

    lwa_clim: Optional[np.ndarray] = None
    latitude: Optional[np.ndarray] = None
    longitude: Optional[np.ndarray] = None
    counts = np.zeros(12, dtype=np.int32)

    for fp in files:
        base = fp.name.replace("_LWA_MON_N.npz", "")
        _, mo_s = base.split("_")
        mo = int(mo_s)
        with np.load(fp) as d:
            arr = d["lwa_stat"].astype(np.float64)
            if latitude is None:
                latitude = (
                    np.array(d["latitude"])
                    if "latitude" in d.files
                    else np.arange(arr.shape[0], dtype="f4")
                )
                longitude = (
                    np.array(d["longitude"])
                    if "longitude" in d.files
                    else np.arange(arr.shape[1], dtype="f4")
                )
        if lwa_clim is None:
            lwa_clim = np.full((12, *arr.shape), np.nan, dtype=np.float32)
        m_idx = mo - 1
        if counts[m_idx] == 0:
            lwa_clim[m_idx, :, :] = arr
        else:
            lwa_clim[m_idx, :, :] = lwa_clim[m_idx, :, :] + arr
        counts[m_idx] += 1

    assert lwa_clim is not None and latitude is not None and longitude is not None
    for m in range(12):
        if counts[m] > 0:
            lwa_clim[m, :, :] = lwa_clim[m, :, :] / counts[m]
        else:
            lwa_clim[m, :, :] = np.nan

    output_directory.mkdir(parents=True, exist_ok=True)
    out_nc = output_directory / f"{dataset_label}_LWAb_stationary_N.nc"
    with netCDF4.Dataset(str(out_nc), "w") as ds:
        ds.createDimension("month", 12)
        ds.createDimension("latitude", lwa_clim.shape[1])
        ds.createDimension("longitude", lwa_clim.shape[2])
        m_var = ds.createVariable("month", "i4", ("month",))
        la = ds.createVariable("latitude", "f4", ("latitude",))
        lo = ds.createVariable("longitude", "f4", ("longitude",))
        lwa = ds.createVariable(
            "lwa",
            "f4",
            ("month", "latitude", "longitude"),
            fill_value=np.float32(np.nan),
        )
        m_var[:] = np.arange(1, 13)
        la[:] = latitude
        la.units = "degrees_north"
        lo[:] = longitude
        lo.units = "degrees_east"
        lwa[:, :, :] = lwa_clim
        ds.title = f"Stationary (monthly-mean-PV-based) LWA, {dataset_label}"
        ds.source = source_note
        ds.method = (
            "falwa QGFieldNHN22 (interpolate_fields, compute_reference_states, "
            "compute_lwa_and_barotropic_fluxes) applied to monthly-mean u, v, t; "
            "lwa_baro of the time-mean state; falwa equivalent of the Nakamura "
            "era4000n_stationary.f90 chain"
        )
        ds.count_per_month = ",".join(f"{i + 1}:{counts[i]}" for i in range(12))

    return out_nc, counts


def _list_input_ym(*, input_directory: Path) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for f in input_directory.rglob("*_u.nc"):
        parts = f.name.replace("_u.nc", "").split("_")
        if len(parts) != 2:
            continue
        try:
            yr, mo = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        out.add((yr, mo))
    return sorted(out)


def main(
    input_directory: Annotated[
        Path,
        typer.Option(
            help="Directory (searched recursively) with native monthly NetCDFs YYYY_MM_{u,v,t}.nc"
        ),
    ],
    work_directory: Annotated[
        Path, typer.Option(help="Scratch directory for the per-month lwa_mon/ npz cache")
    ],
    output_directory: Annotated[
        Path, typer.Option(help="Directory for {label}_LWAb_stationary_N.nc")
    ],
    dataset_label: Annotated[
        str, typer.Option(help="Label used in output file names, e.g. mpas_current")
    ],
    kmax: Annotated[
        int, typer.Option(help="Number of pseudo-height levels (capped by input column depth)")
    ] = 97,
    dz: Annotated[float, typer.Option(help="Pseudo-height spacing in metres")] = 500.0,
    eq_boundary_index: Annotated[
        int, typer.Option(help="falwa equatorward boundary index")
    ] = 5,
    n_parallel: Annotated[int, typer.Option(help="Worker processes")] = 4,
    year: Annotated[Optional[int], typer.Option(help="Single year (debug)")] = None,
    month: Annotated[Optional[int], typer.Option(help="Single month (debug)")] = None,
    aggregate_only: Annotated[
        bool, typer.Option(help="Only aggregate cached per-month results")
    ] = False,
    force: Annotated[
        bool, typer.Option(help="Rebuild per-month npz even if cached")
    ] = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    source_note = f"Monthly-mean u, v, t under {input_directory}"

    if aggregate_only:
        out_nc, counts = aggregate_climatology(
            work_directory=work_directory,
            output_directory=output_directory,
            dataset_label=dataset_label,
            source_note=source_note,
        )
        print(f"[{dataset_label}] aggregate -> {out_nc}  counts={dict(enumerate(counts, 1))}")
        return

    if year is not None and month is not None:
        ym_list = [(year, month)]
    else:
        ym_list = _list_input_ym(input_directory=input_directory)
    print(
        f"[{dataset_label}] processing {len(ym_list)} (year,month) pairs with "
        f"{n_parallel} workers"
    )

    t0 = time.time()
    done, fail = 0, 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_parallel) as ex:
        futs = {
            ex.submit(
                process_month,
                input_directory=input_directory,
                work_directory=work_directory,
                year=y,
                month=m,
                kmax=kmax,
                dz=dz,
                eq_boundary_index=eq_boundary_index,
                force_rebuild=force,
            ): (y, m)
            for (y, m) in ym_list
        }
        for fut in concurrent.futures.as_completed(futs):
            yr, mo, fp, msg = fut.result()
            if fp is None:
                fail += 1
                print(f"[{dataset_label}] {yr}-{mo:02d}  FAIL: {msg}")
            else:
                done += 1
                if msg != "cached":
                    dt = time.time() - t0
                    print(
                        f"[{dataset_label}] {yr}-{mo:02d}  ok   "
                        f"({done}/{len(ym_list)}  cum {dt:.0f}s)"
                    )
    print(f"[{dataset_label}] done={done} fail={fail}  elapsed={time.time() - t0:.0f}s")

    out_nc, counts = aggregate_climatology(
        work_directory=work_directory,
        output_directory=output_directory,
        dataset_label=dataset_label,
        source_note=source_note,
    )
    print(f"[{dataset_label}] aggregate -> {out_nc}  counts_per_month={list(counts)}")


if __name__ == "__main__":
    typer.run(main)
