"""Batch driver: catalog, CTRL/NoLH/NoR+ runs, and 20-80N strip per recurving storm."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from pathlib import Path
from typing import Optional

import netCDF4 as nc
import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

import downstream_et_lwa.lh_removal.advection_kernel as advection_kernel
import downstream_et_lwa.lh_removal.event_catalog as event_catalog
import downstream_et_lwa.lh_removal.forward_integrate as forward_integrate

_LOG = logging.getLogger(__name__)

LAT_MIN_STRIP = 20.0
LAT_MAX_STRIP = 80.0


def _strip_merid_mean(*, arr_t_lat_lon: np.ndarray) -> np.ndarray:
    sel = ((advection_kernel.LATS >= LAT_MIN_STRIP)
           & (advection_kernel.LATS <= LAT_MAX_STRIP))
    w = advection_kernel.COSPHI[sel]
    return ((arr_t_lat_lon[:, sel, :] * w[None, :, None]).sum(axis=1)
            / w.sum())


def _process_one(*,
                 storm_id: str,
                 t_pre_days: float,
                 t_post_days: float,
                 dt_int_min: float,
                 dt_sec: float,
                 k_diffusion: float,
                 boundary_north: int,
                 boundary_south: int,
                 scheme: str,
                 tc_radius_deg: float,
                 cleanup: bool,
                 overwrite: bool,
                 tracks_file: Path,
                 individual_tracks_directory: Path,
                 baro_directory: Path,
                 lh_source_directory: Path,
                 catalog_directory: Path,
                 run_directory: Path,
                 strip_directory: Path) -> str:
    t0 = time.time()
    strip_directory.mkdir(parents=True, exist_ok=True)
    strip_path = strip_directory / f"strip_{storm_id}.nc"
    if strip_path.exists() and not overwrite:
        return f"[skip] {storm_id} strip exists"

    try:
        cat_path = event_catalog.build_event_catalog(
            storm_id=storm_id,
            tracks_file=tracks_file,
            individual_tracks_directory=individual_tracks_directory,
            baro_directory=baro_directory,
            lh_source_directory=lh_source_directory,
            output_directory=catalog_directory,
            t_pre_days=t_pre_days, t_post_days=t_post_days,
            dt_int_min=dt_int_min,
            tc_radius_deg=tc_radius_deg,
            overwrite=overwrite,
        )
        run_paths = []
        for mode in ("CTRL", "NoLH", "NoR+"):
            p = forward_integrate.run_one(
                storm_id=storm_id, mode=mode,
                dt_sec=dt_sec, k_diffusion=k_diffusion, asselin=0.05,
                boundary_north=boundary_north, boundary_south=boundary_south,
                scheme=scheme,
                catalog_directory=catalog_directory,
                run_directory=run_directory,
                overwrite=overwrite,
            )
            run_paths.append((mode, p))

        strips = {}
        with nc.Dataset(cat_path, "r") as d:
            times_s = np.array(d["time"][:], dtype=np.float64)
            t_units = d["time"].units
            attrs = {k: getattr(d, k) for k in d.ncattrs()}
            A_obs = np.array(d["A_obs"][:], dtype=np.float32)
            LH = np.array(d["LH"][:], dtype=np.float32)
            S_other = np.array(d["S_other"][:], dtype=np.float32)
            gamma = np.array(d["gamma"][:], dtype=np.float32)
            eps4 = np.array(d["eps4"][:], dtype=np.float32)
        strips["A_obs"] = _strip_merid_mean(arr_t_lat_lon=A_obs)
        strips["LH"] = _strip_merid_mean(arr_t_lat_lon=LH)
        strips["S_other"] = _strip_merid_mean(arr_t_lat_lon=S_other)
        strips["gamma_A"] = _strip_merid_mean(arr_t_lat_lon=gamma * A_obs)
        strips["eps4"] = _strip_merid_mean(arr_t_lat_lon=eps4)

        for mode, p in run_paths:
            with nc.Dataset(p, "r") as d:
                A = np.array(d["A"][:], dtype=np.float32)
            key = "A_" + mode.replace("+", "Pos")
            strips[key] = _strip_merid_mean(arr_t_lat_lon=A)

        tmp = strip_path.with_suffix(".nc.tmp")
        with nc.Dataset(tmp, "w", format="NETCDF4") as o:
            o.createDimension("time", len(times_s))
            o.createDimension("lon", advection_kernel.NLON)
            vt = o.createVariable("time", "f8", ("time",))
            vt.units = t_units
            vt[:] = times_s
            vlon = o.createVariable("lon", "f4", ("lon",))
            vlon.units = "degrees_east"
            vlon[:] = advection_kernel.LONS.astype(np.float32)
            for name, arr in strips.items():
                v = o.createVariable(name, "f4", ("time", "lon"),
                                     zlib=True, complevel=4, shuffle=True)
                v[:] = arr.astype(np.float32)
                v.cos_lat_band = f"{LAT_MIN_STRIP:g}-{LAT_MAX_STRIP:g}N"
            for k, v_attr in attrs.items():
                try:
                    setattr(o, k, v_attr)
                except Exception:
                    _LOG.exception("Could not copy attribute %s", k)
            o.dt_sec = float(dt_sec)
            o.K_diff = float(k_diffusion)
            o.scheme = scheme
            o.tc_radius_deg = float(tc_radius_deg)
            o.note = ("1-D meridional-mean (cos-lat 20-80N) Hovmoller strips "
                      "for the LH/R+ removal experiment.")
        os.replace(tmp, strip_path)

        if cleanup:
            try:
                cat_path.unlink(missing_ok=True)
            except Exception:
                _LOG.exception("Could not delete %s", cat_path)
            for _, p in run_paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    _LOG.exception("Could not delete %s", p)

        elapsed = time.time() - t0
        return (f"[ok]  {storm_id}  strip n_t={len(times_s)}  "
                f"|dLH|max={np.abs(strips['A_CTRL'] - strips['A_NoLH']).max():.2f}  "
                f"|dR+|max={np.abs(strips['A_CTRL'] - strips['A_NoRPos']).max():.2f}  "
                f"({elapsed:.1f}s)")
    except Exception as e:
        _LOG.exception("Storm %s failed", storm_id)
        return f"[ERR] {storm_id}: {type(e).__name__}: {e}"


def select_storms(*, tracks_file: Path, basins: list[str],
                  year_start: int, year_end: int) -> pd.DataFrame:
    df = pd.read_csv(tracks_file,
                     parse_dates=["recurv_time", "et_time"],
                     keep_default_na=False, na_values=[""])
    yr = df["recurv_time"].dt.year
    sel = (yr >= year_start) & (yr <= year_end) & df["basin"].isin(basins)
    return df[sel].copy().reset_index(drop=True)


def main(
    tracks_file: Annotated[Path, typer.Option(help="Recurving NH tracks CSV.")],
    individual_tracks_directory: Annotated[Path, typer.Option(
        help="Directory of per-storm track CSVs (<storm_id>.csv).")],
    baro_directory: Annotated[Path, typer.Option(
        help="ERA5 BARO_N archive with yearly files per variable.")],
    lh_source_directory: Annotated[Path, typer.Option(
        help="Monthly latent-heating LWA tendency NetCDFs.")],
    output_directory: Annotated[Path, typer.Option(
        help="Base output directory; catalogs/, runs/, and runs/strips/ "
             "are created inside it.")],
    year_start: Annotated[int, typer.Option()] = 2000,
    year_end: Annotated[int, typer.Option()] = 2022,
    basins: Annotated[list[str], typer.Option()] = ["WP", "NA"],
    t_pre_days: Annotated[float, typer.Option()] = 1.0,
    t_post_days: Annotated[float, typer.Option()] = 8.0,
    dt_int_min: Annotated[float, typer.Option()] = 30.0,
    dt_sec: Annotated[float, typer.Option()] = 300.0,
    k_diffusion: Annotated[float, typer.Option()] = 2.3e4,
    boundary_north: Annotated[int, typer.Option()] = 11,
    boundary_south: Annotated[int, typer.Option()] = 6,
    scheme: Annotated[str, typer.Option(
        help="rk3 | rk4 | euler | leapfrog.")] = "rk3",
    tc_radius_deg: Annotated[float, typer.Option(
        help="Great-circle radius (deg) of TC mask used to localise "
             "LH / R+ removal.")] = event_catalog.DEFAULT_TC_RADIUS_DEG,
    n_workers: Annotated[int, typer.Option()] = 8,
    cleanup: Annotated[bool, typer.Option(
        help="Delete catalog + 2-D run files after extracting strip.")] = False,
    overwrite: Annotated[bool, typer.Option()] = False,
    limit: Annotated[Optional[int], typer.Option(
        help="Only run the first N storms (debug).")] = None,
) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if scheme not in ("rk3", "rk4", "euler", "leapfrog"):
        raise typer.BadParameter("scheme must be rk3, rk4, euler, or leapfrog")

    catalog_directory = output_directory / "catalogs"
    run_directory = output_directory / "runs"
    strip_directory = run_directory / "strips"

    df = select_storms(tracks_file=tracks_file, basins=basins,
                       year_start=year_start, year_end=year_end)
    if limit is not None:
        df = df.iloc[:limit].copy()
    print(f"[info] {len(df)} storms in basins {basins} "
          f"({year_start}-{year_end}), n_workers={n_workers}, "
          f"cleanup={cleanup}")

    storm_ids = list(df["storm_id"])
    t_start = time.time()
    if n_workers <= 1:
        for sid in storm_ids:
            print(_process_one(
                storm_id=sid,
                t_pre_days=t_pre_days, t_post_days=t_post_days,
                dt_int_min=dt_int_min, dt_sec=dt_sec,
                k_diffusion=k_diffusion,
                boundary_north=boundary_north,
                boundary_south=boundary_south,
                scheme=scheme, tc_radius_deg=tc_radius_deg,
                cleanup=cleanup, overwrite=overwrite,
                tracks_file=tracks_file,
                individual_tracks_directory=individual_tracks_directory,
                baro_directory=baro_directory,
                lh_source_directory=lh_source_directory,
                catalog_directory=catalog_directory,
                run_directory=run_directory,
                strip_directory=strip_directory,
            ), flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {
                ex.submit(
                    _process_one,
                    storm_id=sid,
                    t_pre_days=t_pre_days, t_post_days=t_post_days,
                    dt_int_min=dt_int_min, dt_sec=dt_sec,
                    k_diffusion=k_diffusion,
                    boundary_north=boundary_north,
                    boundary_south=boundary_south,
                    scheme=scheme, tc_radius_deg=tc_radius_deg,
                    cleanup=cleanup, overwrite=overwrite,
                    tracks_file=tracks_file,
                    individual_tracks_directory=individual_tracks_directory,
                    baro_directory=baro_directory,
                    lh_source_directory=lh_source_directory,
                    catalog_directory=catalog_directory,
                    run_directory=run_directory,
                    strip_directory=strip_directory,
                ): sid
                for sid in storm_ids
            }
            done = 0
            for f in concurrent.futures.as_completed(futs):
                done += 1
                msg = f.result()
                print(f"[{done:3d}/{len(storm_ids):3d}] {msg}", flush=True)
    elapsed = time.time() - t_start
    print(f"[done] {len(storm_ids)} storms in {elapsed:.1f} s "
          f"({elapsed / max(len(storm_ids), 1):.1f} s/storm)")


if __name__ == "__main__":
    typer.run(main)
