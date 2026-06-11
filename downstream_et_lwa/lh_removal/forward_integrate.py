"""Forward-integrate the 2-D barotropic LWA budget in CTRL, NoLH, and NoR+ modes."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
import typer
from typing_extensions import Annotated

import downstream_et_lwa.lh_removal.advection_kernel as advection_kernel

_LOG = logging.getLogger(__name__)

MODES = ("CTRL", "NoLH", "NoR+")


def load_catalog(*, storm_id: str, catalog_directory: Path) -> dict[str, Any]:
    path = catalog_directory / f"event_{storm_id}.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog {path} not found. Build the event catalog first."
        )
    with nc.Dataset(path, "r") as d:
        has_mask = "tc_mask" in d.variables
        out: dict[str, Any] = {
            "times": np.array(d["time"][:], dtype=np.float64),
            "lat": np.array(d["lat"][:], dtype=np.float64),
            "lon": np.array(d["lon"][:], dtype=np.float64),
            "A_obs": np.array(d["A_obs"][:], dtype=np.float64),
            "c_x": np.array(d["c_x"][:], dtype=np.float64),
            "cyp": np.array(d["cyp"][:], dtype=np.float64),
            "cym": np.array(d["cym"][:], dtype=np.float64),
            "gamma": np.array(d["gamma"][:], dtype=np.float64),
            "eps4": np.array(d["eps4"][:], dtype=np.float64),
            "S_other": np.array(d["S_other"][:], dtype=np.float64),
            "LH": np.array(d["LH"][:], dtype=np.float64),
            "tc_mask": (np.array(d["tc_mask"][:], dtype=np.float64)
                        if has_mask else None),
            "tc_radius_deg": (float(d["tc_mask"].tc_radius_deg)
                              if has_mask else None),
            "attrs": {k: getattr(d, k) for k in d.ncattrs()},
        }
    out["S_other"] = np.maximum(out["S_other"], 0.0)
    out["gamma"] = np.minimum(out["gamma"], 0.0)
    return out


def run_one(*, storm_id: str, mode: str, dt_sec: float, k_diffusion: float,
            asselin: float, boundary_north: int, boundary_south: int,
            scheme: str, catalog_directory: Path, run_directory: Path,
            overwrite: bool = False) -> Path:
    if mode not in MODES:
        raise ValueError(mode)
    run_directory.mkdir(parents=True, exist_ok=True)
    safe_mode = mode.replace("+", "Pos")
    out_path = run_directory / f"{storm_id}_{safe_mode}.nc"
    if out_path.exists() and not overwrite:
        _LOG.info("Skipping %s; exists (use --overwrite to rerun)", out_path)
        return out_path

    cat = load_catalog(storm_id=storm_id, catalog_directory=catalog_directory)
    times = cat["times"]
    A_obs = cat["A_obs"]
    catalog_dt = times[1] - times[0]

    LH_eff = cat["LH"].copy()
    So_eff = cat["S_other"].copy()
    mask = cat["tc_mask"]
    R_deg = cat["tc_radius_deg"]
    if mode in ("NoLH", "NoR+"):
        if mask is None:
            raise RuntimeError(
                f"Catalog for {storm_id} lacks tc_mask: rebuild the event "
                f"catalog with --tc-radius-deg=...; cannot run mode={mode}."
            )
        keep = (1.0 - mask).astype(np.float64)
        LH_eff *= keep
        if mode == "NoR+":
            So_eff *= keep
        n_t = mask.shape[0]
        n_active = int((mask.sum(axis=(1, 2)) > 0).sum())
        _LOG.info("Localised removal: R=%g deg; active steps %d/%d (%.1f h)",
                  R_deg, n_active, n_t,
                  n_active * (times[1] - times[0]) / 3600)

    cfg = advection_kernel.IntegratorConfig(
        dt=dt_sec, K=k_diffusion, asselin_alpha=asselin,
        boundary_north=boundary_north, boundary_south=boundary_south,
        include_LH=True,
        include_S_other=True,
        scheme=scheme,
    )

    forcing = advection_kernel.build_forcing_table(
        times=times,
        A_obs=cat["A_obs"], c_x=cat["c_x"],
        cyp=cat["cyp"], cym=cat["cym"],
        gamma=cat["gamma"], eps4=cat["eps4"],
        S_other=So_eff, LH=LH_eff,
    )

    _LOG.info("storm_id=%s mode=%s scheme=%s window=%.1f h dt=%g s "
              "K=%g alpha=%g bN=%d bS=%d",
              storm_id, mode, scheme, times[-1] / 3600, dt_sec,
              k_diffusion, asselin, boundary_north, boundary_south)
    A0 = A_obs[0].copy()
    t0 = time.time()
    out_times, A_out = advection_kernel.integrate(
        A0=A0, forcing=forcing,
        t_start=float(times[0]), t_end=float(times[-1]),
        cfg=cfg, snapshot_dt=float(catalog_dt),
        progress_every=int(round(6 * 3600.0 / dt_sec)),
    )
    elapsed = time.time() - t0

    n_save = len(out_times)
    diff = A_out - A_obs[:n_save]
    rmse_t = np.sqrt(np.nanmean(diff[1:] ** 2, axis=(1, 2)))
    bias_t = np.nanmean(diff[1:], axis=(1, 2))
    _LOG.info("Integrate done in %.1f s; saved %d snapshots", elapsed, n_save)
    _LOG.info("%s-vs-obs (t>0): RMSE max=%.3g mean=%.3g bias mean=%+.3g",
              mode, rmse_t.max(), rmse_t.mean(), bias_t.mean())

    tmp = out_path.with_suffix(".nc.tmp")
    with nc.Dataset(tmp, "w", format="NETCDF4") as o:
        o.createDimension("time", n_save)
        o.createDimension("lat", advection_kernel.NLAT)
        o.createDimension("lon", advection_kernel.NLON)
        vt = o.createVariable("time", "f8", ("time",))
        vt.units = f"seconds since {cat['attrs']['window_start']}"
        vt[:] = out_times
        vlat = o.createVariable("lat", "f4", ("lat",))
        vlat.units = "degrees_north"
        vlat[:] = advection_kernel.LATS.astype(np.float32)
        vlon = o.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"
        vlon[:] = cat["lon"].astype(np.float32)
        vA = o.createVariable("A", "f4", ("time", "lat", "lon"),
                              zlib=True, complevel=4, shuffle=True)
        vA.units = "m s-1"
        vA.long_name = f"Reconstructed barotropic LWA ({mode})"
        vA[:] = A_out.astype(np.float32)
        vAo = o.createVariable("A_obs", "f4", ("time", "lat", "lon"),
                               zlib=True, complevel=4, shuffle=True)
        vAo.units = "m s-1"
        vAo.long_name = "Observed LWA (reference)"
        vAo[:] = A_obs[:n_save].astype(np.float32)
        vRMSE = o.createVariable("rmse_vs_obs", "f4", ("time",))
        vRMSE.units = "m s-1"
        rmse_full = np.sqrt(np.nanmean((A_out - A_obs[:n_save]) ** 2, axis=(1, 2)))
        vRMSE[:] = rmse_full.astype(np.float32)

        for k, v in cat["attrs"].items():
            try:
                setattr(o, f"src_{k}", v)
            except Exception:
                _LOG.exception("Could not copy attribute %s", k)
        o.mode = mode
        o.dt_sec = float(dt_sec)
        o.K = float(k_diffusion)
        o.asselin_alpha = float(asselin)
        o.boundary_north = int(boundary_north)
        o.boundary_south = int(boundary_south)
        o.elapsed_sec = float(elapsed)
        if cat["tc_radius_deg"] is not None:
            o.tc_radius_deg = float(cat["tc_radius_deg"])
        o.localised_removal = ("yes (LH and/or S_other zeroed only inside "
                               "TC disk)" if mode != "CTRL" else "n/a")
        o.note = "CTRL / NoLH / NoR+ reconstruction of the 2-D LWA budget."

    os.replace(tmp, out_path)
    _LOG.info("Wrote %s", out_path)
    return out_path


def main(
    storm_id: Annotated[str, typer.Option()],
    catalog_directory: Annotated[Path, typer.Option(
        help="Directory with event_<storm_id>.nc catalogs.")],
    output_directory: Annotated[Path, typer.Option(
        help="Directory for the per-mode run NetCDFs.")],
    mode: Annotated[str, typer.Option(
        help="CTRL | NoLH | NoR+ | all ('all' runs the three modes).")] = "all",
    dt_sec: Annotated[float, typer.Option(
        help="Integration timestep (seconds).")] = 600.0,
    k_diffusion: Annotated[float, typer.Option(
        help="Horizontal diffusivity m^2 s^-1 (LN24 default 2.3e4).")] = 2.3e4,
    asselin: Annotated[float, typer.Option(
        help="Robert-Asselin filter coefficient (leapfrog only).")] = 0.05,
    scheme: Annotated[str, typer.Option(
        help="rk3 | rk4 | euler | leapfrog.")] = "rk3",
    boundary_north: Annotated[int, typer.Option()] = 5,
    boundary_south: Annotated[int, typer.Option()] = 6,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if mode not in MODES + ("all",):
        raise typer.BadParameter(f"mode must be one of {MODES + ('all',)}")
    if scheme not in ("rk3", "rk4", "euler", "leapfrog"):
        raise typer.BadParameter("scheme must be rk3, rk4, euler, or leapfrog")
    modes = MODES if mode == "all" else (mode,)
    for m in modes:
        out = run_one(
            storm_id=storm_id, mode=m,
            dt_sec=dt_sec, k_diffusion=k_diffusion, asselin=asselin,
            boundary_north=boundary_north, boundary_south=boundary_south,
            scheme=scheme,
            catalog_directory=catalog_directory,
            run_directory=output_directory,
            overwrite=overwrite,
        )
        print(out)


if __name__ == "__main__":
    typer.run(main)
