"""Build rwb_masks_{Y}_{MM}.nc for one MPAS month from the Fortran QGPV binary.

RWB detection requires the public ``wave_breaking_using_qgpv`` repository
(https://github.com/gmsarro/wave_breaking_using_qgpv) to be importable; pass
its checkout directory via ``--rwb-repo-directory`` (inserted into sys.path)."""

from __future__ import annotations

import datetime
import importlib
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import numpy as np
import typer
import xarray as xr
from typing_extensions import Annotated

import downstream_et_lwa.composites.mpas_qgpv_export as mpas_qgpv_export

_LOG = logging.getLogger(__name__)


def _load_detection_module(*, repo_directory: Path) -> ModuleType:
    sys.path.insert(0, str(repo_directory))
    for name in ("detection", "detect_wave_breaking_qgpv"):
        try:
            return importlib.import_module(name)
        except ImportError:
            _LOG.exception("Could not import %s from %s", name, repo_directory)
    raise SystemExit(
        f"No RWB detection module (detection.py / detect_wave_breaking_qgpv.py) "
        f"importable from {repo_directory}")


def main(
        rwb_repo_directory: Annotated[Path, typer.Option(
            help="Checkout of wave_breaking_using_qgpv (inserted into sys.path)")],
        work_root: Annotated[Path, typer.Option(
            help="Root containing {mode}/work/{year}/{Y}_{MM}_QGPV")],
        output_directory: Annotated[Path, typer.Option(
            help="Mask output directory")],
        mode: Annotated[str, typer.Option(help="current or future")] = "current",
        year: Annotated[int, typer.Option(
            help="Calendar year of the month tag")] = 2013,
        month: Annotated[int, typer.Option(help="Month 1-12")] = 1,
        height_km: Annotated[float, typer.Option()] = 10.0,
        contour_level: Annotated[Optional[float], typer.Option(
            help="QGPV contour level (default: detection module default)")] = None,
        buffer_deg: Annotated[float, typer.Option()] = 1.0,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())

    detection = _load_detection_module(repo_directory=rwb_repo_directory)
    if contour_level is None:
        contour_level = float(detection.DEFAULT_QGPV_LEVEL)

    y, m = year, month
    mm = f"{m:02d}"
    tag = f"{y}_{mm}"
    work = os.path.join(str(work_root), mode, "work", str(y))
    qgpv_path = os.path.join(work, f"{tag}_QGPV")
    if not os.path.isfile(qgpv_path):
        raise SystemExit(f"Missing QGPV binary: {qgpv_path}")

    os.makedirs(output_directory, exist_ok=True)
    out_nc = os.path.join(str(output_directory), f"rwb_masks_{tag}.nc")

    heights = mpas_qgpv_export.HEIGHTS[:41]
    height_idx = int(np.argmin(np.abs(heights - height_km)))
    actual_km = float(heights[height_idx])

    qgpv, times = mpas_qgpv_export.read_fortran_qgpv(
        filepath=qgpv_path, year=y, month=m, truncate_height=True)
    nt = qgpv.shape[0]

    lats_full = np.arange(-90, 91, 1.0, dtype=np.float64)
    lons = np.arange(0, 360, 1.0, dtype=np.float64)
    lat_nh = lats_full[lats_full >= 0]

    all_awb = []
    all_cwb = []
    valid = np.ones(nt, dtype=np.int8)

    for t in range(nt):
        q2 = np.ascontiguousarray(qgpv[t, height_idx, :, :], dtype=np.float32)
        q2_nh = q2[lats_full >= 0, :]
        try:
            _, awb, cwb = detection.process_single_timestep(
                q2_nh,
                lat_nh,
                lons,
                np.datetime64(times[t]),
                actual_km,
                contour_level=contour_level,
                buffer_deg=buffer_deg,
            )
        except Exception:
            _LOG.exception("timestep %d %s failed", t, times[t])
            awb = np.zeros((len(lat_nh), len(lons)), dtype=np.float32)
            cwb = np.zeros((len(lat_nh), len(lons)), dtype=np.float32)
            valid[t] = 0
        all_awb.append(awb)
        all_cwb.append(cwb)

    masks_awb = np.stack(all_awb, axis=0)
    masks_cwb = np.stack(all_cwb, axis=0)

    ds_out = xr.Dataset(
        {
            "rwb_mask_awb": (["time", "lat", "lon"], masks_awb),
            "rwb_mask_cwb": (["time", "lat", "lon"], masks_cwb),
            "valid_mask": (["time"], valid),
        },
        coords={
            "time": [np.datetime64(x) for x in times],
            "lat": lat_nh,
            "lon": lons,
        },
    )
    ds_out["rwb_mask_awb"].attrs["long_name"] = "Anticyclonic Wave Breaking mask"
    ds_out["rwb_mask_awb"].attrs["units"] = "1"
    ds_out["rwb_mask_cwb"].attrs["long_name"] = "Cyclonic Wave Breaking mask"
    ds_out["rwb_mask_cwb"].attrs["units"] = "1"

    ds_out.attrs["source"] = f"MPAS {mode}, month={tag}"
    ds_out.attrs["qgpv_source"] = qgpv_path
    ds_out.attrs["qgpv_height_m"] = float(actual_km * 1000.0)
    ds_out.attrs["contour_level"] = float(contour_level)
    ds_out.attrs["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    ds_out.attrs["method"] = (
        "AWB/CWB detection on z=10 km slab of Fortran QGPV "
        "(era1000.f90 pv output, kmax=97, dz=500 m)."
    )

    enc = {
        "rwb_mask_awb": {"zlib": True, "complevel": 4},
        "rwb_mask_cwb": {"zlib": True, "complevel": 4},
        "valid_mask": {"zlib": True, "complevel": 4},
    }
    ds_out.to_netcdf(out_nc, encoding=enc)
    print(f"Wrote {out_nc}", flush=True)


if __name__ == "__main__":
    typer.run(main)
