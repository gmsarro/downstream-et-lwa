"""Build a 1-D JJASON LWA climatology (cos-lat 20-80N) from BARO_N ERA5 LWAb_N files."""

from __future__ import annotations

import logging
from pathlib import Path

import netCDF4 as nc
import numpy as np
import typer
from typing_extensions import Annotated

import downstream_et_lwa.lh_removal.advection_kernel as advection_kernel

_LOG = logging.getLogger(__name__)

JJASON = (6, 7, 8, 9, 10, 11)


def build_climatology(*, baro_directory: Path, year_start: int, year_end: int,
                      lat_min: float, lat_max: float,
                      output_path: Path) -> Path:
    sel = ((advection_kernel.LATS >= lat_min)
           & (advection_kernel.LATS <= lat_max))
    w = np.cos(np.deg2rad(advection_kernel.LATS[sel]))
    w_sum = float(w.sum())

    acc = np.zeros(advection_kernel.NLON, dtype=np.float64)
    n = 0
    for y in range(year_start, year_end + 1):
        fp = baro_directory / f"{y}_LWAb_N.nc"
        if not fp.exists():
            _LOG.info("Skipping %s; missing", fp)
            continue
        with nc.Dataset(fp, "r") as d:
            lwa = np.asarray(d["lwa"][:], dtype=np.float64)
        for mi in range(12):
            month = mi + 1
            if month not in JJASON:
                continue
            slab = lwa[:, mi, :, :]
            valid = (np.isfinite(slab).any(axis=(1, 2))
                     & (slab != 0).any(axis=(1, 2)))
            slab = slab[valid]
            if slab.size == 0:
                continue
            r1d = (slab[:, sel, :] * w[None, :, None]).sum(axis=1) / w_sum
            acc += r1d.sum(axis=0)
            n += r1d.shape[0]
        _LOG.info("[%d] cumulative samples = %d", y, n)

    if n == 0:
        raise RuntimeError("No samples accumulated.")
    clim = (acc / n).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(output_path, "w", format="NETCDF4") as o:
        o.createDimension("lon", advection_kernel.NLON)
        vlon = o.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"
        vlon[:] = np.linspace(0, 359, advection_kernel.NLON, dtype=np.float32)
        vc = o.createVariable("lwa_clim", "f4", ("lon",), zlib=True, complevel=4)
        vc.units = "m s-1"
        vc.long_name = (f"JJASON cos-lat {lat_min:g}-{lat_max:g}N mean "
                        "barotropic LWA, ERA5 BARO_N")
        vc[:] = clim
        o.year_start = year_start
        o.year_end = year_end
        o.lat_min = lat_min
        o.lat_max = lat_max
        o.n_samples = int(n)
    _LOG.info("Wrote %s (n_samples = %d)", output_path, n)
    return output_path


def main(
    baro_directory: Annotated[Path, typer.Option(
        help="ERA5 BARO_N archive with yearly <year>_LWAb_N.nc files.")],
    output_directory: Annotated[Path, typer.Option(
        help="Directory for the climatology NetCDF.")],
    output_name: Annotated[str, typer.Option()] = "lwa_clim_jjason_20-80N.nc",
    year_start: Annotated[int, typer.Option()] = 2000,
    year_end: Annotated[int, typer.Option()] = 2022,
    lat_min: Annotated[float, typer.Option()] = 20.0,
    lat_max: Annotated[float, typer.Option()] = 80.0,
) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    out = build_climatology(
        baro_directory=baro_directory,
        year_start=year_start,
        year_end=year_end,
        lat_min=lat_min,
        lat_max=lat_max,
        output_path=output_directory / output_name,
    )
    print(out)


if __name__ == "__main__":
    typer.run(main)
