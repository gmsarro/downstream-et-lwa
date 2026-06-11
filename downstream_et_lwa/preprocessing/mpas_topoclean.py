"""Mask ERA5-style MPAS pressure-level fields below topography and fill from the nearest valid grid point."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import scipy.ndimage
import typer
import xarray as xr

import downstream_et_lwa.constants as constants

_LOG = logging.getLogger(__name__)

VARS = ("u", "v", "w", "t", "z")


def _fill_nearest_2d(*, field: np.ndarray, bad: np.ndarray) -> np.ndarray:
    out = np.asarray(field, dtype=np.float32).copy()
    fill_mask = bad | ~np.isfinite(out)
    if not np.any(fill_mask):
        return out
    valid = ~fill_mask
    if not np.any(valid):
        out[fill_mask] = 0.0
        return out
    _, inds = scipy.ndimage.distance_transform_edt(fill_mask, return_indices=True)
    out[fill_mask] = out[tuple(inds[:, fill_mask])]
    return out


def _load_topography(*, path: Path, target_lat: np.ndarray) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        topo = ds["topo"].load()
        lat = ds["lat"].values
    topo_arr = np.asarray(topo.values, dtype=np.float32)
    if lat[0] < lat[-1] and target_lat[0] > target_lat[-1]:
        topo_arr = topo_arr[::-1, :]
    elif lat[0] > lat[-1] and target_lat[0] < target_lat[-1]:
        topo_arr = topo_arr[::-1, :]
    return topo_arr


def main(
    year: Annotated[int, typer.Option()],
    month: Annotated[int, typer.Option()],
    input_directory: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    topo_file: Annotated[Path, typer.Option()],
    clearance_m: Annotated[float, typer.Option()] = 1.0,
    compress: Annotated[bool, typer.Option()] = False,
) -> None:
    tag = f"{year}_{month:02d}"
    output_directory.mkdir(parents=True, exist_ok=True)

    z_path = input_directory / f"{tag}_z.nc"
    if not z_path.is_file():
        print(f"Missing z file: {z_path}")
        raise typer.Exit(code=1)

    with xr.open_dataset(z_path) as zds:
        z = zds["z"].load()
        lat = zds["lat"].values
        z_height_m = np.asarray(z.values, dtype=np.float32) / constants.GRAVITY
        template_coords = {name: coord for name, coord in zds.coords.items()}

    topo = _load_topography(path=topo_file, target_lat=lat)
    below = z_height_m < (topo[None, None, :, :] + float(clearance_m))
    below_count = below.sum(axis=(0, 2, 3))
    total_per_level = below.shape[0] * below.shape[2] * below.shape[3]

    print(f"[{tag}] below-topography mask from {z_path}")
    for lev, count in zip(template_coords["lev"].values, below_count):
        frac = float(count) / float(total_per_level)
        if frac > 0:
            print(f"  lev={float(lev):7.1f} hPa  masked={frac * 100:6.3f}%")

    for var in VARS:
        in_path = input_directory / f"{tag}_{var}.nc"
        out_path = output_directory / f"{tag}_{var}.nc"
        if not in_path.is_file():
            print(f"Missing input: {in_path}")
            raise typer.Exit(code=1)
        print(f"[{tag}] cleaning {var}: {in_path} -> {out_path}", flush=True)
        with xr.open_dataset(in_path) as ds:
            da = ds[var].load()
            vals = np.asarray(da.values, dtype=np.float32)
            cleaned = np.empty_like(vals, dtype=np.float32)
            nt, nk = vals.shape[:2]
            for it in range(nt):
                if it == 0 or (it + 1) % 20 == 0 or it == nt - 1:
                    print(f"  {var}: time {it + 1}/{nt}", flush=True)
                for ik in range(nk):
                    cleaned[it, ik] = _fill_nearest_2d(field=vals[it, ik], bad=below[it, ik])
            out = xr.Dataset(
                {var: (da.dims, cleaned, dict(da.attrs))},
                coords={name: coord for name, coord in ds.coords.items()},
                attrs=dict(ds.attrs),
            )
            out.attrs["topoclean_method"] = (
                "masked z/g < topography + clearance and nearest-filled "
                "horizontally per time/pressure level"
            )
            out.attrs["topoclean_source"] = str(in_path)
            out.attrs["topoclean_topography"] = str(topo_file)
            out.attrs["topoclean_clearance_m"] = float(clearance_m)
            if compress:
                out[var].encoding.update({"zlib": True, "complevel": 2})
            out.to_netcdf(out_path)
            out.close()

    print(f"[{tag}] wrote cleaned files to {output_directory}")


if __name__ == "__main__":
    typer.run(main)
