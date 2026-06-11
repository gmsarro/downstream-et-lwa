"""Quinting-Jones RWP frequency/amplitude climatology from pre-computed envelope files."""
from __future__ import annotations

import dataclasses
import datetime
import logging
from pathlib import Path
from typing import Annotated

import netCDF4
import numpy as np
import typer

_LOG = logging.getLogger(__name__)

NLAT = 91
NLON = 360
LATS = np.linspace(0, 90, NLAT)
LONS = np.linspace(0, 359, NLON)
COSPHI = np.cos(np.deg2rad(LATS))
ENVELOPE_FILENAME_PATTERN = "rwp_envelope_{year}_{month:02d}.nc"

SEASONS: dict[str, list[int]] = {
    "JJASON": [6, 7, 8, 9, 10, 11],
    "DJFMA": [12, 1, 2, 3, 4],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "ANN": list(range(1, 13)),
}


@dataclasses.dataclass
class _SeasonAccumulation:
    mask2d_sum: np.ndarray
    env2d_sum: np.ndarray
    mask1d_sum: np.ndarray
    env1d_sum: np.ndarray
    env1d_mask_sum: np.ndarray
    n_times: int


@dataclasses.dataclass
class _SeasonClimatology:
    freq_2d: np.ndarray
    ampl_2d: np.ndarray
    freq_1d: np.ndarray
    ampl_1d: np.ndarray
    n_times: int


def _merid_band(*, lat_min: float = 20, lat_max: float = 80) -> tuple[np.ndarray, np.ndarray]:
    sel = (LATS >= lat_min) & (LATS <= lat_max)
    return sel, COSPHI[sel]


def accumulate_season(*, season_months: list[int], year_start: int, year_end: int,
                      lat_min: float = 20, lat_max: float = 80,
                      input_directory: Path) -> _SeasonAccumulation:
    sel, w = _merid_band(lat_min=lat_min, lat_max=lat_max)
    w_sum = w.sum()

    mask2d_sum = np.zeros((NLAT, NLON), dtype=np.float64)
    env2d_sum = np.zeros((NLAT, NLON), dtype=np.float64)
    mask1d_sum = np.zeros(NLON, dtype=np.float64)
    env1d_sum = np.zeros(NLON, dtype=np.float64)
    env1d_mask_sum = np.zeros(NLON, dtype=np.float64)
    n_times = 0

    for year in range(year_start, year_end + 1):
        for month in season_months:
            path = input_directory / ENVELOPE_FILENAME_PATTERN.format(year=year, month=month)
            if not path.exists():
                continue
            with netCDF4.Dataset(path, "r") as d:
                mask = d["mask"][:].astype(np.float32)
                env_thr = d["envelope_thr"][:]

            mask2d_sum += mask.sum(axis=0)
            env2d_sum += env_thr.sum(axis=0)

            e1d = (env_thr[:, sel, :] * w[None, :, None]).sum(axis=1) / w_sum
            m1d = (e1d > 0).astype(np.float32)

            env1d_sum += e1d.sum(axis=0)
            mask1d_sum += m1d.sum(axis=0)
            env1d_mask_sum += (e1d * m1d).sum(axis=0)
            n_times += mask.shape[0]

    return _SeasonAccumulation(
        mask2d_sum=mask2d_sum,
        env2d_sum=env2d_sum,
        mask1d_sum=mask1d_sum,
        env1d_sum=env1d_sum,
        env1d_mask_sum=env1d_mask_sum,
        n_times=n_times,
    )


def main(
    input_directory: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    year_start: Annotated[int, typer.Option()] = 2000,
    year_end: Annotated[int, typer.Option()] = 2023,
    seasons: Annotated[list[str] | None, typer.Option()] = None,
    lat_min: Annotated[float, typer.Option()] = 20.0,
    lat_max: Annotated[float, typer.Option()] = 80.0,
    output_filename: Annotated[str, typer.Option()] = "rwp_climatology.nc",
) -> None:
    logging.basicConfig(level=logging.INFO)
    season_names = seasons if seasons else ["JJASON", "DJFMA", "ANN"]
    for season in season_names:
        if season not in SEASONS:
            raise typer.BadParameter(f"unknown season {season}; choose from {sorted(SEASONS)}")

    results: dict[str, _SeasonClimatology] = {}
    for season in season_names:
        months = SEASONS[season]
        print(f"[{season}] accumulating months={months} years={year_start}-{year_end}...")
        agg = accumulate_season(season_months=months, year_start=year_start,
                                year_end=year_end, lat_min=lat_min, lat_max=lat_max,
                                input_directory=input_directory)
        n = agg.n_times
        if n == 0:
            print(f"  no data for {season}, skipping")
            continue

        f2 = (agg.mask2d_sum / n).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            a2 = np.where(agg.mask2d_sum > 0,
                          agg.env2d_sum / np.maximum(agg.mask2d_sum, 1),
                          0.0).astype(np.float32)

        f1 = (agg.mask1d_sum / n).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            a1 = np.where(agg.mask1d_sum > 0,
                          agg.env1d_mask_sum / np.maximum(agg.mask1d_sum, 1),
                          0.0).astype(np.float32)

        results[season] = _SeasonClimatology(
            freq_2d=f2, ampl_2d=a2, freq_1d=f1, ampl_1d=a1, n_times=n)
        print(f"  n_times={n}  F_1d peak={f1.max()*100:.1f}%  "
              f"E_1d peak={a1.max():.2f} m/s")

    output_directory.mkdir(parents=True, exist_ok=True)
    out_path = output_directory / output_filename

    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as out:
        out.createDimension("lat", NLAT)
        out.createDimension("lon", NLON)

        vlat = out.createVariable("lat", "f4", ("lat",))
        vlat.units = "degrees_north"
        vlat[:] = LATS.astype(np.float32)
        vlon = out.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"
        vlon[:] = LONS.astype(np.float32)

        for season, r in results.items():
            grp = out.createGroup(season)
            for name, arr, lname, unit, dims in [
                ("freq_2d", r.freq_2d,
                 f"Climatological RWP frequency 2D ({season})", "1", ("lat", "lon")),
                ("ampl_2d", r.ampl_2d,
                 f"Climatological RWP amplitude 2D ({season})", "m s-1", ("lat", "lon")),
                ("freq_1d", r.freq_1d,
                 f"{season} 1D freq, cos-wgt 20-80N per time, Q&J Eq.1",
                 "1", ("lon",)),
                ("ampl_1d", r.ampl_1d,
                 f"{season} 1D amplitude, cos-wgt 20-80N per time, Q&J Eq.1",
                 "m s-1", ("lon",)),
            ]:
                v = grp.createVariable(name, "f4", dims,
                                       zlib=True, complevel=4)
                v.long_name = lname
                v.units = unit
                v[:] = arr
            grp.n_times = np.int32(r.n_times)
            grp.months = str(SEASONS[season])

        out.source = str(input_directory) + "/rwp_envelope_YYYY_MM.nc"
        out.year_start = np.int32(year_start)
        out.year_end = np.int32(year_end)
        out.lat_min = np.float32(lat_min)
        out.lat_max = np.float32(lat_max)
        out.created = datetime.datetime.utcnow().isoformat()

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    typer.run(main)
