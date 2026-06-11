"""Build seasonal longitude-only climatologies of the LWA budget strips.

Aggregates the monthly budget_strips_{source}_{year}_{month:02d}.nc files
produced by downstream_et_lwa.budget.strips into per-season longitude
climatologies written to a single NetCDF with one group per season.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Annotated

import netCDF4
import numpy as np
import typer

_LOG = logging.getLogger(__name__)

NLON = 360
LONS = np.linspace(0, 359, NLON)

BASE_TERMS = ("lwa_raw", "tendency", "termI", "termII", "termIII", "residual")
BASE_UNITS = {
    "lwa_raw": "m s-1",
    "tendency": "m s-1 day-1",
    "termI": "m s-1 day-1",
    "termII": "m s-1 day-1",
    "termIII": "m s-1 day-1",
    "residual": "m s-1 day-1",
}

SEASONS = {
    "JJASON": [6, 7, 8, 9, 10, 11],
    "DJFMA": [12, 1, 2, 3, 4],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "ANN": list(range(1, 13)),
}


def accumulate_season(
    *,
    strip_directory: Path,
    source_name: str,
    season_months: list[int],
    year_start: int,
    year_end: int,
    extra_terms: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    all_terms = list(BASE_TERMS) + list(extra_terms)
    sums = {t: np.zeros(NLON, dtype=np.float64) for t in all_terms}
    counts = {t: 0 for t in all_terms}
    for year in range(year_start, year_end + 1):
        for month in season_months:
            path = strip_directory / f"budget_strips_{source_name}_{year}_{month:02d}.nc"
            if not path.exists():
                continue
            with netCDF4.Dataset(path, "r") as d:
                nt = d["time"].shape[0]
                for t in all_terms:
                    if t in d.variables:
                        arr = np.asarray(d[t][:])
                        sums[t] += arr.sum(axis=0)
                        counts[t] += nt
    return sums, counts


def main(
    strip_directory: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    source_name: Annotated[str, typer.Option()] = "era5",
    year_start: Annotated[int, typer.Option()] = 2000,
    year_end: Annotated[int, typer.Option()] = 2022,
    seasons: Annotated[list[str] | None, typer.Option()] = None,
    extra_terms: Annotated[list[str] | None, typer.Option()] = None,
    output_filename: Annotated[str | None, typer.Option()] = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seasons_list = seasons if seasons else ["JJASON", "DJFMA", "ANN"]
    for season in seasons_list:
        if season not in SEASONS:
            raise typer.BadParameter(f"Unknown season {season}; choices: {list(SEASONS)}")
    extra_list = extra_terms if extra_terms else ["lh_lwa", "nonqg_lwa"]

    out_file = output_directory / (
        output_filename
        if output_filename
        else f"lwa_budget_climatology_{source_name}.nc"
    )

    results: dict[str, tuple[dict[str, int], dict[str, np.ndarray]]] = {}
    for season in seasons_list:
        months = SEASONS[season]
        print(
            f"[{source_name} / {season}] months={months} "
            f"years={year_start}-{year_end}..."
        )
        sums, counts = accumulate_season(
            strip_directory=strip_directory,
            source_name=source_name,
            season_months=months,
            year_start=year_start,
            year_end=year_end,
            extra_terms=extra_list,
        )
        if max(counts.values()) == 0:
            print(f"  no data for {season}, skipping")
            continue
        means = {t: (sums[t] / max(counts[t], 1)).astype(np.float32) for t in sums}
        results[season] = (counts, means)
        residual_peak = float(np.abs(means.get("residual", np.zeros(NLON))).max())
        print(
            f"  n_times(max)={max(counts.values())}  "
            f"residual peak={residual_peak:.2f}"
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(out_file, "w", format="NETCDF4") as out:
        out.createDimension("lon", NLON)
        vlon = out.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"
        vlon[:] = LONS.astype(np.float32)
        for season, (counts_dict, means_dict) in results.items():
            grp = out.createGroup(season)
            grp.months = str(SEASONS[season])
            for t, arr in means_dict.items():
                v = grp.createVariable(t, "f4", ("lon",), zlib=True, complevel=4)
                v.long_name = f"{season} climatology of {t}"
                v.units = BASE_UNITS.get(t, "m s-1 day-1")
                v.n_times = np.int32(counts_dict[t])
                v[:] = arr
        out.source_label = source_name
        out.year_start = np.int32(year_start)
        out.year_end = np.int32(year_end)
        out.created = datetime.datetime.utcnow().isoformat()
    print(f"\nWrote: {out_file}")


if __name__ == "__main__":
    typer.run(main)
