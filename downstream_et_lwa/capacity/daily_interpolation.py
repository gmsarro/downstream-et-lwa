"""Cubic interpolation of monthly-climatological carrying-capacity fields to daily dates."""

from __future__ import annotations

import calendar
import datetime
import logging
from pathlib import Path
from typing import Any, Optional

import netCDF4
import numpy as np
import scipy.interpolate
import typer
from typing_extensions import Annotated

_LOG = logging.getLogger(__name__)

MID_MONTH_DOY = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])


def interpolate_monthly_to_daily(
    *, monthly: np.ndarray, year: int, clip_negative: bool = True
) -> np.ndarray:
    nd = 366 if calendar.isleap(year) else 365
    mid_ext = np.concatenate(
        [MID_MONTH_DOY[-1:] - 365, MID_MONTH_DOY, MID_MONTH_DOY[:1] + 365]
    )
    days = np.arange(nd)
    daily = np.empty((nd, *monthly.shape[1:]))
    for idx in np.ndindex(monthly.shape[1:]):
        index: tuple[Any, ...] = (slice(None), *idx)
        vals = monthly[index]
        vals_ext = np.concatenate([vals[-1:], vals, vals[:1]])
        f = scipy.interpolate.interp1d(
            mid_ext, vals_ext, kind="cubic", fill_value="extrapolate"
        )
        daily[index] = f(days)
    if clip_negative:
        daily = np.maximum(daily, 0)
    return daily


def interpolate_to_dates(
    *, monthly: np.ndarray, dates: list[datetime.date], clip_negative: bool = True
) -> np.ndarray:
    out = np.empty((len(dates), *monthly.shape[1:]))
    by_year: dict[int, list[int]] = {}
    for pos, d in enumerate(dates):
        by_year.setdefault(d.year, []).append(pos)
    for year, positions in by_year.items():
        daily = interpolate_monthly_to_daily(
            monthly=monthly, year=year, clip_negative=clip_negative
        )
        for pos in positions:
            doy = (dates[pos] - datetime.date(year, 1, 1)).days
            out[pos] = daily[doy]
    return out


def read_dates_file(*, path: Path) -> list[datetime.date]:
    dates = []
    with open(path) as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            dates.append(datetime.date.fromisoformat(text))
    return dates


def write_daily_netcdf(
    *,
    path: Path,
    dates: list[datetime.date],
    values: np.ndarray,
    variable: str,
    latitude: np.ndarray,
    longitude: Optional[np.ndarray],
) -> None:
    epoch = datetime.date(1900, 1, 1)
    with netCDF4.Dataset(str(path), "w", clobber=True) as ds:
        ds.createDimension("time", len(dates))
        ds.createDimension("latitude", values.shape[1])
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = "days since 1900-01-01"
        tv.calendar = "standard"
        tv[:] = np.array([(d - epoch).days for d in dates], dtype=np.float64)
        lv = ds.createVariable("latitude", "f4", ("latitude",))
        lv.units = "degrees_north"
        lv[:] = latitude
        dims: tuple[str, ...] = ("time", "latitude")
        if values.ndim == 3:
            assert longitude is not None
            ds.createDimension("longitude", values.shape[2])
            gv = ds.createVariable("longitude", "f4", ("longitude",))
            gv.units = "degrees_east"
            gv[:] = longitude
            dims = ("time", "latitude", "longitude")
        v = ds.createVariable(
            variable, "f4", dims, zlib=True, complevel=4, fill_value=np.nan
        )
        v[:] = values.astype(np.float32)
        v.long_name = (
            f"{variable} cubically interpolated from monthly climatology to daily dates"
        )
        ds.method = (
            "12 monthly values anchored at mid-month day-of-year "
            "[15,46,74,105,135,166,196,227,258,288,319,349], periodic wrap "
            "extension, scipy interp1d kind=cubic with extrapolation"
        )
        ds.references = "Barpanda & Nakamura (2025, JAS)"


def main(
    params_file: Annotated[
        Path,
        typer.Option(help="Monthly-climatology NPZ ({label}_monthly_clim_params.npz)"),
    ],
    output_directory: Annotated[
        Path, typer.Option(help="Directory for the output NetCDF")
    ],
    variable: Annotated[
        str, typer.Option(help="NPZ key to interpolate (e.g. Fc, Ac, C, A0, alpha_zm)")
    ] = "Fc",
    dates_file: Annotated[
        Optional[Path],
        typer.Option(help="Text file with one ISO date (YYYY-MM-DD) per line"),
    ] = None,
    year: Annotated[
        Optional[int], typer.Option(help="Interpolate to every day of this year")
    ] = None,
    clip_negative: Annotated[
        bool, typer.Option(help="Clip interpolated values at zero")
    ] = True,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if (dates_file is None) == (year is None):
        raise typer.BadParameter("Provide exactly one of --dates-file or --year")

    if dates_file is not None:
        dates = read_dates_file(path=dates_file)
    else:
        assert year is not None
        nd = 366 if calendar.isleap(year) else 365
        start = datetime.date(year, 1, 1)
        dates = [start + datetime.timedelta(days=k) for k in range(nd)]

    with np.load(params_file) as npz:
        if variable not in npz.files:
            raise typer.BadParameter(
                f"{variable} not in {params_file} (available: {sorted(npz.files)})"
            )
        monthly = np.array(npz[variable], dtype=np.float64)
        if monthly.ndim < 2 or monthly.shape[0] != 12:
            raise typer.BadParameter(
                f"{variable} must have shape (12, lat[, lon]); got {monthly.shape}"
            )
        if "latitude" in npz.files:
            latitude = np.array(npz["latitude"])
        else:
            latitude = np.linspace(0, 90, monthly.shape[1])
        longitude = None
        if monthly.ndim == 3:
            if "longitude" in npz.files:
                longitude = np.array(npz["longitude"])
            else:
                longitude = np.linspace(0, 359, monthly.shape[2])

    values = interpolate_to_dates(
        monthly=monthly, dates=dates, clip_negative=clip_negative
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    out_path = output_directory / f"{params_file.stem}_{variable}_daily.nc"
    write_daily_netcdf(
        path=out_path,
        dates=dates,
        values=values,
        variable=variable,
        latitude=latitude,
        longitude=longitude,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    typer.run(main)
