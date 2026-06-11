"""Ghinassi-style CWT-filtered LWA envelope and RWP mask from barotropic LWA and 250-hPa meridional wind."""
from __future__ import annotations

import concurrent.futures
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

WAVENUMBERS = np.arange(2, 16)
CWT_BANDWIDTH = 1.0
TAU_STAR_DEFAULT = 1.9
LWA_FILENAME_PATTERN = "{year}_LWAb_N.nc"
V250_FILENAME_PATTERN = "{month:02d}_{year}.nc"
OUTPUT_FILENAME_PATTERN = "lwa_filtered_{year}_{month:02d}.nc"


def _cwt_bank_fourier(*, nlon: int, wavenumbers: np.ndarray = WAVENUMBERS,
                      sigma: float = CWT_BANDWIDTH) -> np.ndarray:
    K = np.fft.fftfreq(nlon, d=1.0 / nlon)
    bank = np.zeros((len(wavenumbers), nlon), dtype=np.float64)
    for i, k in enumerate(wavenumbers):
        g = np.exp(-((K - k) ** 2) / (2.0 * sigma ** 2))
        g[K <= 0] = 0.0
        bank[i] = 2.0 * g
    return bank


def dominant_wavenumber(*, v_field: np.ndarray,
                        wavenumbers: np.ndarray = WAVENUMBERS,
                        cwt_bank: np.ndarray | None = None) -> np.ndarray:
    v = np.where(np.isfinite(v_field), v_field, 0.0).astype(np.float64)
    nlon = v.shape[-1]
    if cwt_bank is None:
        cwt_bank = _cwt_bank_fourier(nlon=nlon, wavenumbers=wavenumbers)

    V = np.fft.fft(v, axis=-1)
    power = np.empty(cwt_bank.shape[:1] + v.shape, dtype=np.float64)
    for i in range(cwt_bank.shape[0]):
        W = np.fft.ifft(V * cwt_bank[i], axis=-1)
        power[i] = (W.real ** 2 + W.imag ** 2)

    idx = np.argmax(power, axis=0)
    sd = np.asarray(wavenumbers)[idx].astype(np.uint8)
    return sd


def _hann_kernel_fourier(*, nlon: int, fwhm_pts: int) -> np.ndarray:
    L = max(3, 2 * int(fwhm_pts) + 1)
    if L > nlon:
        L = nlon - (nlon % 2 == 0)
    hann = np.hanning(L)
    hann /= hann.sum()
    half = L // 2
    pad = np.zeros(nlon, dtype=np.float64)
    pad[:half + 1] = hann[half:]
    pad[-half:] = hann[:half]
    return np.fft.fft(pad)


def ghinassi_filter(*, lwa: np.ndarray, sd: np.ndarray,
                    wavenumbers: np.ndarray = WAVENUMBERS) -> np.ndarray:
    nlon = lwa.shape[-1]
    lwa_clean = np.where(np.isfinite(lwa), lwa, 0.0).astype(np.float64)
    LWA_k = np.fft.fft(lwa_clean, axis=-1)
    out = np.zeros_like(lwa_clean)

    for s in wavenumbers:
        fwhm_deg = 360.0 / float(s)
        H = _hann_kernel_fourier(nlon=nlon, fwhm_pts=int(round(fwhm_deg)))
        smoothed = np.fft.ifft(LWA_k * H, axis=-1).real
        mask_s = (sd == s)
        out[mask_s] = smoothed[mask_s]

    return out.astype(np.float32)


def hemispheric_mean(*, field: np.ndarray) -> np.ndarray:
    weights = COSPHI[:, None]
    w_sum = weights.sum() * NLON
    return (field * weights).sum(axis=(-2, -1)) / w_sum


def _days_in_month(*, year: int, month: int) -> int:
    if month == 12:
        return (datetime.datetime(year + 1, 1, 1) - datetime.datetime(year, 12, 1)).days
    return (datetime.datetime(year, month + 1, 1) - datetime.datetime(year, month, 1)).days


def _load_lwa_month(*, year: int, month: int, lwa_directory: Path) -> np.ndarray:
    path = lwa_directory / LWA_FILENAME_PATTERN.format(year=year)
    n_t = _days_in_month(year=year, month=month) * 4
    with netCDF4.Dataset(path, "r") as d:
        lwa = np.asarray(d["lwa"][:n_t, month - 1, :, :], dtype=np.float32)
    return lwa


def _load_v250_month(*, year: int, month: int, v250_directory: Path) -> np.ndarray | None:
    path = v250_directory / V250_FILENAME_PATTERN.format(month=month, year=year)
    if not path.exists():
        return None
    with netCDF4.Dataset(path, "r") as d:
        v = np.asarray(d["v250"][:], dtype=np.float32)
    return v


def _month_times(*, year: int, month: int) -> np.ndarray:
    t0 = datetime.datetime(year, month, 1)
    n_t = _days_in_month(year=year, month=month) * 4
    return np.array([(t0 + datetime.timedelta(hours=6 * i)).timestamp()
                     for i in range(n_t)], dtype=np.float64)


def _process_month(*, year: int, month: int, lwa_directory: Path,
                   v250_directory: Path, output_directory: Path,
                   tau_star: float = TAU_STAR_DEFAULT,
                   overwrite: bool = False) -> str:
    out_path = output_directory / OUTPUT_FILENAME_PATTERN.format(year=year, month=month)
    if out_path.exists() and not overwrite:
        return f"[skip] exists: {out_path.name}"

    output_directory.mkdir(parents=True, exist_ok=True)

    try:
        lwa = _load_lwa_month(year=year, month=month, lwa_directory=lwa_directory)
    except Exception as exc:
        _LOG.exception("No LWA input for %04d-%02d", year, month)
        return f"[err]  {year}-{month:02d}  no LWA input: {exc}"
    v = _load_v250_month(year=year, month=month, v250_directory=v250_directory)
    if v is None or v.shape[0] < lwa.shape[0]:
        return f"[err]  {year}-{month:02d}  missing/short v250 input"

    n_t = lwa.shape[0]
    v = v[:n_t]

    cwt_bank = _cwt_bank_fourier(nlon=NLON, wavenumbers=WAVENUMBERS, sigma=CWT_BANDWIDTH)
    sd = dominant_wavenumber(v_field=v, wavenumbers=WAVENUMBERS, cwt_bank=cwt_bank)

    lwa_f = ghinassi_filter(lwa=lwa, sd=sd, wavenumbers=WAVENUMBERS)

    hem = hemispheric_mean(field=lwa_f)
    tau = (tau_star * hem)[:, None, None]
    mask = (lwa_f > tau).astype(np.uint8)
    lwa_f_thr = np.where(mask.astype(bool), lwa_f, 0.0).astype(np.float32)

    times_s = _month_times(year=year, month=month)
    tmp_path = out_path.with_suffix(".nc.tmp")
    with netCDF4.Dataset(tmp_path, "w", format="NETCDF4") as out:
        out.createDimension("time", n_t)
        out.createDimension("lat", NLAT)
        out.createDimension("lon", NLON)

        vt = out.createVariable("time", "f8", ("time",))
        vt.units = "seconds since 1970-01-01"
        vt[:] = times_s

        vlat = out.createVariable("lat", "f4", ("lat",))
        vlat.units = "degrees_north"
        vlat[:] = LATS.astype(np.float32)
        vlon = out.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"
        vlon[:] = LONS.astype(np.float32)

        def _mk(*, name: str, arr: np.ndarray, long_name: str, units: str,
                dtype: str = "f4") -> None:
            var = out.createVariable(name, dtype, ("time", "lat", "lon"),
                                     zlib=True, complevel=4, shuffle=True,
                                     chunksizes=(1, NLAT, NLON))
            var.long_name = long_name
            var.units = units
            var[:] = arr

        _mk(name="lwa_raw", arr=lwa,
            long_name="Raw barotropic LWA (NHN)", units="m s-1")
        _mk(name="lwa_filt", arr=lwa_f,
            long_name="Ghinassi (2018) zonally-filtered LWA", units="m s-1")
        _mk(name="lwa_filt_thr", arr=lwa_f_thr,
            long_name=f"Thresholded filtered LWA (tau={tau_star}*<LWA_f>_hem)",
            units="m s-1")
        _mk(name="mask", arr=mask,
            long_name="Binary mask where filtered LWA exceeds threshold",
            units="1", dtype="u1")
        _mk(name="sd", arr=sd,
            long_name="Local dominant zonal wavenumber (Morlet CWT of v250)",
            units="1", dtype="u1")

        vh = out.createVariable("hem_mean_filt", "f4", ("time",))
        vh.long_name = "Cos(lat)-weighted hemispheric mean of filtered LWA"
        vh.units = "m s-1"
        vh[:] = hem.astype(np.float32)

        out.source_lwa = str(lwa_directory / LWA_FILENAME_PATTERN.format(year=year))
        out.source_v250 = str(v250_directory / V250_FILENAME_PATTERN.format(month=month, year=year))
        out.method = (
            "Ghinassi et al. 2018, sec. 2c: local dominant zonal wavenumber "
            "s_d(lambda, phi) from Morlet CWT of v250; Hann-filter LWA "
            "zonally with FWHM = 360 deg / s_d. Threshold tau=tau_star*<LWA_f>_hem."
        )
        out.tau_star = np.float32(tau_star)
        out.wavenumbers_min = np.int32(WAVENUMBERS.min())
        out.wavenumbers_max = np.int32(WAVENUMBERS.max())

    tmp_path.replace(out_path)
    return f"[ok]   {out_path.name}  n_times={n_t}"


def main(
    lwa_directory: Annotated[Path, typer.Option()],
    v250_directory: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    year_start: Annotated[int, typer.Option()] = 2000,
    year_end: Annotated[int, typer.Option()] = 2023,
    months: Annotated[list[int] | None, typer.Option()] = None,
    tau_star: Annotated[float, typer.Option()] = TAU_STAR_DEFAULT,
    workers: Annotated[int, typer.Option()] = 8,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    logging.basicConfig(level=logging.INFO)
    month_values = months if months else list(range(1, 13))
    jobs = [(year, month) for year in range(year_start, year_end + 1)
            for month in month_values]
    print(f"Processing {len(jobs)} (year, month) jobs with {workers} workers.")
    print(f"Output dir: {output_directory}")
    print(f"tau_star={tau_star}  wavenumbers=[{WAVENUMBERS.min()}..{WAVENUMBERS.max()}]")
    output_directory.mkdir(parents=True, exist_ok=True)

    if workers <= 1:
        for year, month in jobs:
            print(_process_month(year=year, month=month,
                                 lwa_directory=lwa_directory,
                                 v250_directory=v250_directory,
                                 output_directory=output_directory,
                                 tau_star=tau_star, overwrite=overwrite),
                  flush=True)
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_process_month, year=year, month=month,
                            lwa_directory=lwa_directory,
                            v250_directory=v250_directory,
                            output_directory=output_directory,
                            tau_star=tau_star, overwrite=overwrite)
            for year, month in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    typer.run(main)
