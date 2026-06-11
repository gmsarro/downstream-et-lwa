"""Compute the material derivative of specific humidity Dq/Dt from 6-hourly data.

DqDt = dq/dt + u*dq/dx + v*dq/dy + omega*dq/dp on the native grid, with
centred differences in time (forward/backward at the ends), periodic centred
differences in longitude, centred differences in latitude, and centred
differences in pressure.  Processes in time chunks to bound memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import netCDF4
import numpy as np
import typer

import downstream_et_lwa.constants as constants

_LOG = logging.getLogger(__name__)


def _pick_name(*, dataset: netCDF4.Dataset, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in dataset.variables:
            return name
    return None


def main(
    input_file: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
    output_filename: Annotated[str | None, typer.Option()] = None,
    time_chunk: Annotated[int, typer.Option()] = 8,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_directory.mkdir(parents=True, exist_ok=True)
    out_path = output_directory / (
        output_filename if output_filename else f"{input_file.stem}_DqDt.nc"
    )
    print(f"Opening input file: {input_file}")

    with netCDF4.Dataset(input_file) as ds:
        t_name = _pick_name(dataset=ds, candidates=("valid_time", "time", "Time"))
        p_name = _pick_name(dataset=ds, candidates=("pressure_level", "level", "plev"))
        lat_name = "latitude" if "latitude" in ds.variables else "lat"
        lon_name = "longitude" if "longitude" in ds.variables else "lon"

        if t_name is None or p_name is None:
            raise RuntimeError(f"Missing coords: time={t_name} p={p_name}")

        time_var = ds.variables[t_name]
        time_vals = time_var[:]
        time_units = time_var.units
        time_cal = getattr(time_var, "calendar", "standard")

        p_hpa = ds.variables[p_name][:].astype(np.float64)
        lat = ds.variables[lat_name][:].astype(np.float64)
        lon = ds.variables[lon_name][:].astype(np.float64)

        nt = len(time_vals)
        nlev = len(p_hpa)
        nlat = len(lat)
        nlon = len(lon)

        print(
            f"Coords: time={t_name} ({nt}), p={p_name} ({nlev}), "
            f"lat ({nlat}), lon ({nlon})"
        )

        qn = _pick_name(dataset=ds, candidates=("q", "specific_humidity"))
        un = _pick_name(dataset=ds, candidates=("u", "ua", "u_wind"))
        vn = _pick_name(dataset=ds, candidates=("v", "va", "v_wind"))
        wn = _pick_name(dataset=ds, candidates=("w", "omega", "wap"))

        if qn is None or un is None or vn is None or wn is None:
            raise RuntimeError(f"Missing vars: q={qn} u={un} v={vn} w={wn}")
        print(f"Variables: q={qn}, u={un}, v={vn}, omega={wn}")

        dlon_rad = np.deg2rad(float(lon[1] - lon[0]))
        dlat_rad = np.deg2rad(float(lat[1] - lat[0]))
        a = constants.FALWA_PLANET_RADIUS_M

        cos_lat = np.cos(np.deg2rad(lat))
        cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)
        cos_lat_2d = cos_lat[:, np.newaxis]

        p_units = (ds.variables[p_name].units or "").lower()
        if "hpa" in p_units or np.max(p_hpa) < 2000:
            p_pa = p_hpa * 100.0
        else:
            p_pa = p_hpa

        if np.issubdtype(time_vals.dtype, np.datetime64):
            dt_s = np.diff(time_vals.astype("datetime64[s]").astype(np.float64))
        else:
            dt_s = np.diff(time_vals.astype(np.float64))
            if np.mean(dt_s) > 1e6:
                pass
            elif np.mean(dt_s) < 100:
                dt_s = dt_s * 3600

        print(f"Time spacing: ~{np.mean(dt_s):.0f} seconds")

        print(f"Creating output: {out_path}")
        with netCDF4.Dataset(out_path, "w", format="NETCDF4") as ncout:
            ncout.createDimension("time", nt)
            ncout.createDimension(p_name, nlev)
            ncout.createDimension(lat_name, nlat)
            ncout.createDimension(lon_name, nlon)

            t_out = ncout.createVariable("time", "f8", ("time",))
            t_out.units = time_units
            t_out.calendar = time_cal
            t_out[:] = time_vals

            p_out = ncout.createVariable(p_name, "f8", (p_name,))
            p_out.units = ds.variables[p_name].units
            p_out[:] = p_hpa

            lat_out = ncout.createVariable(lat_name, "f8", (lat_name,))
            lat_out.units = "degrees_north"
            lat_out[:] = lat

            lon_out = ncout.createVariable(lon_name, "f8", (lon_name,))
            lon_out.units = "degrees_east"
            lon_out[:] = lon

            dqdt_out = ncout.createVariable(
                "DqDt",
                "f4",
                ("time", p_name, lat_name, lon_name),
                zlib=True,
                complevel=4,
            )
            dqdt_out.units = "kg kg-1 s-1"
            dqdt_out.long_name = "material derivative of specific humidity"

            chunk = time_chunk
            print(f"Processing in chunks of {chunk} (with halo for time derivative)...")

            for t0 in range(0, nt, chunk):
                t1 = min(t0 + chunk, nt)
                th0 = max(t0 - 1, 0)
                th1 = min(t1 + 1, nt)
                print(f"  Chunk [{t0}:{t1}] / {nt} (halo [{th0}:{th1}])")

                q = ds.variables[qn][th0:th1, :, :, :].astype(np.float32)
                u = ds.variables[un][th0:th1, :, :, :].astype(np.float32)
                v = ds.variables[vn][th0:th1, :, :, :].astype(np.float32)
                omega = ds.variables[wn][th0:th1, :, :, :].astype(np.float32)

                if hasattr(q, "filled"):
                    q = q.filled(np.nan)
                if hasattr(u, "filled"):
                    u = u.filled(np.nan)
                if hasattr(v, "filled"):
                    v = v.filled(np.nan)
                if hasattr(omega, "filled"):
                    omega = omega.filled(np.nan)

                nch = th1 - th0

                dqdt = np.full_like(q, np.nan)
                for it in range(nch):
                    abs_t = th0 + it
                    if abs_t == 0:
                        dt_local = dt_s[0]
                        dqdt[it] = (q[min(it + 1, nch - 1)] - q[it]) / dt_local
                    elif abs_t == nt - 1:
                        dt_local = dt_s[-1]
                        dqdt[it] = (q[it] - q[max(it - 1, 0)]) / dt_local
                    else:
                        dt_local = dt_s[abs_t - 1] + dt_s[min(abs_t, len(dt_s) - 1)]
                        dqdt[it] = (q[min(it + 1, nch - 1)] - q[max(it - 1, 0)]) / dt_local

                dqdx = (np.roll(q, -1, axis=3) - np.roll(q, 1, axis=3)) / (
                    2.0 * dlon_rad * a * cos_lat_2d
                )

                dqdy = np.zeros_like(q)
                dqdy[:, :, 1:-1, :] = (q[:, :, 2:, :] - q[:, :, :-2, :]) / (
                    2.0 * dlat_rad * a
                )
                dqdy[:, :, 0, :] = np.nan
                dqdy[:, :, -1, :] = np.nan

                dqdp = np.zeros_like(q)
                for k in range(1, nlev - 1):
                    dp = p_pa[k + 1] - p_pa[k - 1]
                    dqdp[:, k, :, :] = (q[:, k + 1, :, :] - q[:, k - 1, :, :]) / dp
                dp0 = p_pa[1] - p_pa[0]
                dqdp[:, 0, :, :] = (q[:, 1, :, :] - q[:, 0, :, :]) / dp0
                dpn = p_pa[-1] - p_pa[-2]
                dqdp[:, -1, :, :] = (q[:, -1, :, :] - q[:, -2, :, :]) / dpn

                dqdt_total = dqdt + u * dqdx + v * dqdy + omega * dqdp

                lo = t0 - th0
                hi = lo + (t1 - t0)
                dqdt_out[t0:t1, :, :, :] = dqdt_total[lo:hi].astype(np.float32)

    print(f"DONE: {out_path}")


if __name__ == "__main__":
    typer.run(main)
