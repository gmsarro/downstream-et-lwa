"""DJF-pooled carrying capacity from MERRA-2 BARO_N files (legacy single-season recipe)."""

from __future__ import annotations

import calendar
import logging
from pathlib import Path
from typing import Optional, TypedDict

import netCDF4
import numpy as np
import typer
from typing_extensions import Annotated

_LOG = logging.getLogger(__name__)

NLAT = 91
NLON = 360
LATITUDE = np.linspace(0, 90, NLAT)
LONGITUDE = np.linspace(0, 359, NLON)
COSPHI = np.cos(np.deg2rad(LATITUDE))

ZONAL_WINDOW = 15
MIN_SAMPLES = 50
VAR_FLOOR = 1e-6
CORR_THR = 0.5
MIN_ZM_GRIDPTS = 10
ALPHA_FLOOR = 1e-6

DJF_MONTHS = (1, 2, 12)


class MonthData(TypedDict):
    lwa: np.ndarray
    ue: np.ndarray
    f_linear: np.ndarray
    ndays: int


def zonal_smooth(*, field: np.ndarray, window_deg: int = ZONAL_WINDOW) -> np.ndarray:
    hw = window_deg // 2
    out = np.empty_like(field)
    for i in range(NLON):
        idx = np.arange(i - hw, i + hw + 1) % NLON
        out[..., i] = np.nanmean(field[..., idx], axis=-1)
    return out


def load_month_daily(
    *, baro_directory: Path, year: int, month: int
) -> Optional[MonthData]:
    ms = f"{month:02d}"
    mo_idx = month - 1

    f_lwa = baro_directory / f"{year}_{ms}_LWAb_N.nc"
    f_u = baro_directory / f"{year}_{ms}_Ub_N.nc"
    f_uref = baro_directory / f"{year}_{ms}_Urefb_N.nc"
    f_ep1 = baro_directory / f"{year}_{ms}_ep1_N.nc"
    f_ep3 = baro_directory / f"{year}_{ms}_ep3_N.nc"

    for f in (f_lwa, f_u, f_uref, f_ep1, f_ep3):
        if not f.exists():
            return None

    ndays = calendar.monthrange(year, month)[1]

    with netCDF4.Dataset(str(f_lwa), "r") as ds:
        lwa_6h = np.array(ds.variables["lwa"][: ndays * 4, mo_idx, :, :])
    with netCDF4.Dataset(str(f_u), "r") as ds:
        u_6h = np.array(ds.variables["u"][: ndays * 4, mo_idx, :, :])
    with netCDF4.Dataset(str(f_uref), "r") as ds:
        uref_6h = np.array(ds.variables["uref"][: ndays * 4, mo_idx, :])
    with netCDF4.Dataset(str(f_ep1), "r") as ds:
        ep1_6h = np.array(ds.variables["ep1"][: ndays * 4, mo_idx, :, :])
    with netCDF4.Dataset(str(f_ep3), "r") as ds:
        ep3_6h = np.array(ds.variables["ep3"][: ndays * 4, mo_idx, :, :])

    lwa = lwa_6h.reshape(ndays, 4, NLAT, NLON).mean(1)
    u = u_6h.reshape(ndays, 4, NLAT, NLON).mean(1)
    uref_raw = uref_6h.reshape(ndays, 4, NLAT).mean(1)
    uref = uref_raw[:, :, np.newaxis] * np.ones((1, 1, NLON))
    ep1 = ep1_6h.reshape(ndays, 4, NLAT, NLON).mean(1)
    ep3 = ep3_6h.reshape(ndays, 4, NLAT, NLON).mean(1)

    ue = u - uref
    f_linear = ep1 + ep3

    return MonthData(lwa=lwa, ue=ue, f_linear=f_linear, ndays=ndays)


def find_djf_months(*, baro_directory: Path) -> list[tuple[int, int]]:
    out = []
    for f in sorted(baro_directory.glob("*_LWAb_N.nc")):
        base = f.name.replace("_LWAb_N.nc", "")
        yr_s, mo_s = base.split("_")
        yr, mo = int(yr_s), int(mo_s)
        if mo in DJF_MONTHS:
            out.append((yr, mo))
    return out


def compute_gridpoint_regressions(
    *,
    a_anom: np.ndarray,
    ue_anom: np.ndarray,
    flin: np.ndarray,
    a_total: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_local = np.full((NLAT, NLON), np.nan)
    corr_local = np.full((NLAT, NLON), np.nan)
    cdopp_local = np.full((NLAT, NLON), np.nan)

    for j in range(NLAT):
        for i in range(NLON):
            a = a_anom[:, j, i]
            u = ue_anom[:, j, i]
            fl = flin[:, j, i]
            at = a_total[:, j, i]

            ok = np.isfinite(a) & np.isfinite(u) & (a != 0)
            if ok.sum() < MIN_SAMPLES:
                continue

            a_ok, u_ok = a[ok], u[ok]
            var_a = np.var(a_ok)
            if var_a < VAR_FLOOR:
                continue

            cov_ua = np.mean(u_ok * a_ok) - np.mean(u_ok) * np.mean(a_ok)
            alpha_local[j, i] = -cov_ua / var_a

            std_u = np.std(u_ok)
            std_a = np.std(a_ok)
            if std_u > 0 and std_a > 0:
                corr_local[j, i] = cov_ua / (std_u * std_a)

            ok2 = np.isfinite(fl) & np.isfinite(at) & (at != 0)
            if ok2.sum() < MIN_SAMPLES:
                continue
            fl_ok, at_ok = fl[ok2], at[ok2]
            var_at = np.var(at_ok)
            if var_at < VAR_FLOOR:
                continue
            cov_fa = np.mean(fl_ok * at_ok) - np.mean(fl_ok) * np.mean(at_ok)
            cdopp_local[j, i] = cov_fa / var_at

    return alpha_local, corr_local, cdopp_local


def compute_zonal_means(
    *,
    alpha_local: np.ndarray,
    corr_local: np.ndarray,
    cdopp_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_zm = np.full(NLAT, np.nan)
    cdopp_zm = np.full(NLAT, np.nan)
    corr_zm = np.full(NLAT, np.nan)

    for j in range(NLAT):
        mask = (alpha_local[j, :] > 0) & (np.abs(corr_local[j, :]) > CORR_THR)
        if mask.sum() > MIN_ZM_GRIDPTS:
            alpha_zm[j] = np.nanmean(alpha_local[j, mask])
            cdopp_zm[j] = np.nanmean(cdopp_local[j, mask])
            corr_zm[j] = np.nanmean(corr_local[j, mask])

    return alpha_zm, cdopp_zm, corr_zm


def main(
    baro_directory: Annotated[
        Path,
        typer.Option(help="MERRA-2 BARO_N directory with YYYY_MM_*_N.nc monthly files"),
    ],
    output_directory: Annotated[
        Path, typer.Option(help="Directory for the output NPZ")
    ],
    snapshot_year: Annotated[
        int, typer.Option(help="Year of the single-month snapshot Fc")
    ] = 2014,
    snapshot_month: Annotated[
        int, typer.Option(help="Month of the single-month snapshot Fc")
    ] = 11,
    era5_reference_file: Annotated[
        Optional[Path],
        typer.Option(help="ERA5 reference carrying-capacity NetCDF for comparison table"),
    ] = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("=== MERRA2 end-to-end carrying capacity computation ===", flush=True)

    djf_months = find_djf_months(baro_directory=baro_directory)
    print(f"Found {len(djf_months)} DJF months", flush=True)

    all_a_anom = []
    all_ue_anom = []
    all_flin = []
    all_a_total = []
    all_a_mean = []

    for yr, mo in djf_months:
        data = load_month_daily(baro_directory=baro_directory, year=yr, month=mo)
        if data is None:
            print(f"  Loading {yr}_{mo:02d} ... MISSING, skip")
            continue

        lwa = data["lwa"]
        ue = data["ue"]
        flin = data["f_linear"]
        nd = data["ndays"]

        for t in range(nd):
            lwa[t] = zonal_smooth(field=lwa[t])
            ue[t] = zonal_smooth(field=ue[t])
            flin[t] = zonal_smooth(field=flin[t])

        lwa_mean = lwa.mean(0)
        ue_mean = ue.mean(0)

        all_a_anom.append(lwa - lwa_mean)
        all_ue_anom.append(ue - ue_mean)
        all_flin.append(flin)
        all_a_total.append(lwa)
        all_a_mean.append(lwa_mean)

        print(f"  Loading {yr}_{mo:02d} ... OK ({nd} days)", flush=True)

    print(f"\nPooling {len(all_a_anom)} months ...", flush=True)
    a_anom = np.concatenate(all_a_anom, axis=0)
    ue_anom = np.concatenate(all_ue_anom, axis=0)
    flin_all = np.concatenate(all_flin, axis=0)
    a_total = np.concatenate(all_a_total, axis=0)
    a0_djf = np.mean(all_a_mean, axis=0)

    print(f"Total days pooled: {a_anom.shape[0]}", flush=True)

    print("Computing gridpoint regressions ...", flush=True)
    alpha_local, corr_local, cdopp_local = compute_gridpoint_regressions(
        a_anom=a_anom, ue_anom=ue_anom, flin=flin_all, a_total=a_total
    )
    alpha_zm, cdopp_zm, corr_zm = compute_zonal_means(
        alpha_local=alpha_local, corr_local=corr_local, cdopp_local=cdopp_local
    )

    print("Computing Fc ...", flush=True)
    cos2 = COSPHI**2
    c_djf = cdopp_zm[:, None] - 2 * alpha_zm[:, None] * a0_djf
    fc_djf = cos2[:, None] * c_djf**2 / (4 * alpha_zm[:, None])
    ac_djf = c_djf / (2 * alpha_zm[:, None])

    ok = np.isfinite(fc_djf) & (np.abs(alpha_zm[:, None]) > ALPHA_FLOOR)
    fc_djf = np.where(ok, fc_djf, np.nan)
    ac_djf = np.where(ok, ac_djf, np.nan)

    data_snap = load_month_daily(
        baro_directory=baro_directory, year=snapshot_year, month=snapshot_month
    )
    if data_snap is not None:
        a0_snap = data_snap["lwa"].mean(0)
        for j in range(NLAT):
            a0_snap[j] = zonal_smooth(field=a0_snap[j][None, :])[0]
    else:
        a0_snap = a0_djf

    c_snap = cdopp_zm[:, None] - 2 * alpha_zm[:, None] * a0_snap
    fc_snap = cos2[:, None] * c_snap**2 / (4 * alpha_zm[:, None])
    ac_snap = c_snap / (2 * alpha_zm[:, None])
    fc_snap = np.where(ok, fc_snap, np.nan)
    ac_snap = np.where(ok, ac_snap, np.nan)

    output_directory.mkdir(parents=True, exist_ok=True)
    out_path = output_directory / "merra2_carrying_capacity_params.npz"
    np.savez(
        out_path,
        alpha_zm=alpha_zm,
        cdopp_zm=cdopp_zm,
        corr_zm=corr_zm,
        alpha_local=alpha_local,
        corr_local=corr_local,
        cdopp_local=cdopp_local,
        A0_djf=a0_djf,
        A0_nov=a0_snap,
        Fc_djf=fc_djf,
        Ac_djf=ac_djf,
        Fc_nov=fc_snap,
        Ac_nov=ac_snap,
        latitude=LATITUDE,
        longitude=LONGITUDE,
    )
    print(f"Saved: {out_path}")

    if era5_reference_file is None:
        return

    with netCDF4.Dataset(str(era5_reference_file), "r") as ds_era5:
        e5_alpha = np.array(ds_era5.variables["alpha_zm"][:]).squeeze()
        e5_cdopp = np.array(ds_era5.variables["cdopp_zm"][:]).squeeze()
        e5_fc_djf = np.array(ds_era5.variables["Fc_djf"][:]).squeeze()

    print("\n=== MERRA2 vs ERA5 parameter comparison ===")
    print(
        f"{'Lat':>5s}  {'a_M2':>7s} {'a_E5':>7s}  {'cd_M2':>7s} {'cd_E5':>7s}  "
        f"{'Fc_M2':>7s} {'Fc_E5':>7s}"
    )
    for j in (25, 30, 35, 40, 45, 50, 55, 60, 65, 70):
        jj = int(np.argmin(np.abs(LATITUDE - j)))
        print(
            f"{j:>4d}N  {alpha_zm[jj]:7.3f} {e5_alpha[jj]:7.3f}  "
            f"{cdopp_zm[jj]:7.1f} {e5_cdopp[jj]:7.1f}  "
            f"{np.nanmean(fc_djf[jj, :]):7.1f} {np.nanmean(e5_fc_djf[jj, :]):7.1f}"
        )


if __name__ == "__main__":
    typer.run(main)
