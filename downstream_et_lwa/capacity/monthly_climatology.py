"""Monthly-climatological jet carrying capacity Fc (BN25-canonical pooled regressions)."""

from __future__ import annotations

import calendar
import logging
from pathlib import Path
from typing import Any, Optional, TypedDict

import netCDF4
import numpy as np
import scipy.ndimage
import typer
from typing_extensions import Annotated

_LOG = logging.getLogger(__name__)

NLAT = 91
NLON = 360
LATITUDE = np.linspace(0, 90, NLAT)
LONGITUDE = np.linspace(0, 359, NLON)
COSPHI = np.cos(np.deg2rad(LATITUDE))
COS2 = COSPHI**2

LAT_LO = 20
LAT_HI = 80
ALPHA_FLOOR = 1e-6
VAR_FLOOR = 1e-4
CORR_THR = 0.3
CORR_THR_MONTH: Optional[float] = None
ZSMOOTH_DEG = 15
LAT_SMOOTH_DEG = 0
ALPHA_MAX_PHYS = 1.0
CDOPP_BLOWUP = 200.0
MIN_ZM_GRIDPTS = 10
CDOPP_MAX_PHYS = 40.0
LAT_FILL_WIN = 9
LAT_FILL_MIN = 4

_STEMS = {
    "lwa": ("LWAb_N", "lwa"),
    "u": ("Ub_N", "u"),
    "uref": ("Urefb_N", "uref"),
    "ua1": ("ua1_N", "ua1"),
    "ep1": ("ep1_N", "ep1"),
}


class MonthData(TypedDict):
    lwa: np.ndarray
    ue: np.ndarray
    f_linear: np.ndarray
    uref: np.ndarray
    ndays: int


def _resolve_baro_file(
    *, baro_directory: Path, year: int, month: int, suffix: str
) -> Optional[Path]:
    monthly = baro_directory / f"{year}_{month:02d}_{suffix}.nc"
    if monthly.exists():
        return monthly
    yearly = baro_directory / f"{year}_{suffix}.nc"
    if yearly.exists():
        return yearly
    matches = sorted(baro_directory.glob(f"*/BARO_N/{year}_{month:02d}_{suffix}.nc"))
    if matches:
        if len(matches) > 1:
            _LOG.warning(
                "%d BARO_N matches for %d_%02d_%s.nc; using %s",
                len(matches),
                year,
                month,
                suffix,
                matches[0],
            )
        return matches[0]
    return None


def _read_month_block(
    *, path: Path, var: str, year: int, month: int
) -> Optional[np.ndarray]:
    nd = calendar.monthrange(year, month)[1]
    nt = nd * 4
    m_idx = month - 1
    with netCDF4.Dataset(str(path), "r") as ds:
        raw_var = ds.variables[var]
        is_1d = raw_var.ndim == 3
        raw = np.array(raw_var[:nt, m_idx, :] if is_1d else raw_var[:nt, m_idx, :, :])
    if raw.shape[0] < nt:
        return None
    return raw.reshape(nd, 4, *raw.shape[1:]).mean(1)


def load_month_daily(
    *, baro_directory: Path, year: int, month: int
) -> Optional[MonthData]:
    paths: dict[str, Path] = {}
    for key, (suffix, _) in _STEMS.items():
        path = _resolve_baro_file(
            baro_directory=baro_directory, year=year, month=month, suffix=suffix
        )
        if path is None:
            return None
        paths[key] = path
    nd = calendar.monthrange(year, month)[1]
    fields: dict[str, np.ndarray] = {}
    for key, (_, var) in _STEMS.items():
        block = _read_month_block(path=paths[key], var=var, year=year, month=month)
        if block is None:
            return None
        fields[key] = block
    lwa = fields["lwa"]
    u = fields["u"]
    uref = fields["uref"]
    ua1 = fields["ua1"]
    ep1 = fields["ep1"]
    zero_day = np.all(lwa == 0.0, axis=(1, 2))
    n_trunc = int(zero_day.sum())
    if n_trunc > 0:
        lwa = np.where(zero_day[:, None, None], np.nan, lwa)
        u = np.where(zero_day[:, None, None], np.nan, u)
        ua1 = np.where(zero_day[:, None, None], np.nan, ua1)
        ep1 = np.where(zero_day[:, None, None], np.nan, ep1)
        uref = np.where(zero_day[:, None], np.nan, uref)
    ue = u - uref[:, :, np.newaxis]
    return MonthData(
        lwa=lwa, ue=ue, f_linear=ua1 + ep1, uref=uref, ndays=nd - n_trunc
    )


def load_stationary_a0(*, path: Path) -> np.ndarray:
    with netCDF4.Dataset(str(path), "r") as ds:
        return np.array(ds.variables["lwa"][:])


def zsmooth(*, x: np.ndarray, deg: float = ZSMOOTH_DEG) -> np.ndarray:
    if deg <= 1:
        return x
    return scipy.ndimage.uniform_filter1d(x, int(round(deg)), axis=-1, mode="wrap")


def lat_smooth(*, x: np.ndarray, deg: float = 0) -> np.ndarray:
    if deg <= 1:
        return x
    axis = -2 if x.ndim >= 2 and x.shape[-1] == NLON else -1
    return scipy.ndimage.uniform_filter1d(x, int(round(deg)), axis=axis, mode="nearest")


def _lat_med_fill(
    *, profile: np.ndarray, win: int = LAT_FILL_WIN, min_finite: int = LAT_FILL_MIN
) -> np.ndarray:
    out = profile.copy().astype(np.float64)
    half = win // 2
    for ax in np.ndindex(profile.shape[:-1]):
        row = profile[ax].astype(np.float64)
        sm = np.full_like(row, np.nan)
        n = len(row)
        for j in range(n):
            lo, hi = max(0, j - half), min(n, j + half + 1)
            w = row[lo:hi]
            finite = w[np.isfinite(w)]
            if finite.size >= min_finite:
                sm[j] = np.median(finite)
        out[ax] = sm
    return out


def _apply_corr_month_gate(
    *, alpha: np.ndarray, cdopp: np.ndarray, corr: np.ndarray
) -> None:
    if CORR_THR_MONTH is None:
        return
    band = (LATITUDE >= LAT_LO) & (LATITUDE <= LAT_HI)
    for m in range(12):
        row = corr[m, band]
        if not np.isfinite(row).any():
            continue
        med_corr = np.nanmedian(np.abs(row))
        if not np.isfinite(med_corr) or med_corr < CORR_THR_MONTH:
            _LOG.info(
                "Zonal-mean month gate m=%d: median|corr_zm|=%.3f < %.2f, masking month",
                m + 1,
                med_corr,
                CORR_THR_MONTH,
            )
            alpha[m, :] = np.nan
            cdopp[m, :] = np.nan
            corr[m, :] = np.nan


def compute_monthly_clim_params(
    *,
    baro_directory: Path,
    years: list[int],
    label: str = "",
    lat_smooth_deg: int = LAT_SMOOTH_DEG,
) -> dict[str, np.ndarray]:
    alpha_zm_all = np.full((12, NLAT), np.nan)
    cdopp_zm_all = np.full((12, NLAT), np.nan)
    corr_zm_all = np.full((12, NLAT), np.nan)
    u0_zm_all = np.full((12, NLAT), np.nan)
    a0_all = np.full((12, NLAT, NLON), np.nan)
    alpha_loc_all = np.full((12, NLAT, NLON), np.nan)
    cdopp_loc_all = np.full((12, NLAT, NLON), np.nan)
    corr_loc_all = np.full((12, NLAT, NLON), np.nan)
    ndays_all = np.zeros(12, dtype=int)

    for m in range(1, 13):
        _LOG.info("[%s] === Month %02d ===", label, m)

        sum_a2 = np.zeros((NLAT, NLON))
        sum_ue_a = np.zeros((NLAT, NLON))
        sum_ue2 = np.zeros((NLAT, NLON))
        sum_a = np.zeros((NLAT, NLON))
        sum_ue = np.zeros((NLAT, NLON))
        sum_fl_ac = np.zeros((NLAT, NLON))
        sum_ac2 = np.zeros((NLAT, NLON))
        sum_fl = np.zeros((NLAT, NLON))
        sum_ac = np.zeros((NLAT, NLON))
        ntot = np.zeros((NLAT, NLON))

        a0_sum = np.zeros((NLAT, NLON))
        a0_cnt = np.zeros((NLAT, NLON))
        u0_sum = np.zeros(NLAT)
        u0_cnt = np.zeros(NLAT)
        n_months = 0

        for yr in years:
            data = load_month_daily(baro_directory=baro_directory, year=yr, month=m)
            if data is None:
                continue
            _LOG.info("[%s] %d-%02d (%d days)", label, yr, m, data["ndays"])

            lwa = data["lwa"]
            ue = data["ue"]
            flin = data["f_linear"]
            uref = data["uref"]

            lwa = zsmooth(x=lwa)
            ue = zsmooth(x=ue)
            flin = zsmooth(x=flin)

            if lat_smooth_deg > 1:
                lwa = lat_smooth(x=lwa, deg=lat_smooth_deg)
                ue = lat_smooth(x=ue, deg=lat_smooth_deg)
                flin = lat_smooth(x=flin, deg=lat_smooth_deg)
                uref = lat_smooth(x=uref, deg=lat_smooth_deg)

            lwa_mean = np.nanmean(lwa, axis=0)
            ue_mean = np.nanmean(ue, axis=0)
            flin_mean = np.nanmean(flin, axis=0)
            uref_mean = np.nanmean(uref, axis=0)

            a_anom = lwa - lwa_mean
            ue_anom = ue - ue_mean
            fl_anom = flin - flin_mean
            ac_anom = a_anom * COSPHI[np.newaxis, :, np.newaxis]

            fin = np.isfinite(a_anom) & np.isfinite(ue_anom) & np.isfinite(fl_anom)
            a_a = np.where(fin, a_anom, 0.0)
            ue_a = np.where(fin, ue_anom, 0.0)
            fl_a = np.where(fin, fl_anom, 0.0)
            ac_a = np.where(fin, ac_anom, 0.0)
            n = fin.astype(np.float64)

            sum_a2 += (a_a**2).sum(0)
            sum_ue_a += (ue_a * a_a).sum(0)
            sum_ue2 += (ue_a**2).sum(0)
            sum_a += a_a.sum(0)
            sum_ue += ue_a.sum(0)
            sum_fl_ac += (fl_a * ac_a).sum(0)
            sum_ac2 += (ac_a**2).sum(0)
            sum_fl += fl_a.sum(0)
            sum_ac += ac_a.sum(0)
            ntot += n.sum(0)

            a0_good = np.isfinite(lwa_mean)
            a0_sum[a0_good] += lwa_mean[a0_good]
            a0_cnt[a0_good] += 1
            good = np.isfinite(uref_mean)
            u0_sum[good] += uref_mean[good]
            u0_cnt[good] += 1
            n_months += 1

        if n_months == 0:
            _LOG.info("[%s] month %02d: no valid years", label, m)
            continue

        _LOG.info(
            "[%s] month %02d: pooled %d years, %d days/gridpoint",
            label,
            m,
            n_months,
            int(ntot.max()),
        )
        ndays_all[m - 1] = int(ntot.max())

        with np.errstate(divide="ignore", invalid="ignore"):
            a0_all[m - 1] = np.where(
                a0_cnt > 0, a0_sum / np.maximum(a0_cnt, 1), np.nan
            )
            u0_zm_all[m - 1] = np.where(
                u0_cnt > 0, u0_sum / np.maximum(u0_cnt, 1), np.nan
            )

        ns = np.maximum(ntot, 1)
        var_a = sum_a2 / ns - (sum_a / ns) ** 2
        var_ue = sum_ue2 / ns - (sum_ue / ns) ** 2
        cov_ue_a = sum_ue_a / ns - (sum_ue / ns) * (sum_a / ns)
        var_ac = sum_ac2 / ns - (sum_ac / ns) ** 2
        cov_fl_ac = sum_fl_ac / ns - (sum_fl / ns) * (sum_ac / ns)

        alpha_local = np.where(
            var_a > VAR_FLOOR, -cov_ue_a / np.maximum(var_a, VAR_FLOOR), np.nan
        )
        d = np.sqrt(np.maximum(var_a, 0) * np.maximum(var_ue, 0))
        corr_local = np.where(d > 1e-12, cov_ue_a / d, np.nan)
        cdopp_local = np.where(
            var_ac > VAR_FLOOR, cov_fl_ac / np.maximum(var_ac, VAR_FLOOR), np.nan
        )

        alpha_loc_all[m - 1] = alpha_local
        cdopp_loc_all[m - 1] = cdopp_local
        corr_loc_all[m - 1] = corr_local

        for j in range(NLAT):
            if LATITUDE[j] < LAT_LO or LATITUDE[j] > LAT_HI:
                continue
            a_row = alpha_local[j, :]
            c_row = corr_local[j, :]
            d_row = cdopp_local[j, :]
            ok_a = np.isfinite(a_row) & (a_row > 0) & (a_row < ALPHA_MAX_PHYS)
            if ok_a.sum() >= MIN_ZM_GRIDPTS:
                alpha_zm_all[m - 1, j] = np.nanmean(a_row[ok_a])
                corr_zm_all[m - 1, j] = np.nanmean(c_row[ok_a & np.isfinite(c_row)])
            ok_d = np.isfinite(d_row) & (np.abs(d_row) < CDOPP_BLOWUP)
            if ok_d.sum() >= MIN_ZM_GRIDPTS:
                cdopp_zm_all[m - 1, j] = np.nanmean(d_row[ok_d])

    bad_zm = (
        (~np.isfinite(corr_zm_all))
        | (np.abs(corr_zm_all) < CORR_THR)
        | (~np.isfinite(cdopp_zm_all))
        | (np.abs(cdopp_zm_all) > CDOPP_MAX_PHYS)
    )
    alpha_zm_all[bad_zm] = np.nan
    cdopp_zm_all[bad_zm] = np.nan
    corr_zm_all[bad_zm] = np.nan

    _apply_corr_month_gate(alpha=alpha_zm_all, cdopp=cdopp_zm_all, corr=corr_zm_all)

    alpha_zm_all = _lat_med_fill(profile=alpha_zm_all)
    cdopp_zm_all = _lat_med_fill(profile=cdopp_zm_all)
    corr_zm_all = _lat_med_fill(profile=corr_zm_all)
    lat_mask_out = (LATITUDE < LAT_LO) | (LATITUDE > LAT_HI)
    alpha_zm_all[:, lat_mask_out] = np.nan
    cdopp_zm_all[:, lat_mask_out] = np.nan
    corr_zm_all[:, lat_mask_out] = np.nan

    return {
        "alpha_zm": alpha_zm_all,
        "cdopp_zm": cdopp_zm_all,
        "corr_zm": corr_zm_all,
        "u0_zm": u0_zm_all,
        "A0": a0_all,
        "alpha_local": alpha_loc_all,
        "cdopp_local": cdopp_loc_all,
        "corr_local": corr_loc_all,
        "ndays": ndays_all,
    }


def compute_fc_monthly(
    *, params: dict[str, np.ndarray], a0: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = params["alpha_zm"]
    cdopp = params["cdopp_zm"]
    if a0 is None:
        a0 = params["A0"]
    fc = np.full_like(a0, np.nan)
    ac = np.full_like(a0, np.nan)
    c_all = np.full_like(a0, np.nan)
    for m in range(12):
        a = alpha[m, :, np.newaxis]
        cd = cdopp[m, :, np.newaxis]
        valid_zm = np.isfinite(a) & np.isfinite(cd) & (a > ALPHA_FLOOR)
        c = cd - 2 * a * a0[m]
        fc_m = COS2[:, np.newaxis] * c**2 / (4 * a)
        ac_m = c / (2 * a)
        ok = valid_zm & np.isfinite(fc_m) & (c > 0)
        fc[m] = np.where(ok, fc_m, np.nan)
        ac[m] = np.where(ok, ac_m, np.nan)
        c_all[m] = np.where(ok, c, np.nan)
    return fc, ac, c_all


def write_monthly_clim_netcdf(
    *,
    path: Path,
    params: dict[str, np.ndarray],
    fc: np.ndarray,
    ac: np.ndarray,
    c: np.ndarray,
    title: str,
    source: str = "",
    a0_note: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), "w", clobber=True) as ds:
        ds.createDimension("month", 12)
        ds.createDimension("latitude", NLAT)
        ds.createDimension("longitude", NLON)

        mv = ds.createVariable("month", "i4", ("month",))
        mv[:] = np.arange(1, 13)
        mv.long_name = "calendar month"

        lv = ds.createVariable("latitude", "f4", ("latitude",))
        lv[:] = LATITUDE
        lv.units = "degrees_north"

        gv = ds.createVariable("longitude", "f4", ("longitude",))
        gv[:] = LONGITUDE
        gv.units = "degrees_east"

        def _put(
            *,
            name: str,
            data: np.ndarray,
            dims: tuple[str, ...],
            long_name: str,
            units: str,
        ) -> None:
            v = ds.createVariable(
                name, "f4", dims, zlib=True, complevel=4, fill_value=np.nan
            )
            v[:] = np.asarray(data, dtype="f4")
            v.long_name = long_name
            v.units = units

        _put(
            name="alpha_zm",
            data=params["alpha_zm"],
            dims=("month", "latitude"),
            long_name="LWA diffusivity (zonal-mean, per month)",
            units="day^-1 or 1/(time unit)",
        )
        _put(
            name="cdopp_zm",
            data=params["cdopp_zm"],
            dims=("month", "latitude"),
            long_name="Doppler-shifted group velocity u0+cgx (zonal-mean, per month)",
            units="m s-1",
        )
        _put(
            name="corr_zm",
            data=params["corr_zm"],
            dims=("month", "latitude"),
            long_name="Zonal-mean correlation corr(ue, A)",
            units="1",
        )
        if "u0_zm" in params:
            _put(
                name="u0_zm",
                data=params["u0_zm"],
                dims=("month", "latitude"),
                long_name="Pooled monthly-mean zonal-mean reference wind u0 (uref)",
                units="m s-1",
            )
        _put(
            name="A0",
            data=params["A0"],
            dims=("month", "latitude", "longitude"),
            long_name="Stationary LWA proxy (multi-year monthly-mean LWA)",
            units="m s-1",
        )
        _put(
            name="Fc",
            data=fc,
            dims=("month", "latitude", "longitude"),
            long_name="Jet-stream carrying capacity",
            units="m^2 s^-2",
        )
        _put(
            name="Ac",
            data=ac,
            dims=("month", "latitude", "longitude"),
            long_name="LWA threshold (C / (2 alpha))",
            units="m s-1",
        )
        _put(
            name="C",
            data=c,
            dims=("month", "latitude", "longitude"),
            long_name="C = u0+cgx - 2*alpha*A0",
            units="m s-1",
        )

        ndv = ds.createVariable("ndays", "i4", ("month",))
        ndv[:] = params["ndays"]
        ndv.long_name = "pooled days/gridpoint per month used in regression"

        ds.title = title
        ds.source = source
        ds.method = (
            "Fc(lambda,phi,m) = cos^2(phi) * [cdopp_zm(m,phi) "
            "- 2*alpha_zm(m,phi)*A0(m,lambda,phi)]^2 / (4*alpha_zm(m,phi)); "
            "alpha, c_dopp from monthly-pooled gridpoint regressions of "
            "ue ~ -alpha*A and (F1+F3) ~ cdopp*(A cos phi), then zonal-mean "
            "filters |corr_zm|>=" + f"{CORR_THR}, |cdopp_zm|<={CDOPP_MAX_PHYS:g} m/s "
            "(same pipeline as ERA5/MERRA-2 monthly climatology)."
        )
        ds.a0_note = a0_note
        ds.references = "Barpanda & Nakamura (2025, JAS)"


def main(
    baro_directory: Annotated[
        Path,
        typer.Option(
            help="BARO_N-layout directory with monthly/yearly NetCDFs of lwa, u, uref, ua1, ep1"
        ),
    ],
    year_start: Annotated[int, typer.Option(help="First year to pool (inclusive)")],
    year_end: Annotated[int, typer.Option(help="Last year to pool (inclusive)")],
    dataset_label: Annotated[
        str, typer.Option(help="Label used in output file names, e.g. era5")
    ],
    output_directory: Annotated[Path, typer.Option(help="Directory for NPZ and NetCDF outputs")],
    stationary_a0_file: Annotated[
        Optional[Path],
        typer.Option(
            help="PV-based stationary LWA NetCDF (variable lwa, shape 12 x lat x lon); "
            "when omitted, the multi-year monthly-mean LWA is used as A0"
        ),
    ] = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    years = list(range(year_start, year_end + 1))
    params = compute_monthly_clim_params(
        baro_directory=baro_directory, years=years, label=dataset_label
    )
    a0_native = params["A0"].copy()
    a0_note = "A0 = multi-year monthly-mean daily LWA from regression run."
    if stationary_a0_file is not None:
        params["A0"] = load_stationary_a0(path=stationary_a0_file)
        a0_note = f"A0 = PV-based stationary LWA ({stationary_a0_file.name})."
        print("Using PV-based stationary LWA for A0 in Fc")
    fc, ac, c = compute_fc_monthly(params=params)

    output_directory.mkdir(parents=True, exist_ok=True)
    npz_path = output_directory / f"{dataset_label}_monthly_clim_params.npz"
    npz_payload: dict[str, Any] = dict(params)
    npz_payload.update(
        A0_native=a0_native,
        Fc=fc,
        Ac=ac,
        C=c,
        latitude=LATITUDE,
        longitude=LONGITUDE,
    )
    np.savez(npz_path, **npz_payload)
    print(f"Saved {npz_path}")

    nc_path = output_directory / f"{dataset_label}_carrying_capacity.nc"
    write_monthly_clim_netcdf(
        path=nc_path,
        params=params,
        fc=fc,
        ac=ac,
        c=c,
        title=f"{dataset_label} monthly-climatological jet-stream carrying capacity",
        source=f"Computed from {baro_directory}",
        a0_note=a0_note
        + f"  Fc uses pooled alpha_zm/cdopp_zm (|corr|>={CORR_THR}, "
        + f"|cdopp|<={CDOPP_MAX_PHYS:g} m/s) and C=c_dopp-2*alpha*A0>0.",
    )
    print(f"Saved {nc_path}")


if __name__ == "__main__":
    typer.run(main)
