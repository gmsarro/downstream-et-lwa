"""Time-resolved (rolling-window) jet carrying capacity, BN25 single-year style."""

from __future__ import annotations

import calendar
import logging
from pathlib import Path
from typing import Optional

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

LOWPASS_DAYS = 5
ZSMOOTH_DEG = 15
REG_WINDOW = 61
MIN_FRAC = 0.70
ALPHA_FLOOR = 1e-6
VAR_FLOOR = 1e-4
LAT_LO = 20
LAT_HI = 80
CORR_THRESHOLD = 0.5
CDOPP_BLOWUP = 200.0


def _ndays(*, year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def load_year(*, baro_directory: Path, year: int, suffix: str, var: str) -> np.ndarray:
    with netCDF4.Dataset(str(baro_directory / f"{year}_{suffix}.nc"), "r") as ds:
        raw = np.array(ds.variables[var][:])
    is1d = raw.ndim == 3
    chunks = []
    for m in range(12):
        nd = calendar.monthrange(year, m + 1)[1]
        nt = nd * 4
        chunks.append(raw[:nt, m, :] if is1d else raw[:nt, m, :, :])
    return np.concatenate(chunks, axis=0)


def daily_mean(*, x6h: np.ndarray, year: int) -> np.ndarray:
    nd = _ndays(year=year)
    sp = x6h.shape[1:]
    d = np.empty((nd, *sp), np.float64)
    i = 0
    for j in range(nd):
        d[j] = x6h[i : i + 4].mean(0)
        i += 4
    return d


def _rmean(*, x: np.ndarray, hw: int) -> np.ndarray:
    nt = x.shape[0]
    pad = np.zeros((1, *x.shape[1:]), np.float64)
    cs = np.concatenate([pad, np.nancumsum(x, axis=0)])
    cn = np.concatenate([pad, np.nancumsum(np.isfinite(x).astype(np.float64), axis=0)])
    out = np.full_like(x, np.nan, np.float64)
    for t in range(nt):
        lo, hi = max(0, t - hw), min(nt, t + hw + 1)
        s = cs[hi] - cs[lo]
        c = cn[hi] - cn[lo]
        out[t] = np.where(c >= MIN_FRAC * (hi - lo), s / np.maximum(c, 1), np.nan)
    return out


def zsmooth(*, x: np.ndarray, deg: float = ZSMOOTH_DEG) -> np.ndarray:
    if deg <= 1:
        return x
    return scipy.ndimage.uniform_filter1d(x, int(round(deg)), axis=-1, mode="wrap")


def _rolling_reg(
    *, y: np.ndarray, x: np.ndarray, hw: int
) -> tuple[np.ndarray, np.ndarray]:
    nt = x.shape[0]
    beta = np.full_like(x, np.nan, np.float64)
    corr = np.full_like(x, np.nan, np.float64)
    pad = np.zeros((1, *x.shape[1:]), np.float64)
    fin = np.isfinite(x) & np.isfinite(y)
    xf = np.where(fin, x, 0.0)
    yf = np.where(fin, y, 0.0)
    msk = fin.astype(np.float64)
    csx = np.concatenate([pad, np.cumsum(xf, 0)])
    csy = np.concatenate([pad, np.cumsum(yf, 0)])
    csxx = np.concatenate([pad, np.cumsum(xf * xf, 0)])
    csyy = np.concatenate([pad, np.cumsum(yf * yf, 0)])
    csxy = np.concatenate([pad, np.cumsum(xf * yf, 0)])
    csn = np.concatenate([pad, np.cumsum(msk, 0)])
    for t in range(nt):
        lo, hi = max(0, t - hw), min(nt, t + hw + 1)
        n = csn[hi] - csn[lo]
        ns = np.maximum(n, 1)
        ok = n >= MIN_FRAC * (hi - lo)
        sx = (csx[hi] - csx[lo]) / ns
        sy = (csy[hi] - csy[lo]) / ns
        cov = (csxy[hi] - csxy[lo]) / ns - sx * sy
        vx = (csxx[hi] - csxx[lo]) / ns - sx**2
        vy = (csyy[hi] - csyy[lo]) / ns - sy**2
        vok = vx > VAR_FLOOR
        beta[t] = np.where(ok & vok, cov / np.maximum(vx, VAR_FLOOR), np.nan)
        d = np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
        corr[t] = np.where(ok & vok & (d > 1e-12), cov / d, np.nan)
    return beta, corr


def _djf_indices(*, year: int) -> np.ndarray:
    nd = _ndays(year=year)
    leap = calendar.isleap(year)
    jan = list(range(0, 31))
    feb = list(range(31, 60 if leap else 59))
    dec = list(range(336 if leap else 335, nd))
    return np.array(jan + feb + dec)


def load_stationary_a0_djf(*, path: Path) -> np.ndarray:
    with netCDF4.Dataset(str(path), "r") as ds:
        a0_monthly = np.array(ds.variables["lwa"][:])
    return a0_monthly[[0, 1, 11]].mean(0)


def compute_carrying_capacity(
    *, baro_directory: Path, stationary_a0_file: Path, year: int
) -> tuple[dict[str, np.ndarray], int, dict[str, np.ndarray]]:
    _LOG.info("[%d] Loading ...", year)
    a_6h = load_year(baro_directory=baro_directory, year=year, suffix="LWAb_N", var="lwa")
    u_6h = load_year(baro_directory=baro_directory, year=year, suffix="Ub_N", var="u")
    ur_6h = load_year(baro_directory=baro_directory, year=year, suffix="Urefb_N", var="uref")
    f1_6h = load_year(baro_directory=baro_directory, year=year, suffix="ua1_N", var="ua1")
    f3_6h = load_year(baro_directory=baro_directory, year=year, suffix="ep1_N", var="ep1")

    _LOG.info("[%d] Daily means ...", year)
    a_raw = daily_mean(x6h=a_6h, year=year)
    u_raw = daily_mean(x6h=u_6h, year=year)
    ur_raw = daily_mean(x6h=ur_6h, year=year)
    f1_raw = daily_mean(x6h=f1_6h, year=year)
    f3_raw = daily_mean(x6h=f3_6h, year=year)
    del a_6h, u_6h, ur_6h, f1_6h, f3_6h
    nt = a_raw.shape[0]
    ue_raw = u_raw - ur_raw[:, :, np.newaxis]
    flin_raw = f1_raw + f3_raw

    cph1 = COSPHI[np.newaxis, :, np.newaxis]
    latmask = ((LATITUDE >= LAT_LO) & (LATITUDE <= LAT_HI))[np.newaxis, :, np.newaxis]

    _LOG.info("[%d] Loading A0 from stationary LWA file ...", year)
    a0_static = load_stationary_a0_djf(path=stationary_a0_file)
    a0_time = np.broadcast_to(a0_static[np.newaxis, :, :], a_raw.shape).copy()

    _LOG.info("[%d] Filtering ...", year)
    a_filt = zsmooth(x=_rmean(x=a_raw, hw=LOWPASS_DAYS // 2))
    ue_filt = zsmooth(x=_rmean(x=ue_raw, hw=LOWPASS_DAYS // 2))
    fl_filt = zsmooth(x=_rmean(x=flin_raw, hw=LOWPASS_DAYS // 2))

    a_anom = a_filt - _rmean(x=a_filt, hw=REG_WINDOW // 2)
    ue_anom = ue_filt - _rmean(x=ue_filt, hw=REG_WINDOW // 2)
    fl_anom = fl_filt - _rmean(x=fl_filt, hw=REG_WINDOW // 2)
    ac_anom = a_anom * cph1

    rhw = REG_WINDOW // 2
    _LOG.info("[%d] alpha and c_dopp ...", year)
    neg_alpha, corr_ue = _rolling_reg(y=ue_anom, x=a_anom, hw=rhw)
    alpha = -neg_alpha
    aok = (alpha > ALPHA_FLOOR) & np.isfinite(alpha) & latmask
    aok &= np.isfinite(corr_ue) & (corr_ue < -CORR_THRESHOLD)
    alpha = np.where(aok, alpha, np.nan)
    alpha_corr = np.where(np.isfinite(alpha), corr_ue, np.nan)

    c_dopp, _ = _rolling_reg(y=fl_anom, x=ac_anom, hw=rhw)
    c_dopp = np.where(
        np.isfinite(c_dopp) & latmask & (np.abs(c_dopp) < CDOPP_BLOWUP), c_dopp, np.nan
    )

    _LOG.info("[%d] Fc, Ac ...", year)
    c = c_dopp - 2 * alpha * a0_time
    fc = cph1**2 * c**2 / (4 * alpha)
    ac = c / (2 * alpha)

    valid = np.isfinite(fc) & (alpha > ALPHA_FLOOR) & np.isfinite(c_dopp) & latmask
    fc = np.where(valid, fc, np.nan)
    ac = np.where(valid, ac, np.nan)
    ac_abs = np.where(
        valid, np.sqrt(np.maximum(fc / np.maximum(alpha, ALPHA_FLOOR), 0)), np.nan
    )

    nv = np.sum(valid)
    _LOG.info("Valid %d/%d (%.1f%%)", nv, valid.size, 100 * nv / valid.size)

    results = {}
    for nm, arr in [
        ("lwa_va", a_raw),
        ("lwa_stationary", a0_time),
        ("alpha_lwa", alpha),
        ("alpha_corr", alpha_corr),
        ("u0_plus_cgx", c_dopp),
        ("carrying_capacity", fc),
        ("lwa_threshold", ac),
        ("lwa_threshold_abs", ac_abs),
        ("valid_mask", valid),
    ]:
        results[nm] = arr.astype(np.float32) if arr.dtype == np.float64 else arr

    return (
        results,
        nt,
        {
            "A_raw": a_raw,
            "ue_raw": ue_raw,
            "F_lin_raw": flin_raw,
            "A0_static": a0_static,
        },
    )


def _gridpoint_regression_then_zmean(
    *, a_a: np.ndarray, ue_a: np.ndarray, fl_a: np.ndarray, ac_a: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    va = np.nanvar(a_a, axis=0)
    cov_ue_a = np.nanmean(ue_a * a_a, axis=0) - np.nanmean(ue_a, axis=0) * np.nanmean(
        a_a, axis=0
    )
    alpha_local = -cov_ue_a / np.maximum(va, VAR_FLOOR)
    alpha_local = np.where(va > VAR_FLOOR, alpha_local, np.nan)
    vu = np.nanvar(ue_a, axis=0)
    d = np.sqrt(np.maximum(va, 0) * np.maximum(vu, 0))
    corr_local = np.where(d > 1e-12, cov_ue_a / d, np.nan)
    vac = np.nanvar(ac_a, axis=0)
    cov_fl_ac = np.nanmean(fl_a * ac_a, axis=0) - np.nanmean(fl_a, axis=0) * np.nanmean(
        ac_a, axis=0
    )
    cdopp_local = cov_fl_ac / np.maximum(vac, VAR_FLOOR)
    cdopp_local = np.where(vac > VAR_FLOOR, cdopp_local, np.nan)

    alpha_zm = np.full(NLAT, np.nan)
    cdopp_zm = np.full(NLAT, np.nan)
    corr_zm = np.full(NLAT, np.nan)
    for j in range(NLAT):
        if LATITUDE[j] < LAT_LO or LATITUDE[j] > LAT_HI:
            continue
        a_row = alpha_local[j, :]
        c_row = corr_local[j, :]
        d_row = cdopp_local[j, :]
        ok = np.isfinite(a_row) & (a_row > 0) & np.isfinite(c_row)
        if ok.sum() < 10:
            continue
        alpha_zm[j] = np.nanmean(a_row[ok])
        corr_zm[j] = np.nanmean(c_row[ok])
        ok2 = np.isfinite(d_row)
        if ok2.sum() < 10:
            continue
        cdopp_zm[j] = np.nanmean(d_row[ok2])
    return alpha_zm, cdopp_zm, corr_zm


def compute_djf_seasonal(
    *,
    year: int,
    raw: dict[str, np.ndarray],
    multi_year_params: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, np.ndarray]:
    a0_static = raw["A0_static"]

    if multi_year_params is not None:
        alpha_zm = multi_year_params["alpha_zm"]
        cdopp_zm = multi_year_params["cdopp_zm"]
        corr_zm = multi_year_params["corr_zm"]
    else:
        a = raw["A_raw"]
        ue = raw["ue_raw"]
        fl = raw["F_lin_raw"]
        djf = _djf_indices(year=year)
        a_s = zsmooth(x=a[djf])
        ue_s = zsmooth(x=ue[djf])
        fl_s = zsmooth(x=fl[djf])
        a_a = a_s - a_s.mean(0, keepdims=True)
        ue_a = ue_s - ue_s.mean(0, keepdims=True)
        fl_a = fl_s - fl_s.mean(0, keepdims=True)
        ac_a = a_a * COSPHI[np.newaxis, :, np.newaxis]
        alpha_zm, cdopp_zm, corr_zm = _gridpoint_regression_then_zmean(
            a_a=a_a, ue_a=ue_a, fl_a=fl_a, ac_a=ac_a
        )

    a_pos = np.where(
        (alpha_zm > ALPHA_FLOOR) & (np.abs(corr_zm) >= CORR_THRESHOLD), alpha_zm, np.nan
    )

    a0_djf = a0_static

    cph = COSPHI[:, np.newaxis]
    c_djf = cdopp_zm[:, np.newaxis] - 2 * a_pos[:, np.newaxis] * a0_djf
    fc_djf = cph**2 * c_djf**2 / (4 * a_pos[:, np.newaxis])
    ac_djf = c_djf / (2 * a_pos[:, np.newaxis])

    ok = np.isfinite(fc_djf) & (a_pos[:, np.newaxis] > ALPHA_FLOOR)
    fc_djf = np.where(ok, fc_djf, np.nan)
    ac_djf = np.where(ok, ac_djf, np.nan)

    result = {
        k: v.astype(np.float32)
        for k, v in [
            ("alpha_zm", alpha_zm),
            ("cdopp_zm", cdopp_zm),
            ("corr_zm", corr_zm),
            ("A0_djf", a0_djf),
            ("Fc_djf", fc_djf),
            ("Ac_djf", ac_djf),
            ("C_djf", c_djf),
        ]
    }

    if multi_year_params is not None:
        for key in ("corr_local", "alpha_local", "neg_cov_ue_A"):
            if key in multi_year_params:
                result[key] = multi_year_params[key].astype(np.float32)
    return result


def _load_djf_months(
    *, baro_directory: Path, year: int, suffix: str, var: str
) -> np.ndarray:
    with netCDF4.Dataset(str(baro_directory / f"{year}_{suffix}.nc"), "r") as ds:
        raw = ds.variables[var]
        is1d = raw.ndim == 3
        chunks = []
        for m_idx in (0, 1, 11):
            nd = calendar.monthrange(year, m_idx + 1)[1]
            nt = nd * 4
            block = np.array(raw[:nt, m_idx, :] if is1d else raw[:nt, m_idx, :, :])
            daily = block.reshape(nd, 4, *block.shape[1:]).mean(1)
            chunks.append(daily)
    return np.concatenate(chunks, axis=0)


def compute_multi_year_djf_params(
    *, baro_directory: Path, stationary_a0_file: Path, years: list[int]
) -> dict[str, np.ndarray]:
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

    for yr in years:
        _LOG.info("Loading DJF %d ...", yr)
        a = _load_djf_months(baro_directory=baro_directory, year=yr, suffix="LWAb_N", var="lwa")
        u = _load_djf_months(baro_directory=baro_directory, year=yr, suffix="Ub_N", var="u")
        ur = _load_djf_months(baro_directory=baro_directory, year=yr, suffix="Urefb_N", var="uref")
        f1 = _load_djf_months(baro_directory=baro_directory, year=yr, suffix="ua1_N", var="ua1")
        f3 = _load_djf_months(baro_directory=baro_directory, year=yr, suffix="ep1_N", var="ep1")
        ue = u - ur[:, :, np.newaxis]
        fl = f1 + f3
        del u, ur, f1, f3
        a = zsmooth(x=a)
        ue = zsmooth(x=ue)
        fl = zsmooth(x=fl)
        a_a = a - a.mean(0, keepdims=True)
        ue_a = ue - ue.mean(0, keepdims=True)
        fl_a = fl - fl.mean(0, keepdims=True)
        ac_a = a_a * COSPHI[np.newaxis, :, np.newaxis]
        fin = np.isfinite(a_a) & np.isfinite(ue_a) & np.isfinite(fl_a)
        a_a = np.where(fin, a_a, 0.0)
        ue_a = np.where(fin, ue_a, 0.0)
        fl_a = np.where(fin, fl_a, 0.0)
        ac_a = np.where(fin, ac_a, 0.0)
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
        del a, ue, fl, a_a, ue_a, fl_a, ac_a

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

    alpha_zm = np.full(NLAT, np.nan)
    cdopp_zm = np.full(NLAT, np.nan)
    corr_zm = np.full(NLAT, np.nan)
    for j in range(NLAT):
        if LATITUDE[j] < LAT_LO or LATITUDE[j] > LAT_HI:
            continue
        a_row = alpha_local[j, :]
        c_row = corr_local[j, :]
        d_row = cdopp_local[j, :]
        ok = np.isfinite(a_row) & (a_row > 0) & np.isfinite(c_row)
        if ok.sum() < 10:
            continue
        alpha_zm[j] = np.nanmean(a_row[ok])
        corr_zm[j] = np.nanmean(c_row[ok])
        ok2 = np.isfinite(d_row)
        if ok2.sum() < 10:
            continue
        cdopp_zm[j] = np.nanmean(d_row[ok2])

    _LOG.info(
        "Multi-year DJF: %d years, %d samples/gridpoint", len(years), int(ntot.max())
    )
    with netCDF4.Dataset(str(stationary_a0_file), "r") as ds:
        a0s = np.array(ds.variables["lwa"][:])[[0, 1, 11]].mean(0)
    for j in (25, 30, 35, 40, 45, 50, 55, 60, 65, 70):
        jj = int(np.argmin(np.abs(LATITUDE - j)))
        c_test = cdopp_zm[jj] - 2 * alpha_zm[jj] * a0s[jj, :].mean()
        fc_test = COSPHI[jj] ** 2 * c_test**2 / (4 * alpha_zm[jj])
        _LOG.info(
            "%dN: a=%.3f cd=%.1f r=%.3f A0=%.1f Fc_zm=%.1f",
            j,
            alpha_zm[jj],
            cdopp_zm[jj],
            corr_zm[jj],
            a0s[jj, :].mean(),
            fc_test,
        )
    neg_cov_ue_a = -cov_ue_a
    return {
        "alpha_zm": alpha_zm,
        "cdopp_zm": cdopp_zm,
        "corr_zm": corr_zm,
        "corr_local": corr_local,
        "alpha_local": alpha_local,
        "neg_cov_ue_A": neg_cov_ue_a,
    }


def save_nc(
    *,
    res: dict[str, np.ndarray],
    nt: int,
    djf: dict[str, np.ndarray],
    year: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("latitude", NLAT)
        ds.createDimension("longitude", NLON)
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = f"days since {year}-01-01"
        tv[:] = np.arange(nt, dtype=np.float64)
        lav = ds.createVariable("latitude", "f4", ("latitude",))
        lav.units = "degrees_north"
        lav[:] = LATITUDE
        lov = ds.createVariable("longitude", "f4", ("longitude",))
        lov.units = "degrees_east"
        lov[:] = LONGITUDE
        for nm, ln, un in [
            ("lwa_va", "vertically averaged LWA", "m s-1"),
            ("lwa_stationary", "A0 PV-based stationary wave", "m s-1"),
            ("alpha_lwa", "alpha from ue~-alpha*A", "1"),
            ("alpha_corr", "corr(ue,A)", "1"),
            ("u0_plus_cgx", "c_dopp", "m s-1"),
            ("carrying_capacity", "Fc=cos^2(phi)*C^2/(4alpha)", "m2 s-2"),
            ("lwa_threshold", "Ac=C/(2alpha)", "m s-1"),
            ("lwa_threshold_abs", "sqrt(Fc/alpha)", "m s-1"),
        ]:
            v = ds.createVariable(
                nm,
                "f4",
                ("time", "latitude", "longitude"),
                zlib=True,
                complevel=4,
                fill_value=np.float32(np.nan),
            )
            v.long_name = ln
            v.units = un
            v[:] = res[nm]
        v = ds.createVariable(
            "valid_mask",
            "i1",
            ("time", "latitude", "longitude"),
            zlib=True,
            complevel=4,
            fill_value=-1,
        )
        v[:] = res["valid_mask"].astype(np.int8)
        for nm, ln, un, dims in [
            ("Fc_djf", "DJF Fc cos(phi)*C^2/(4alpha)", "m2 s-2", ("latitude", "longitude")),
            ("Ac_djf", "DJF Ac", "m s-1", ("latitude", "longitude")),
            ("C_djf", "DJF C", "m s-1", ("latitude", "longitude")),
            ("A0_djf", "DJF A0 stationary wave (PV-based)", "m s-1", ("latitude", "longitude")),
            ("alpha_zm", "DJF zonal-mean alpha", "1", ("latitude",)),
            ("cdopp_zm", "DJF zonal-mean c_dopp", "m s-1", ("latitude",)),
            ("corr_zm", "DJF zonal-mean corr(ue,A)", "1", ("latitude",)),
        ]:
            v = ds.createVariable(
                nm, "f4", dims, zlib=True, complevel=4, fill_value=np.float32(np.nan)
            )
            v.long_name = ln
            v.units = un
            v[:] = djf[nm]
        for key, ln, un in [
            ("corr_local", "gridpoint-level corr(ue,A)", "1"),
            ("alpha_local", "gridpoint-level alpha", "1"),
            ("neg_cov_ue_A", "-Cov(ue,A) at gridpoints", "m2 s-2"),
        ]:
            if key in djf:
                v = ds.createVariable(
                    key,
                    "f4",
                    ("latitude", "longitude"),
                    zlib=True,
                    complevel=4,
                    fill_value=np.float32(np.nan),
                )
                v.long_name = ln
                v.units = un
                v[:] = djf[key]
        ds.title = f"Carrying capacity {year}"
        ds.method_note = (
            "Fc=cos(phi)*C^2/(4alpha) [Fig 6b]. "
            "A0 from LWAb_stationary_N.nc (PV-based monthly-mean stationary wave). "
            "alpha, c_dopp: gridpoint regression then zonal mean."
        )
    _LOG.info("Saved -> %s", path)


def main(
    baro_directory: Annotated[
        Path,
        typer.Option(help="BARO_N directory with yearly NetCDFs YYYY_{LWAb,Ub,Urefb,ua1,ep1}_N.nc"),
    ],
    stationary_a0_file: Annotated[
        Path,
        typer.Option(help="PV-based stationary LWA NetCDF (variable lwa, 12 x lat x lon)"),
    ],
    output_directory: Annotated[
        Path, typer.Option(help="Directory for the output NetCDF")
    ],
    year: Annotated[int, typer.Option(help="Target year")] = 2014,
    multi_year: Annotated[
        Optional[list[int]],
        typer.Option(help="Years pooled for the multi-year DJF parameters (repeatable)"),
    ] = None,
    dataset_label: Annotated[
        str, typer.Option(help="Label used in the output file name")
    ] = "era5",
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    multi_yr_params = None
    if multi_year:
        print(f"Computing multi-year DJF params from {len(multi_year)} years ...", flush=True)
        multi_yr_params = compute_multi_year_djf_params(
            baro_directory=baro_directory,
            stationary_a0_file=stationary_a0_file,
            years=list(multi_year),
        )
    res, nt, raw = compute_carrying_capacity(
        baro_directory=baro_directory,
        stationary_a0_file=stationary_a0_file,
        year=year,
    )
    djf = compute_djf_seasonal(year=year, raw=raw, multi_year_params=multi_yr_params)
    out_path = output_directory / f"{dataset_label}_carrying_capacity_{year}.nc"
    save_nc(res=res, nt=nt, djf=djf, year=year, path=out_path)
    print(f"Saved {out_path}")
    print("Done.")


if __name__ == "__main__":
    typer.run(main)
