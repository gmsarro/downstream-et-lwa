"""TC-relative 2-D composite maps of LWA budget terms (QJ Fig 10 style):
composite loading, budget computation, significance tests, and map figures."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import cartopy.feature
import matplotlib
import matplotlib.axes
import matplotlib.contour
import matplotlib.lines
import netCDF4 as nc
import numpy as np
import scipy.ndimage
import scipy.stats
import typer
from typing_extensions import Annotated

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.composites.io as composites_io
import downstream_et_lwa.plotting.budget_hovmoller as budget_hovmoller
import downstream_et_lwa.plotting.qj_hovmoller as qj3

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)

BUDGET_LEVELS = budget_hovmoller.BUDGET_LEVELS

MIN_TRACK_COUNT = 15
MIN_TRACK_COUNT_SMALL = 3

_SEC_PER_DAY = 86400.0

COMPOSITE_2D_FILENAME = "composite_2d_{reference}_{basin}.nc"
COMPOSITE_2D_GROUP_FILENAME = "composite_2d_{reference}_{basin}_{group}.nc"
MC_2D_ENVELOPE_FILENAME = "mc_2d_envelope_{reference}_{basin}{tag}.nc"


def _read_composite_vars_into(*, ds: nc.Dataset, data: dict[str, Any]) -> None:
    for vname in ds.variables:
        if vname.endswith("_mean"):
            key = vname[:-5]
            data[key] = np.array(ds[vname][:])
        elif vname.endswith("_sumsq"):
            key = vname[:-6]
            data[f"{key}__sumsq"] = np.array(ds[vname][:])
        elif vname.endswith("_count_field"):
            key = vname[:-12]
            data[f"{key}__count_field"] = np.array(ds[vname][:])


def load_mc_envelope_2d(*, basin: str, reference: str,
                        composites_dir: Path,
                        out_tag: str = "") -> dict[str, Any] | None:
    tag = f"_{out_tag}" if out_tag and not out_tag.startswith("_") else out_tag
    path = Path(composites_dir) / MC_2D_ENVELOPE_FILENAME.format(
        reference=reference, basin=basin, tag=tag)
    if not path.exists():
        return None
    p5: dict[str, np.ndarray] = {}
    p95: dict[str, np.ndarray] = {}
    quantity: dict[str, str] = {}
    with nc.Dataset(path, "r") as ds:
        for vname in ds.variables:
            if not vname.startswith(f"{basin}__"):
                continue
            if vname.endswith("__p5"):
                vk = vname[len(basin) + 2:-len("__p5")]
                p5[vk] = np.array(ds[vname][:])
                quantity[vk] = getattr(ds[vname], "quantity", "absolute")
            elif vname.endswith("__p95"):
                vk = vname[len(basin) + 2:-len("__p95")]
                p95[vk] = np.array(ds[vname][:])
                quantity[vk] = getattr(ds[vname], "quantity", "absolute")
        env = {
            "p5": p5, "p95": p95, "quantity": quantity,
            "n_iter": int(getattr(ds, "n_iter", 0)),
            "t_window_h": list(getattr(ds, "t_window_h", [0, 144])),
            "baseline_window_h": list(
                getattr(ds, "baseline_window_h", [-48, -12])),
            "_path": str(path),
        }
    return env


def _mc_sig_mask(*, plotted_field_2d: np.ndarray | None,
                 env: dict[str, Any] | None, var_key: str,
                 sigma_2d: float = 0.0,
                 scale: float = 1.0) -> np.ndarray | None:
    if env is None or plotted_field_2d is None:
        return None
    if var_key not in env["p5"] or var_key not in env["p95"]:
        return None
    p5 = env["p5"][var_key] * scale
    p95 = env["p95"][var_key] * scale
    if sigma_2d and sigma_2d > 0:
        p5 = smooth(field=p5, sigma=sigma_2d)
        p95 = smooth(field=p95, sigma=sigma_2d)
    out = np.zeros(plotted_field_2d.shape, dtype=bool)
    finite = (np.isfinite(plotted_field_2d) & np.isfinite(p5)
              & np.isfinite(p95))
    out[finite] = ((plotted_field_2d[finite] < p5[finite])
                   | (plotted_field_2d[finite] > p95[finite]))
    return out


def load_composite(*, basin: str, reference: str,
                   composites_dir: Path,
                   group: str | None = None,
                   supp_dir: Path | None = None) -> dict[str, Any] | None:
    if group:
        fname = COMPOSITE_2D_GROUP_FILENAME.format(
            reference=reference, basin=basin, group=group)
    else:
        fname = COMPOSITE_2D_FILENAME.format(reference=reference, basin=basin)
    path = Path(composites_dir) / fname
    if not path.exists():
        _LOG.error("  ERROR: %s not found", path)
        return None

    data: dict[str, Any] = {}
    with nc.Dataset(path, "r") as ds:
        _read_composite_vars_into(ds=ds, data=data)

        supp_name = fname.replace(".nc", "_merra2src.nc")
        supp_path = (Path(supp_dir) / supp_name) if supp_dir is not None else None
        if supp_path is not None and supp_path.exists():
            with nc.Dataset(supp_path, "r") as sds:
                _read_composite_vars_into(ds=sds, data=data)
            data["_has_merra2_source"] = True
        else:
            data["_has_merra2_source"] = False

        data["_lat"] = np.array(ds["rel_lat"][:])
        data["_lon"] = np.array(ds["rel_lon"][:])
        data["_lag_hours"] = np.array(ds["lag_hours"][:])
        data["_n_storms"] = ds.dimensions["storm"].size

        track_thresh = MIN_TRACK_COUNT_SMALL if group else MIN_TRACK_COUNT

        if "track_rel_lat" in ds.variables:
            trl = ds["track_rel_lat"][:]
            trn = ds["track_rel_lon"][:]
            valid_per_lag = np.sum(np.isfinite(trl), axis=0)
            mean_rl = np.where(valid_per_lag >= track_thresh,
                               np.nanmean(trl, axis=0), np.nan)
            mean_rn = np.where(valid_per_lag >= track_thresh,
                               np.nanmean(trn, axis=0), np.nan)
            data["_mean_rel_lat"] = mean_rl
            data["_mean_rel_lon"] = mean_rn
            data["_track_count"] = valid_per_lag

        ref_col = "recurv_lat" if reference == "recurvature" else "et_lat"
        lon_col = "recurv_lon" if reference == "recurvature" else "et_lon"
        if ref_col in ds.variables:
            data["_mean_abs_lat"] = float(np.nanmean(ds[ref_col][:]))
            data["_mean_abs_lon"] = float(np.nanmean(ds[lon_col][:]))

    return data


def _cumulative_integrate(*, field_3d: np.ndarray, dt: float,
                          ref_idx: int) -> np.ndarray:
    nlat, nlon, nlags = field_3d.shape
    result = np.full_like(field_3d, np.nan)
    result[:, :, ref_idx] = 0.0
    for t in range(ref_idx + 1, nlags):
        prev = result[:, :, t - 1]
        r_curr = field_3d[:, :, t]
        r_prev = field_3d[:, :, t - 1]
        both = np.isfinite(prev) & np.isfinite(r_curr) & np.isfinite(r_prev)
        result[:, :, t] = np.where(both, prev + 0.5 * (r_curr + r_prev) * dt,
                                   np.nan)
    for t in range(ref_idx - 1, -1, -1):
        nxt = result[:, :, t + 1]
        r_curr = field_3d[:, :, t]
        r_nxt = field_3d[:, :, t + 1]
        both = np.isfinite(nxt) & np.isfinite(r_curr) & np.isfinite(r_nxt)
        result[:, :, t] = np.where(both, nxt - 0.5 * (r_curr + r_nxt) * dt,
                                   np.nan)
    return result


def _get_var_from_sumsq(*, mean_3d: np.ndarray, sumsq_3d: np.ndarray,
                        count_3d: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = count_3d.astype(np.float64)
    var = np.full_like(mean_3d, np.nan)
    valid = n > 2
    var[valid] = sumsq_3d[valid] / n[valid] - mean_3d[valid] ** 2
    return np.maximum(np.where(valid, var, np.nan), 0.0), n, valid


def _sig_mask_ttest(*, mean_3d: np.ndarray, sumsq_3d: np.ndarray,
                    count_3d: np.ndarray,
                    alpha: float = 0.05) -> np.ndarray:
    var, n, valid = _get_var_from_sumsq(mean_3d=mean_3d, sumsq_3d=sumsq_3d,
                                        count_3d=count_3d)
    se = np.full_like(mean_3d, np.nan)
    se[valid] = np.sqrt(var[valid] / n[valid])
    t_stat = np.full_like(mean_3d, 0.0)
    se_ok = valid & (se > 0)
    t_stat[se_ok] = np.abs(mean_3d[se_ok]) / se[se_ok]
    t_crit = np.full_like(mean_3d, np.inf)
    for nval in np.unique(n[valid]):
        nval = int(nval)
        if nval > 2:
            t_crit[count_3d == nval] = scipy.stats.t.ppf(
                1.0 - alpha / 2.0, nval - 1)
    return t_stat > t_crit


def _two_sample_diff_sig(*, mean_a: np.ndarray | None,
                         sumsq_a: np.ndarray | None,
                         count_a: np.ndarray | None,
                         mean_b: np.ndarray | None,
                         sumsq_b: np.ndarray | None,
                         count_b: np.ndarray | None,
                         alpha: float = 0.05) -> np.ndarray | None:
    if any(x is None for x in (mean_a, sumsq_a, count_a,
                               mean_b, sumsq_b, count_b)):
        return None
    assert (mean_a is not None and sumsq_a is not None and count_a is not None
            and mean_b is not None and sumsq_b is not None
            and count_b is not None)
    var_a, _, valid_a = _get_var_from_sumsq(mean_3d=mean_a, sumsq_3d=sumsq_a,
                                            count_3d=count_a)
    var_b, _, valid_b = _get_var_from_sumsq(mean_3d=mean_b, sumsq_3d=sumsq_b,
                                            count_3d=count_b)
    na = count_a.astype(np.float64)
    nb = count_b.astype(np.float64)
    se2 = np.full_like(mean_a, np.nan)
    valid = valid_a & valid_b & (na > 2) & (nb > 2)
    se2[valid] = (var_a[valid] / na[valid]) + (var_b[valid] / nb[valid])
    se = np.sqrt(np.where(se2 > 0, se2, np.nan))
    diff = mean_a - mean_b
    t_stat = np.full_like(mean_a, 0.0)
    se_ok = valid & (se > 0)
    t_stat[se_ok] = np.abs(diff[se_ok]) / se[se_ok]
    df_field = np.full_like(mean_a, 1.0)
    df_field[se_ok] = np.minimum(na[se_ok], nb[se_ok]) - 1.0
    t_crit = np.full_like(mean_a, np.inf)
    for dval in np.unique(df_field[se_ok]):
        d_int = int(dval)
        if d_int > 2:
            t_crit[df_field == dval] = scipy.stats.t.ppf(
                1.0 - alpha / 2.0, d_int)
    return t_stat > t_crit


def _sig_from_var(*, mean_field: np.ndarray, var_field: np.ndarray,
                  n_field: np.ndarray,
                  alpha: float = 0.05) -> np.ndarray:
    se = np.full_like(mean_field, np.nan)
    valid = (n_field > 2) & np.isfinite(var_field) & (var_field > 0)
    se[valid] = np.sqrt(var_field[valid] / n_field[valid])
    t_stat = np.full_like(mean_field, 0.0)
    se_ok = valid & (se > 0)
    t_stat[se_ok] = np.abs(mean_field[se_ok]) / se[se_ok]
    t_crit = np.full_like(mean_field, np.inf)
    for nval in np.unique(n_field[valid]):
        nval = int(nval)
        if nval > 2:
            mask = (n_field == nval)
            t_crit[mask] = scipy.stats.t.ppf(1.0 - alpha / 2.0, nval - 1)
    return t_stat > t_crit


def _cumulative_integrate_var(*, var_3d: np.ndarray, dt: float,
                              ref_idx: int) -> np.ndarray:
    nlat, nlon, nlags = var_3d.shape
    result = np.full_like(var_3d, np.nan)
    result[:, :, ref_idx] = 0.0
    for t in range(ref_idx + 1, nlags):
        prev = result[:, :, t - 1]
        both = np.isfinite(prev) & np.isfinite(var_3d[:, :, t])
        result[:, :, t] = np.where(both, prev + var_3d[:, :, t] * dt * dt,
                                   np.nan)
    for t in range(ref_idx - 1, -1, -1):
        nxt = result[:, :, t + 1]
        both = np.isfinite(nxt) & np.isfinite(var_3d[:, :, t])
        result[:, :, t] = np.where(both, nxt + var_3d[:, :, t] * dt * dt,
                                   np.nan)
    return result


def _tendency_lwa_per_day(*, lwa: np.ndarray,
                          lag_hours: np.ndarray) -> np.ndarray:
    coord_sec = np.asarray(lag_hours, dtype=np.float64) * 3600.0
    return np.gradient(lwa, coord_sec, axis=2) * _SEC_PER_DAY


def compute_budget(*, data: dict[str, Any],
                   strip_rate_units: bool = False) -> dict[str, Any] | None:
    lag_hours = data["_lag_hours"]
    ref_idx = int(np.argmin(np.abs(lag_hours)))
    dt = composite_config.DT_SEC

    lwa = data.get("era5_lwa")
    tI = data.get("era5_budget_termI")
    tII = data.get("era5_budget_termII")
    tIII = data.get("era5_budget_termIII")

    if lwa is None:
        return None

    if strip_rate_units:
        tendency = _tendency_lwa_per_day(lwa=lwa, lag_hours=lag_hours)
        term_I_int = ((tI * _SEC_PER_DAY) if tI is not None
                      else np.full_like(lwa, np.nan))
        term_II_int = ((tII * _SEC_PER_DAY) if tII is not None
                       else np.full_like(lwa, np.nan))
        term_III_int = ((tIII * _SEC_PER_DAY) if tIII is not None
                        else np.full_like(lwa, np.nan))
        residual = tendency - term_I_int - term_II_int - term_III_int
        lh_lwa = data.get("era5_lh_lwa")
        lh_lwa_int = ((lh_lwa * _SEC_PER_DAY) if lh_lwa is not None
                      else np.full_like(lwa, np.nan))
        nonqg_lwa = data.get("era5_nonqg_lwa")
        nonqg_lwa_int = ((nonqg_lwa * _SEC_PER_DAY) if nonqg_lwa is not None
                         else np.full_like(lwa, np.nan))
        mst = data.get("merra2_heat_fortran_DTDTMST")
        rad = data.get("merra2_heat_fortran_DTDTRAD")
        ana = data.get("merra2_heat_fortran_DTDTANA")
        tot = data.get("merra2_heat_fortran_DTDTTOT")
        mst_int = (mst * _SEC_PER_DAY) if mst is not None else None
        rad_int = (rad * _SEC_PER_DAY) if rad is not None else None
        ana_int = (ana * _SEC_PER_DAY) if ana is not None else None
        tot_int = (tot * _SEC_PER_DAY) if tot is not None else None
    else:
        tendency = lwa - lwa[:, :, ref_idx:ref_idx + 1]

        term_I_int = (_cumulative_integrate(field_3d=tI, dt=dt, ref_idx=ref_idx)
                      if tI is not None else np.full_like(lwa, np.nan))
        term_II_int = (_cumulative_integrate(field_3d=tII, dt=dt,
                                             ref_idx=ref_idx)
                       if tII is not None else np.full_like(lwa, np.nan))
        term_III_int = (_cumulative_integrate(field_3d=tIII, dt=dt,
                                              ref_idx=ref_idx)
                        if tIII is not None else np.full_like(lwa, np.nan))

        residual = tendency - term_I_int - term_II_int - term_III_int

        lh_lwa = data.get("era5_lh_lwa")
        lh_lwa_int = (_cumulative_integrate(field_3d=lh_lwa, dt=dt,
                                            ref_idx=ref_idx)
                      if lh_lwa is not None else np.full_like(lwa, np.nan))
        nonqg_lwa = data.get("era5_nonqg_lwa")
        nonqg_lwa_int = (_cumulative_integrate(field_3d=nonqg_lwa, dt=dt,
                                               ref_idx=ref_idx)
                         if nonqg_lwa is not None
                         else np.full_like(lwa, np.nan))

        mst = data.get("merra2_heat_fortran_DTDTMST")
        rad = data.get("merra2_heat_fortran_DTDTRAD")
        ana = data.get("merra2_heat_fortran_DTDTANA")
        tot = data.get("merra2_heat_fortran_DTDTTOT")
        mst_int = (_cumulative_integrate(field_3d=mst, dt=dt, ref_idx=ref_idx)
                   if mst is not None else None)
        rad_int = (_cumulative_integrate(field_3d=rad, dt=dt, ref_idx=ref_idx)
                   if rad is not None else None)
        ana_int = (_cumulative_integrate(field_3d=ana, dt=dt, ref_idx=ref_idx)
                   if ana is not None else None)
        tot_int = (_cumulative_integrate(field_3d=tot, dt=dt, ref_idx=ref_idx)
                   if tot is not None else None)

    result: dict[str, Any] = {
        "lwa": lwa,
        "tendency": tendency,
        "termI": term_I_int,
        "termII": term_II_int,
        "termIII": term_III_int,
        "residual": residual,
        "lh_lwa": lh_lwa_int,
        "nonqg_lwa": nonqg_lwa_int,
        "merra2_mst": mst_int,
        "merra2_rad": rad_int,
        "merra2_ana": ana_int,
        "merra2_tot": tot_int,
    }

    for nc_key, bud_key in [
        ("era5_budget_termI", "termI"),
        ("era5_budget_termII", "termII"),
        ("era5_budget_termIII", "termIII"),
    ]:
        sumsq = data.get(f"{nc_key}__sumsq")
        cnt = data.get(f"{nc_key}__count_field")
        mean = data.get(nc_key)
        if sumsq is not None and cnt is not None and mean is not None:
            result[f"{bud_key}_sig"] = _sig_mask_ttest(
                mean_3d=mean, sumsq_3d=sumsq, count_3d=cnt)

    lh_sumsq = data.get("era5_lh_lwa__sumsq")
    lh_cnt = data.get("era5_lh_lwa__count_field")
    lh_mean = data.get("era5_lh_lwa")
    if lh_sumsq is not None and lh_cnt is not None and lh_mean is not None:
        result["lh_lwa_sig"] = _sig_mask_ttest(
            mean_3d=lh_mean, sumsq_3d=lh_sumsq, count_3d=lh_cnt)

    nonqg_sumsq = data.get("era5_nonqg_lwa__sumsq")
    nonqg_cnt = data.get("era5_nonqg_lwa__count_field")
    nonqg_mean = data.get("era5_nonqg_lwa")
    if (nonqg_sumsq is not None and nonqg_cnt is not None
            and nonqg_mean is not None):
        result["nonqg_lwa_sig"] = _sig_mask_ttest(
            mean_3d=nonqg_mean, sumsq_3d=nonqg_sumsq, count_3d=nonqg_cnt)

    for nc_key, bud_key in [
        ("merra2_heat_fortran_DTDTMST", "merra2_mst_sig"),
        ("merra2_heat_fortran_DTDTRAD", "merra2_rad_sig"),
        ("merra2_heat_fortran_DTDTANA", "merra2_ana_sig"),
        ("merra2_heat_fortran_DTDTTOT", "merra2_tot_sig"),
    ]:
        sumsq = data.get(f"{nc_key}__sumsq")
        cnt = data.get(f"{nc_key}__count_field")
        mean = data.get(nc_key)
        if sumsq is not None and cnt is not None and mean is not None:
            result[bud_key] = _sig_mask_ttest(
                mean_3d=mean, sumsq_3d=sumsq, count_3d=cnt)

    result["awb_sig"] = None
    result["cwb_sig"] = None
    result["precip_sig"] = None
    result["fc_sig"] = None

    result["lwa_sig"] = None
    lwa_sumsq = data.get("era5_lwa__sumsq")
    lwa_cnt = data.get("era5_lwa__count_field")

    if lwa_sumsq is not None and lwa_cnt is not None:
        lwa_var, _, _ = _get_var_from_sumsq(mean_3d=lwa, sumsq_3d=lwa_sumsq,
                                            count_3d=lwa_cnt)
        result["lwa_var"] = lwa_var
        result["lwa_count"] = lwa_cnt.astype(np.float64)
    else:
        lwa_var = None
        result["lwa_var"] = None
        result["lwa_count"] = None

    if not strip_rate_units:
        if lwa_var is not None and lwa_cnt is not None:
            var_at_ref = lwa_var[:, :, ref_idx:ref_idx + 1]
            tend_var = lwa_var + var_at_ref
            tend_n = np.minimum(
                lwa_cnt, lwa_cnt[:, :, ref_idx:ref_idx + 1]).astype(np.float64)
            result["tendency_sig"] = _sig_from_var(
                mean_field=tendency, var_field=tend_var, n_field=tend_n)
            result["tendency_var"] = tend_var
            result["tendency_count"] = tend_n

        if lwa_var is not None and lwa_cnt is not None:
            res_var = tend_var.copy()
            res_n = lwa_cnt.astype(np.float64).copy()
            for nc_key in ["era5_budget_termI", "era5_budget_termII",
                           "era5_budget_termIII"]:
                sq = data.get(f"{nc_key}__sumsq")
                ct = data.get(f"{nc_key}__count_field")
                mn = data.get(nc_key)
                if sq is not None and ct is not None and mn is not None:
                    term_var, _, _ = _get_var_from_sumsq(
                        mean_3d=mn, sumsq_3d=sq, count_3d=ct)
                    int_var = _cumulative_integrate_var(
                        var_3d=term_var, dt=dt, ref_idx=ref_idx)
                    res_var = res_var + np.where(np.isfinite(int_var),
                                                 int_var, 0.0)
                    res_n = np.minimum(res_n, ct.astype(np.float64))
            result["residual_sig"] = _sig_from_var(
                mean_field=residual, var_field=res_var, n_field=res_n)
            result["residual_var"] = res_var
            result["residual_count"] = res_n
    else:
        rate_prefix = None
        for pref in ("era5", "mpas"):
            if data.get(f"{pref}_dadt") is not None:
                rate_prefix = pref
                break

        if rate_prefix is not None:
            dadt_mn = data.get(f"{rate_prefix}_dadt")
            dadt_sq = data.get(f"{rate_prefix}_dadt__sumsq")
            dadt_ct = data.get(f"{rate_prefix}_dadt__count_field")
            if (dadt_mn is not None and dadt_sq is not None
                    and dadt_ct is not None):
                tend_var, _, _ = _get_var_from_sumsq(
                    mean_3d=dadt_mn, sumsq_3d=dadt_sq, count_3d=dadt_ct)
                result["tendency_var"] = tend_var * (_SEC_PER_DAY ** 2)
                result["tendency_count"] = dadt_ct.astype(np.float64)
                result["tendency_sig"] = None
            res_mn = data.get(f"{rate_prefix}_residual")
            res_sq = data.get(f"{rate_prefix}_residual__sumsq")
            res_ct = data.get(f"{rate_prefix}_residual__count_field")
            if (res_mn is not None and res_sq is not None
                    and res_ct is not None):
                res_var, _, _ = _get_var_from_sumsq(
                    mean_3d=res_mn, sumsq_3d=res_sq, count_3d=res_ct)
                result["residual_var"] = res_var * (_SEC_PER_DAY ** 2)
                result["residual_count"] = res_ct.astype(np.float64)
                result["residual_sig"] = None
        else:
            if lwa_var is not None and lwa_cnt is not None:
                coord_sec = np.asarray(lag_hours, dtype=np.float64) * 3600.0
                tend_var = np.full_like(lwa_var, np.nan)
                for k in range(lwa_var.shape[2]):
                    if 0 < k < lwa_var.shape[2] - 1:
                        h = coord_sec[k + 1] - coord_sec[k - 1]
                        tend_var[..., k] = (lwa_var[..., k + 1]
                                            + lwa_var[..., k - 1]) / (h * h)
                    elif k == 0:
                        h = coord_sec[1] - coord_sec[0]
                        tend_var[..., k] = (lwa_var[..., 1]
                                            + lwa_var[..., 0]) / (h * h)
                    else:
                        h = coord_sec[-1] - coord_sec[-2]
                        tend_var[..., k] = (lwa_var[..., -1]
                                            + lwa_var[..., -2]) / (h * h)
                tend_var = tend_var * (_SEC_PER_DAY ** 2)
                result["tendency_var"] = tend_var
                result["tendency_count"] = lwa_cnt.astype(np.float64)
                result["tendency_sig"] = None

                res_var = result.get("tendency_var")
                res_n = (lwa_cnt.astype(np.float64).copy()
                         if lwa_cnt is not None else None)
                if res_var is not None:
                    res_var = res_var.copy()
                    for nc_key in ["era5_budget_termI", "era5_budget_termII",
                                   "era5_budget_termIII"]:
                        sq = data.get(f"{nc_key}__sumsq")
                        ct = data.get(f"{nc_key}__count_field")
                        mn = data.get(nc_key)
                        if sq is not None and ct is not None and mn is not None:
                            tv, _, _ = _get_var_from_sumsq(
                                mean_3d=mn, sumsq_3d=sq, count_3d=ct)
                            res_var = res_var + tv * (_SEC_PER_DAY ** 2)
                            if res_n is not None:
                                res_n = np.minimum(res_n,
                                                   ct.astype(np.float64))
                    result["residual_var"] = res_var
                    result["residual_count"] = res_n
                    result["residual_sig"] = None

    return result


def time_average(*, field_3d: np.ndarray, lag_hours: np.ndarray,
                 t_start: float = 0, t_end: float = 168) -> np.ndarray:
    mask = (lag_hours >= t_start) & (lag_hours <= t_end)
    if not mask.any():
        return np.full(field_3d.shape[:2], np.nan)
    return np.nanmean(field_3d[:, :, mask], axis=2)


def time_average_sig(*, sig_bool_3d: np.ndarray | None,
                     lag_hours: np.ndarray,
                     t_start: float = 0, t_end: float = 168,
                     frac: float = 0.5) -> np.ndarray | None:
    mask = (lag_hours >= t_start) & (lag_hours <= t_end)
    if not mask.any() or sig_bool_3d is None:
        return None
    return np.nanmean(sig_bool_3d[:, :, mask].astype(float), axis=2) >= frac


def smooth(*, field: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    filled = np.nan_to_num(field, nan=0.0)
    mask = np.isfinite(field).astype(float)
    s_filled = scipy.ndimage.gaussian_filter(filled, sigma=sigma)
    s_mask = scipy.ndimage.gaussian_filter(mask, sigma=sigma)
    s_mask[s_mask < 0.3] = np.nan
    return s_filled / s_mask


def _sigma_3d_for_maps(*, sigma_3d: tuple[float, ...] | None
                       ) -> tuple[float, ...]:
    if sigma_3d is None:
        return composite_config.COMPOSITE_READ_VOLUME_SIGMA_3D
    return tuple(float(x) for x in sigma_3d)


def _smoothed_variance_reduction(*, sigma_2d_extra: float | None) -> float:
    if sigma_2d_extra is None or sigma_2d_extra <= 0.0:
        return 1.0
    sig_engine = tuple(
        float(x) for x in composite_config.COMPOSITE_PREMEAN_VOLUME_SIGMA_3D)
    sl, sln = sig_engine[0], sig_engine[1]
    L_lat = np.sqrt(2.0) * max(sl, 0.5)
    L_lon = np.sqrt(2.0) * max(sln, 0.5)
    fac_lat = L_lat * L_lat / (L_lat * L_lat + 2.0 * sigma_2d_extra ** 2)
    fac_lon = L_lon * L_lon / (L_lon * L_lon + 2.0 * sigma_2d_extra ** 2)
    return float(fac_lat * fac_lon)


def _smooth_volume(*, field_3d: np.ndarray,
                   sigma_3d: tuple[float, ...]) -> np.ndarray:
    out = composites_io.smooth_composite_volume(
        field_3d=field_3d, sigma_3d=sigma_3d)
    assert out is not None
    return out


def _time_mean_anomaly_sig_2d(*, mean_3d: np.ndarray | None,
                              var_3d: np.ndarray | None,
                              count_3d: np.ndarray | None,
                              lag_hours: np.ndarray,
                              t_start: float, t_end: float,
                              baseline_lag: tuple[float, float],
                              sigma_3d: tuple[float, ...] | None,
                              sigma_2d: float,
                              alpha: float = 0.05,
                              use_baseline: bool = True
                              ) -> np.ndarray | None:
    if any(x is None for x in (mean_3d, var_3d, count_3d)):
        return None
    assert mean_3d is not None and var_3d is not None and count_3d is not None
    sig3 = _sigma_3d_for_maps(sigma_3d=sigma_3d)
    m = _smooth_volume(field_3d=mean_3d, sigma_3d=sig3)
    v = _smooth_volume(field_3d=var_3d, sigma_3d=sig3)

    mask_t = (lag_hours >= t_start) & (lag_hours <= t_end)
    if not mask_t.any():
        return None
    mean_t = np.nanmean(m[:, :, mask_t], axis=2)
    var_t = np.nanmean(v[:, :, mask_t], axis=2)
    n_t = np.nanmean(count_3d[:, :, mask_t].astype(np.float64), axis=2)

    if use_baseline:
        mask_b = (lag_hours >= baseline_lag[0]) & (lag_hours <= baseline_lag[1])
        if mask_b.any():
            mean_b = np.nanmean(m[:, :, mask_b], axis=2)
            var_b = np.nanmean(v[:, :, mask_b], axis=2)
            n_b = np.nanmean(count_3d[:, :, mask_b].astype(np.float64), axis=2)
            anomaly = mean_t - mean_b
            var_anom = var_t + var_b
            n_eff = np.minimum(n_t, n_b)
        else:
            anomaly = mean_t
            var_anom = var_t
            n_eff = n_t
    else:
        anomaly = mean_t
        var_anom = var_t
        n_eff = n_t

    if sigma_2d > 0:
        anomaly = smooth(field=anomaly, sigma=sigma_2d)
        var_anom = smooth(field=var_anom, sigma=sigma_2d)
        var_anom = var_anom * _smoothed_variance_reduction(
            sigma_2d_extra=sigma_2d)

    valid = ((n_eff > 2) & np.isfinite(var_anom) & (var_anom > 0)
             & np.isfinite(anomaly))
    se = np.full_like(anomaly, np.nan)
    se[valid] = np.sqrt(var_anom[valid] / n_eff[valid])
    t_stat = np.zeros_like(anomaly)
    ok = valid & (se > 0)
    t_stat[ok] = np.abs(anomaly[ok]) / se[ok]
    t_crit = np.full_like(anomaly, np.inf)
    df_field = np.floor(n_eff).astype(np.int64) - 1
    df_field = np.clip(df_field, 1, 200)
    if valid.any():
        for d in np.unique(df_field[valid]):
            if d >= 2:
                t_crit[df_field == d] = scipy.stats.t.ppf(
                    1.0 - alpha / 2.0, int(d))
    return t_stat > t_crit


def _var_from_data(*, data: dict[str, Any], key: str
                   ) -> tuple[np.ndarray | None, np.ndarray | None]:
    mean = data.get(key)
    sumsq = data.get(f"{key}__sumsq")
    cnt = data.get(f"{key}__count_field")
    if mean is None or sumsq is None or cnt is None:
        return None, None
    var, _, _ = _get_var_from_sumsq(mean_3d=mean, sumsq_3d=sumsq, count_3d=cnt)
    return var, cnt


def _time_mean_diff_sig_2d(*, mean_a_3d: np.ndarray | None,
                           var_a_3d: np.ndarray | None,
                           count_a_3d: np.ndarray | None,
                           mean_b_3d: np.ndarray | None,
                           var_b_3d: np.ndarray | None,
                           count_b_3d: np.ndarray | None,
                           lag_hours: np.ndarray,
                           t_start: float, t_end: float,
                           sigma_3d: tuple[float, ...] | None,
                           sigma_2d: float,
                           alpha: float = 0.05,
                           baseline_lag: tuple[float, float] | None = None
                           ) -> np.ndarray | None:
    if any(x is None for x in (mean_a_3d, var_a_3d, count_a_3d,
                               mean_b_3d, var_b_3d, count_b_3d)):
        return None
    assert (mean_a_3d is not None and var_a_3d is not None
            and count_a_3d is not None and mean_b_3d is not None
            and var_b_3d is not None and count_b_3d is not None)
    sig3 = _sigma_3d_for_maps(sigma_3d=sigma_3d)
    ma = _smooth_volume(field_3d=mean_a_3d, sigma_3d=sig3)
    mb = _smooth_volume(field_3d=mean_b_3d, sigma_3d=sig3)
    va = _smooth_volume(field_3d=var_a_3d, sigma_3d=sig3)
    vb = _smooth_volume(field_3d=var_b_3d, sigma_3d=sig3)

    mask_t = (lag_hours >= t_start) & (lag_hours <= t_end)
    if not mask_t.any():
        return None
    mean_a_t = np.nanmean(ma[:, :, mask_t], axis=2)
    mean_b_t = np.nanmean(mb[:, :, mask_t], axis=2)
    var_a_t = np.nanmean(va[:, :, mask_t], axis=2)
    var_b_t = np.nanmean(vb[:, :, mask_t], axis=2)
    na = np.nanmean(count_a_3d[:, :, mask_t].astype(np.float64), axis=2)
    nb = np.nanmean(count_b_3d[:, :, mask_t].astype(np.float64), axis=2)

    if baseline_lag is not None:
        bl0, bl1 = baseline_lag
        mask_b = (lag_hours >= bl0) & (lag_hours <= bl1)
        if mask_b.any():
            mean_a_b = np.nanmean(ma[:, :, mask_b], axis=2)
            mean_b_b = np.nanmean(mb[:, :, mask_b], axis=2)
            mean_a_t = mean_a_t - mean_a_b
            mean_b_t = mean_b_t - mean_b_b
            var_a_t = 2.0 * var_a_t
            var_b_t = 2.0 * var_b_t

    diff = mean_a_t - mean_b_t
    se2 = (var_a_t / np.maximum(na, 1.0)) + (var_b_t / np.maximum(nb, 1.0))

    if sigma_2d > 0:
        diff = smooth(field=diff, sigma=sigma_2d)
        se2 = smooth(field=se2, sigma=sigma_2d)
        se2 = se2 * _smoothed_variance_reduction(sigma_2d_extra=sigma_2d)

    valid = ((na > 2) & (nb > 2) & np.isfinite(se2) & (se2 > 0)
             & np.isfinite(diff))
    se = np.full_like(diff, np.nan)
    se[valid] = np.sqrt(se2[valid])
    t_stat = np.zeros_like(diff)
    ok = valid & (se > 0)
    t_stat[ok] = np.abs(diff[ok]) / se[ok]
    df_field = np.full_like(diff, 1.0)
    df_field[ok] = np.floor(np.minimum(na[ok], nb[ok])) - 1.0
    df_field = np.clip(df_field, 1.0, 200.0)
    t_crit = np.full_like(diff, np.inf)
    if ok.any():
        for d in np.unique(df_field[ok]):
            d_int = int(d)
            if d_int >= 2:
                t_crit[df_field == d] = scipy.stats.t.ppf(
                    1.0 - alpha / 2.0, d_int)
    return t_stat > t_crit


def _diff_sig_for_derived(*, mean_a_3d: np.ndarray | None,
                          var_a_3d: np.ndarray | None,
                          count_a_3d: np.ndarray | None,
                          mean_b_3d: np.ndarray | None,
                          var_b_3d: np.ndarray | None,
                          count_b_3d: np.ndarray | None,
                          lag_hours: np.ndarray,
                          t_start: float, t_end: float,
                          sigma_3d: tuple[float, ...] | None,
                          sigma_2d: float, scale: float = 1.0,
                          baseline_lag: tuple[float, float] | None = None
                          ) -> np.ndarray | None:
    if any(x is None for x in (mean_a_3d, var_a_3d, count_a_3d,
                               mean_b_3d, var_b_3d, count_b_3d)):
        return None
    assert (mean_a_3d is not None and var_a_3d is not None
            and mean_b_3d is not None and var_b_3d is not None)
    if scale != 1.0:
        mean_a_3d = mean_a_3d * scale
        mean_b_3d = mean_b_3d * scale
        var_a_3d = var_a_3d * (scale * scale)
        var_b_3d = var_b_3d * (scale * scale)
    return _time_mean_diff_sig_2d(
        mean_a_3d=mean_a_3d, var_a_3d=var_a_3d, count_a_3d=count_a_3d,
        mean_b_3d=mean_b_3d, var_b_3d=var_b_3d, count_b_3d=count_b_3d,
        lag_hours=lag_hours,
        t_start=t_start, t_end=t_end,
        sigma_3d=sigma_3d, sigma_2d=sigma_2d,
        baseline_lag=baseline_lag,
    )


def _diff_sig_for_keys(*, data_a: dict[str, Any], data_b: dict[str, Any],
                       alias_keys: tuple[str, ...],
                       lag_hours: np.ndarray, t_start: float, t_end: float,
                       sigma_3d: tuple[float, ...] | None,
                       sigma_2d: float, scale: float = 1.0,
                       baseline_lag: tuple[float, float] | None = None
                       ) -> np.ndarray | None:
    mean_a = sumsq_a = cnt_a = None
    mean_b = sumsq_b = cnt_b = None
    for ak in alias_keys:
        if mean_a is None and ak in data_a:
            mean_a = data_a[ak]
            sumsq_a = data_a.get(f"{ak}__sumsq")
            cnt_a = data_a.get(f"{ak}__count_field")
        if mean_b is None and ak in data_b:
            mean_b = data_b[ak]
            sumsq_b = data_b.get(f"{ak}__sumsq")
            cnt_b = data_b.get(f"{ak}__count_field")
    var_a = var_b = None
    if mean_a is not None and sumsq_a is not None and cnt_a is not None:
        var_a, _, _ = _get_var_from_sumsq(mean_3d=mean_a, sumsq_3d=sumsq_a,
                                          count_3d=cnt_a)
    if mean_b is not None and sumsq_b is not None and cnt_b is not None:
        var_b, _, _ = _get_var_from_sumsq(mean_3d=mean_b, sumsq_3d=sumsq_b,
                                          count_3d=cnt_b)
    if mean_a is None or mean_b is None or var_a is None or var_b is None:
        return None
    if scale != 1.0:
        mean_a = mean_a * scale
        mean_b = mean_b * scale
        var_a = var_a * (scale * scale)
        var_b = var_b * (scale * scale)
    return _time_mean_diff_sig_2d(
        mean_a_3d=mean_a, var_a_3d=var_a, count_a_3d=cnt_a,
        mean_b_3d=mean_b, var_b_3d=var_b, count_b_3d=cnt_b,
        lag_hours=lag_hours, t_start=t_start, t_end=t_end,
        sigma_3d=sigma_3d, sigma_2d=sigma_2d,
        baseline_lag=baseline_lag,
    )


def _anomaly_sig_for_key(*, data: dict[str, Any], key: str,
                         lag_hours: np.ndarray, t_start: float, t_end: float,
                         baseline_lag: tuple[float, float],
                         sigma_3d: tuple[float, ...] | None, sigma_2d: float,
                         use_baseline: bool = True,
                         scale: float = 1.0) -> np.ndarray | None:
    mean = data.get(key)
    var, cnt = _var_from_data(data=data, key=key)
    if mean is None or var is None or cnt is None:
        return None
    if scale != 1.0:
        mean = mean * scale
        var = var * (scale * scale)
    return _time_mean_anomaly_sig_2d(
        mean_3d=mean, var_3d=var, count_3d=cnt, lag_hours=lag_hours,
        t_start=t_start, t_end=t_end,
        baseline_lag=baseline_lag, sigma_3d=sigma_3d, sigma_2d=sigma_2d,
        use_baseline=use_baseline,
    )


def _make_tavg(*, lag_hours: np.ndarray, t_start: float, t_end: float,
               sigma_3d: tuple[float, ...] | None,
               sigma_2d: float) -> Callable[..., np.ndarray]:
    sig3 = _sigma_3d_for_maps(sigma_3d=sigma_3d)

    def tavg(field_3d: np.ndarray, do_smooth: bool = True) -> np.ndarray:
        x = field_3d
        if do_smooth:
            x = _smooth_volume(field_3d=x, sigma_3d=sig3)
        raw = time_average(field_3d=x, lag_hours=lag_hours,
                           t_start=t_start, t_end=t_end)
        if do_smooth and sigma_2d > 0:
            return smooth(field=raw, sigma=sigma_2d)
        return raw

    return tavg


def _pre_event_baseline(*, field_3d: np.ndarray | None,
                        lag_hours: np.ndarray,
                        baseline_start: float = -48.0,
                        baseline_end: float = -12.0) -> np.ndarray | None:
    if field_3d is None:
        return None
    mask = (lag_hours >= baseline_start) & (lag_hours <= baseline_end)
    if not mask.any():
        return None
    base = np.nanmean(field_3d[:, :, mask], axis=2)
    return base


def _assemble_budget_map_fields(*, data: dict[str, Any],
                                budget: dict[str, Any],
                                t_start: float, t_end: float,
                                sigma_2d: float,
                                sigma_3d: tuple[float, ...] | None = None,
                                anomaly_lh_mst: bool = True,
                                baseline_lag: tuple[float, float] = (-48.0, -12.0),
                                qgpv_field_key: str = "era5_qgpv_10km",
                                mc_env: dict | None = None) -> dict[str, Any]:
    lag_hours = data["_lag_hours"]
    lat = data["_lat"]
    lon = data["_lon"]
    n = data["_n_storms"]
    tavg = _make_tavg(lag_hours=lag_hours, t_start=t_start, t_end=t_end,
                      sigma_3d=sigma_3d, sigma_2d=sigma_2d)

    def tavg_anom(field_3d: np.ndarray | None) -> np.ndarray | None:
        if field_3d is None:
            return None
        m = tavg(field_3d)
        if not anomaly_lh_mst:
            return m
        b = _pre_event_baseline(
            field_3d=field_3d, lag_hours=lag_hours,
            baseline_start=baseline_lag[0], baseline_end=baseline_lag[1])
        if b is None:
            return m
        return m - b

    lwa_f = tavg(budget["lwa"])
    tend_f = tavg_anom(budget["tendency"])
    t1_f = tavg_anom(budget["termI"])
    t2_f = tavg_anom(budget["termII"])
    t3_f = tavg_anom(budget["termIII"])
    res_f = tavg_anom(budget["residual"])
    lh_f = tavg_anom(budget.get("lh_lwa"))
    if lh_f is None:
        lh_f = np.full_like(lwa_f, np.nan)
    mst_f = tavg_anom(budget.get("merra2_mst"))
    if mst_f is None:
        mst_f = np.full_like(lwa_f, np.nan)
    nonqg_f = tavg_anom(budget.get("nonqg_lwa"))
    if nonqg_f is None:
        nonqg_f = np.full_like(lwa_f, np.nan)

    sig_kw: dict[str, Any] = dict(
        lag_hours=lag_hours,
        t_start=t_start, t_end=t_end,
        baseline_lag=baseline_lag,
        sigma_3d=sigma_3d, sigma_2d=sigma_2d,
        use_baseline=anomaly_lh_mst,
    )

    def _ttest_direct(key: str, scale: float = 1.0) -> np.ndarray | None:
        return _anomaly_sig_for_key(data=data, key=key, scale=scale, **sig_kw)

    def _ttest_derived(mean_3d: np.ndarray | None,
                       var_3d: np.ndarray | None,
                       count_3d: np.ndarray | None) -> np.ndarray | None:
        if mean_3d is None or var_3d is None or count_3d is None:
            return None
        return _time_mean_anomaly_sig_2d(mean_3d=mean_3d, var_3d=var_3d,
                                         count_3d=count_3d, **sig_kw)

    awb_key = ("mpas_rwb_awb" if data.get("mpas_rwb_awb") is not None
               and np.any(np.isfinite(data["mpas_rwb_awb"]))
               else "era5_rwb_awb")
    cwb_key = ("mpas_rwb_cwb" if data.get("mpas_rwb_cwb") is not None
               and np.any(np.isfinite(data["mpas_rwb_cwb"]))
               else "era5_rwb_cwb")
    fc_key = ("mpas_cc_Fc" if data.get("mpas_cc_Fc") is not None
              and np.any(np.isfinite(data["mpas_cc_Fc"]))
              else "era5_cc_Fc")
    pr_key = ("mpas_precip" if data.get("mpas_precip") is not None
              and np.any(np.isfinite(data["mpas_precip"]))
              else "imerg_precip")

    awb_raw = data.get("mpas_rwb_awb")
    if awb_raw is None or not np.any(np.isfinite(awb_raw)):
        awb_raw = data.get("era5_rwb_awb", np.full_like(budget["lwa"], np.nan))
    cwb_raw = data.get("mpas_rwb_cwb")
    if cwb_raw is None or not np.any(np.isfinite(cwb_raw)):
        cwb_raw = data.get("era5_rwb_cwb", np.full_like(budget["lwa"], np.nan))
    fc_raw = data.get("mpas_cc_Fc")
    if fc_raw is None or not np.any(np.isfinite(fc_raw)):
        fc_raw = data.get("era5_cc_Fc", np.full_like(budget["lwa"], np.nan))
    precip_raw = data.get("mpas_precip")
    if precip_raw is None or not np.any(np.isfinite(precip_raw)):
        precip_raw = data.get(
            "imerg_precip", np.full_like(budget["lwa"], np.nan))
    awb_f = tavg(awb_raw)
    cwb_f = tavg(cwb_raw)
    rwb_f = awb_f + cwb_f
    fc_f = tavg(fc_raw, do_smooth=False)
    precip_f = tavg(precip_raw, do_smooth=False)

    if mc_env is not None:
        sig_t1 = _mc_sig_mask(plotted_field_2d=t1_f, env=mc_env,
                              var_key="era5_budget_termI",
                              sigma_2d=sigma_2d, scale=_SEC_PER_DAY)
        sig_t2 = _mc_sig_mask(plotted_field_2d=t2_f, env=mc_env,
                              var_key="era5_budget_termII",
                              sigma_2d=sigma_2d, scale=_SEC_PER_DAY)
        sig_t3 = _mc_sig_mask(plotted_field_2d=t3_f, env=mc_env,
                              var_key="era5_budget_termIII",
                              sigma_2d=sigma_2d, scale=_SEC_PER_DAY)
        sig_lh = _mc_sig_mask(plotted_field_2d=lh_f, env=mc_env,
                              var_key="era5_lh_lwa",
                              sigma_2d=sigma_2d, scale=_SEC_PER_DAY)
        sig_nonqg = _mc_sig_mask(plotted_field_2d=nonqg_f, env=mc_env,
                                 var_key="era5_nonqg_lwa",
                                 sigma_2d=sigma_2d, scale=_SEC_PER_DAY)
        sig_mst = _mc_sig_mask(plotted_field_2d=mst_f, env=mc_env,
                               var_key="merra2_heat_fortran_DTDTMST",
                               sigma_2d=sigma_2d, scale=_SEC_PER_DAY)
        sig_tend = _mc_sig_mask(plotted_field_2d=tend_f, env=mc_env,
                                var_key="era5_dadt",
                                sigma_2d=sigma_2d, scale=_SEC_PER_DAY)
        sig_res = _mc_sig_mask(plotted_field_2d=res_f, env=mc_env,
                               var_key="era5_residual",
                               sigma_2d=sigma_2d, scale=_SEC_PER_DAY)

        sig_lwa = _mc_sig_mask(plotted_field_2d=lwa_f, env=mc_env,
                               var_key="era5_lwa", sigma_2d=sigma_2d)
        sig_awb = _mc_sig_mask(plotted_field_2d=awb_f, env=mc_env,
                               var_key=awb_key, sigma_2d=sigma_2d)
        sig_cwb = _mc_sig_mask(plotted_field_2d=cwb_f, env=mc_env,
                               var_key=cwb_key, sigma_2d=sigma_2d)
        sig_fc = _mc_sig_mask(plotted_field_2d=fc_f, env=mc_env,
                              var_key=fc_key, sigma_2d=sigma_2d)
        sig_precip = _mc_sig_mask(plotted_field_2d=precip_f, env=mc_env,
                                  var_key=pr_key, sigma_2d=sigma_2d)

        if sig_t1 is None:
            sig_t1 = _ttest_direct("era5_budget_termI", scale=_SEC_PER_DAY)
        if sig_t2 is None:
            sig_t2 = _ttest_direct("era5_budget_termII", scale=_SEC_PER_DAY)
        if sig_t3 is None:
            sig_t3 = _ttest_direct("era5_budget_termIII", scale=_SEC_PER_DAY)
        if sig_lh is None:
            sig_lh = _ttest_direct("era5_lh_lwa", scale=_SEC_PER_DAY)
        if sig_nonqg is None:
            sig_nonqg = _ttest_direct("era5_nonqg_lwa", scale=_SEC_PER_DAY)
        if sig_mst is None:
            sig_mst = _ttest_direct("merra2_heat_fortran_DTDTMST",
                                    scale=_SEC_PER_DAY)
        if sig_tend is None:
            sig_tend = _ttest_derived(budget.get("tendency"),
                                      budget.get("tendency_var"),
                                      budget.get("tendency_count"))
        if sig_res is None:
            sig_res = _ttest_derived(budget.get("residual"),
                                     budget.get("residual_var"),
                                     budget.get("residual_count"))
        if sig_lwa is None:
            sig_lwa = _ttest_direct("era5_lwa")
        if sig_awb is None:
            sig_awb = _ttest_direct(awb_key)
        if sig_cwb is None:
            sig_cwb = _ttest_direct(cwb_key)
        if sig_fc is None:
            sig_fc = _ttest_direct(fc_key)
        if sig_precip is None:
            sig_precip = _ttest_direct(pr_key)
    else:
        sig_t1 = _ttest_direct("era5_budget_termI", scale=_SEC_PER_DAY)
        sig_t2 = _ttest_direct("era5_budget_termII", scale=_SEC_PER_DAY)
        sig_t3 = _ttest_direct("era5_budget_termIII", scale=_SEC_PER_DAY)
        sig_lh = _ttest_direct("era5_lh_lwa", scale=_SEC_PER_DAY)
        sig_nonqg = _ttest_direct("era5_nonqg_lwa", scale=_SEC_PER_DAY)
        sig_lwa = _ttest_direct("era5_lwa")
        sig_mst = _ttest_direct("merra2_heat_fortran_DTDTMST",
                                scale=_SEC_PER_DAY)
        sig_tend = _ttest_derived(budget.get("tendency"),
                                  budget.get("tendency_var"),
                                  budget.get("tendency_count"))
        sig_res = _ttest_derived(budget.get("residual"),
                                 budget.get("residual_var"),
                                 budget.get("residual_count"))
        sig_awb = _ttest_direct(awb_key)
        sig_cwb = _ttest_direct(cwb_key)
        sig_fc = _ttest_direct(fc_key)
        sig_precip = _ttest_direct(pr_key)

    qgpv_raw = data.get(qgpv_field_key)
    if qgpv_raw is None or not np.any(np.isfinite(qgpv_raw)):
        qgpv_raw = np.full_like(budget["lwa"], np.nan)
    qgpv_f = tavg(qgpv_raw)

    ua1_raw = (data.get("era5_ua1") if data.get("era5_ua1") is not None
               else data.get("ua1"))
    ua2_raw = (data.get("era5_ua2") if data.get("era5_ua2") is not None
               else data.get("ua2"))
    ep1_raw = (data.get("era5_ep1") if data.get("era5_ep1") is not None
               else data.get("ep1"))
    ep2a_raw = (data.get("era5_ep2a") if data.get("era5_ep2a") is not None
                else data.get("ep2a"))
    ep3a_raw = (data.get("era5_ep3a") if data.get("era5_ep3a") is not None
                else data.get("ep3a"))

    f_lambda_avg = None
    f_phi_avg = None
    if ua1_raw is not None:
        f_lambda = ua1_raw.copy()
        if ua2_raw is not None:
            f_lambda = f_lambda + ua2_raw
        if ep1_raw is not None:
            f_lambda = f_lambda + ep1_raw
        f_lambda_avg = tavg_anom(f_lambda)
    if ep2a_raw is not None and ep3a_raw is not None:
        mean_abs_lat = data.get("_mean_abs_lat", 35.0)
        abs_lat = mean_abs_lat + lat
        cosphi = np.cos(np.deg2rad(abs_lat))
        cosphi = np.maximum(cosphi, 0.1)
        f_phi = 0.5 * (ep2a_raw + ep3a_raw) / cosphi[:, np.newaxis, np.newaxis]
        f_phi_avg = tavg_anom(f_phi)

    mean_rlat = data.get("_mean_rel_lat", np.full(len(lag_hours), np.nan))
    mean_rlon = data.get("_mean_rel_lon", np.full(len(lag_hours), np.nan))
    track_mask = (lag_hours >= t_start - 24) & (lag_hours <= t_end + 24)
    track_valid = np.isfinite(mean_rlat) & np.isfinite(mean_rlon) & track_mask
    lag0 = int(np.argmin(np.abs(lag_hours)))

    coast_segs = None
    if "_mean_abs_lat" in data:
        coast_segs = _get_coastlines_shifted(
            mean_lat=data["_mean_abs_lat"], mean_lon=data["_mean_abs_lon"])

    sig_rwb = None
    if sig_awb is not None and sig_cwb is not None:
        sig_rwb = sig_awb | sig_cwb
    elif sig_awb is not None:
        sig_rwb = sig_awb
    elif sig_cwb is not None:
        sig_rwb = sig_cwb

    return dict(
        lat=lat, lon=lon, n=n,
        lwa_f=lwa_f, tend_f=tend_f, t1_f=t1_f, t2_f=t2_f, t3_f=t3_f,
        res_f=res_f, lh_f=lh_f, mst_f=mst_f, nonqg_f=nonqg_f,
        awb_f=awb_f, cwb_f=cwb_f, rwb_f=rwb_f, fc_f=fc_f, precip_f=precip_f,
        qgpv_f=qgpv_f,
        sig_t1=sig_t1, sig_t2=sig_t2, sig_t3=sig_t3,
        sig_lwa=sig_lwa,
        sig_awb=sig_awb, sig_cwb=sig_cwb, sig_rwb=sig_rwb,
        sig_precip=sig_precip, sig_fc=sig_fc, sig_lh=sig_lh,
        sig_nonqg=sig_nonqg,
        sig_tend=sig_tend, sig_res=sig_res, sig_mst=sig_mst,
        f_lambda_avg=f_lambda_avg, f_phi_avg=f_phi_avg,
        mean_rlat=mean_rlat, mean_rlon=mean_rlon, track_valid=track_valid,
        lag0=lag0, coast_segs=coast_segs,
    )


def _fig5_draw_panel(*, ax: matplotlib.axes.Axes,
                     lon: np.ndarray, lat: np.ndarray,
                     field: np.ndarray | None, title: str,
                     levels: np.ndarray, cmap: Any,
                     coast_segs: list | None = None,
                     qgpv_f: np.ndarray | None = None,
                     sig_mask: np.ndarray | None = None,
                     flux_u: np.ndarray | None = None,
                     flux_v: np.ndarray | None = None,
                     mean_rlon: np.ndarray | None = None,
                     mean_rlat: np.ndarray | None = None,
                     track_valid: np.ndarray | None = None, lag0: int = 0,
                     ylabel: bool = False, xlabel: bool = False,
                     rel_lat_ymin: Optional[float] = None,
                     avg_field: np.ndarray | None = None,
                     avg_levels: list | None = None
                     ) -> matplotlib.contour.QuadContourSet:
    qgpv_levels = [1e-4, 1.5e-4, 2e-4]
    if coast_segs is not None:
        for rl, ra in coast_segs:
            in_box = ((rl > lon[0] - 5) & (rl < lon[-1] + 5)
                      & (ra > lat[0] - 5) & (ra < lat[-1] + 5))
            if in_box.sum() > 1:
                ax.plot(rl[in_box], ra[in_box], color="0.3",
                        lw=0.8, alpha=0.7, zorder=2)
    if avg_field is not None and avg_levels is not None:
        pos_levs = [lv for lv in avg_levels if lv > 0]
        neg_levs = [lv for lv in avg_levels if lv < 0]
        if pos_levs:
            ax.contour(lon, lat, avg_field, levels=pos_levs,
                       colors="k", linewidths=1.2, linestyles="-",
                       alpha=0.8, zorder=1)
        if neg_levs:
            ax.contour(lon, lat, avg_field, levels=neg_levs,
                       colors="k", linewidths=1.2, linestyles="--",
                       alpha=0.8, zorder=1)
    cf = ax.contourf(lon, lat, field, levels=levels,
                     cmap=cmap, extend="both")
    if sig_mask is not None:
        ax.contourf(lon, lat, sig_mask.astype(float),
                    levels=[0.5, 1.5], colors="none", hatches=[".."],
                    zorder=3)
    if qgpv_f is not None and np.any(np.isfinite(qgpv_f)):
        qgpv_smooth = scipy.ndimage.gaussian_filter(
            np.where(np.isfinite(qgpv_f), qgpv_f, 0.0), sigma=(1.5, 1.5))
        for lev in qgpv_levels:
            ax.contour(lon, lat, qgpv_smooth, levels=[lev],
                       colors="green", linewidths=0.9, linestyles="--",
                       alpha=0.85)
    if flux_u is not None:
        fv = flux_v if flux_v is not None else np.zeros_like(flux_u)
        skip = 6
        lon_q = lon[::skip]
        lat_q = lat[::skip]
        u_q = flux_u[::skip, ::skip]
        v_q = fv[::skip, ::skip]
        mag = np.sqrt(u_q**2 + v_q**2)
        finite_mag = mag[np.isfinite(mag)]
        ref_mag = (max(float(np.nanpercentile(finite_mag, 75)), 1.0)
                   if finite_mag.size else 1.0)
        ax.quiver(lon_q, lat_q, u_q, v_q,
                  scale=ref_mag * 1.0, scale_units="width",
                  width=0.010, headwidth=3.5, headlength=4.5,
                  headaxislength=4.0,
                  color="k", alpha=0.95, zorder=5,
                  minshaft=1.0, minlength=0.2,
                  pivot="middle")
    if (mean_rlon is not None and mean_rlat is not None
            and track_valid is not None):
        ax.plot(mean_rlon[track_valid], mean_rlat[track_valid], "k-", lw=2.5)
        if np.isfinite(mean_rlon[lag0]) and np.isfinite(mean_rlat[lag0]):
            ax.plot(mean_rlon[lag0], mean_rlat[lag0], "+", color="lime",
                    ms=14, mew=2.5, zorder=10)
    ax.set_xlim(lon[0], lon[-1])
    la0, la1 = float(lat[0]), float(lat[-1])
    if rel_lat_ymin is not None and la0 < la1:
        la0 = max(la0, float(rel_lat_ymin))
    ax.set_ylim(la0, la1)
    dx = float(lon[-1] - lon[0])
    dy = float(la1 - la0)
    if dx > 0 and dy > 0:
        ax.set_box_aspect(dy / dx)
    ax.axhline(0, color="k", lw=0.3, ls=":")
    ax.axvline(0, color="k", lw=0.3, ls=":")
    ax.set_title(title, fontsize=11, pad=2.5)
    ax.tick_params(axis="both", labelsize=10, pad=1.5)
    if ylabel:
        ax.set_ylabel("rel. lat (\u00b0)", fontsize=10, labelpad=1.5)
    if xlabel:
        ax.set_xlabel("rel. lon (\u00b0)", fontsize=10, labelpad=1.5)
    return cf


def plot_budget_wp_na_fig5(
        *,
        data_wp: dict[str, Any], budget_wp: dict[str, Any],
        data_na: dict[str, Any], budget_na: dict[str, Any],
        out_path: Path,
        reference: str = "recurvature",
        t_start: int = 0,
        t_end: int = 144,
        sigma: float = 0.0,
        sigma_3d: Optional[Tuple[float, float, float]] = None,
        fixed_vmax: Optional[float] = None,
        rel_lat_ymin: Optional[float] = -10.0,
        budget_rates_ms_day: bool = False,
        strat_column_titles: Optional[Tuple[str, str]] = None,
        include_bottom_row: bool = True,
        allow_missing_mst: bool = False,
        bottom_precip_title: str | None = None,
        lh_title: str | None = None,
        mst_title: str | None = None,
        precip_label: str | None = None,
        show_qgpv: bool = True,
        anomaly_lh_mst: bool = True,
        baseline_lag: Tuple[float, float] = (-48.0, -12.0),
        compare_with_diff_test: bool = False,
        show_flux_arrows: bool = True,
        mst_panel_mode: str = "merra2",
        suptitle: str | None = None,
        qgpv_field_key: str = "era5_qgpv_10km",
        mc_env_left: dict | None = None,
        mc_env_right: dict | None = None,
        use_rwb_nonqg: bool = False,
        nonqg_source_tag: str = "ERA5",
        stacked_layout: bool = False,
        show_coastlines: bool = True) -> Path:
    require_mst = (mst_panel_mode == "merra2") and not allow_missing_mst
    if require_mst:
        if (budget_wp.get("merra2_mst") is None
                or budget_na.get("merra2_mst") is None):
            raise RuntimeError(
                "plot_budget_wp_na_fig5 needs MERRA-2 DTDTMST composites "
                "(merra2_heat_fortran_DTDTMST) for both column blocks. "
                "Pass allow_missing_mst=True for MPAS-only layouts.")

    lag_wp = np.asarray(data_wp["_lag_hours"])
    lag_na = np.asarray(data_na["_lag_hours"])
    if lag_wp.shape != lag_na.shape or not np.allclose(lag_wp, lag_na,
                                                       rtol=0, atol=0):
        raise ValueError(
            "Left and right composites must share identical lag_hours.")

    if budget_rates_ms_day:
        budget_levels = np.asarray(BUDGET_LEVELS, dtype=float)
    elif fixed_vmax is not None:
        vmax = fixed_vmax
        budget_levels = np.linspace(-vmax, vmax, 9)
    else:
        vmax = _compute_joint_vmax(
            data_list=[data_wp, data_na], budget_list=[budget_wp, budget_na],
            lag_hours=lag_wp, t_start=t_start, t_end=t_end,
            sigma_2d=sigma, sigma_3d=sigma_3d)
        budget_levels = np.linspace(-vmax, vmax, 9)
    lwa_levels = np.linspace(20, 65, 19)
    fc_levels = np.linspace(0, 150, 16)
    awb_levels = np.linspace(0, 0.25, 11)
    rwb_freq_cmap = "plasma"
    precip_levels = np.linspace(0, 0.6, 13)

    M = _assemble_budget_map_fields(
        data=data_wp, budget=budget_wp, t_start=t_start, t_end=t_end,
        sigma_2d=sigma, sigma_3d=sigma_3d,
        anomaly_lh_mst=anomaly_lh_mst, baseline_lag=baseline_lag,
        qgpv_field_key=qgpv_field_key, mc_env=mc_env_left)
    N = _assemble_budget_map_fields(
        data=data_na, budget=budget_na, t_start=t_start, t_end=t_end,
        sigma_2d=sigma, sigma_3d=sigma_3d,
        anomaly_lh_mst=anomaly_lh_mst, baseline_lag=baseline_lag,
        qgpv_field_key=qgpv_field_key, mc_env=mc_env_right)

    precip_diff_field = None
    precip_diff_sig = None
    if mst_panel_mode == "precip_diff":
        precip_diff_field = np.abs(M["precip_f"] - N["precip_f"])
        if M.get("sig_precip") is not None:
            precip_diff_sig = M["sig_precip"]

    if compare_with_diff_test:
        lag_hours_diff = np.asarray(data_wp["_lag_hours"], dtype=float)
        bl_for_anom = baseline_lag if anomaly_lh_mst else None

        def _diff(alias_keys: tuple[str, ...], scale: float = 1.0,
                  bl: tuple[float, float] | None = None) -> np.ndarray | None:
            return _diff_sig_for_keys(
                data_a=data_wp, data_b=data_na, alias_keys=alias_keys,
                lag_hours=lag_hours_diff,
                t_start=t_start, t_end=t_end,
                sigma_3d=sigma_3d, sigma_2d=sigma,
                scale=scale,
                baseline_lag=bl,
            )

        for sig_key, alias_keys, sc, bl in (
            ("sig_t1", ("era5_budget_termI", "budget_termI"),
             _SEC_PER_DAY, bl_for_anom),
            ("sig_t2", ("era5_budget_termII", "budget_termII"),
             _SEC_PER_DAY, bl_for_anom),
            ("sig_t3", ("era5_budget_termIII", "budget_termIII"),
             _SEC_PER_DAY, bl_for_anom),
            ("sig_lwa", ("era5_lwa", "lwa"), 1.0, None),
            ("sig_lh", ("era5_lh_lwa", "mpas_lh_lwa", "lh_lwa"),
             _SEC_PER_DAY, bl_for_anom),
            ("sig_nonqg", ("era5_nonqg_lwa", "mpas_nonqg_lwa", "nonqg_lwa"),
             _SEC_PER_DAY, bl_for_anom),
            ("sig_mst", ("merra2_heat_fortran_DTDTMST",),
             _SEC_PER_DAY, bl_for_anom),
            ("sig_awb", ("era5_rwb_awb", "mpas_rwb_awb"), 1.0, None),
            ("sig_cwb", ("era5_rwb_cwb", "mpas_rwb_cwb"), 1.0, None),
            ("sig_fc", ("era5_cc_Fc", "mpas_cc_Fc"), 1.0, None),
            ("sig_precip", ("imerg_precip", "mpas_precip"), 1.0, None),
        ):
            ta = _diff(alias_keys, scale=sc, bl=bl)
            if ta is not None:
                M[sig_key] = ta
                N[sig_key] = ta
        for sig_key, mean_key, var_key, count_key, sc in (
            ("sig_tend", "tendency", "tendency_var", "tendency_count",
             1.0 if not budget_rates_ms_day else 1.0),
            ("sig_res", "residual", "residual_var", "residual_count",
             1.0 if not budget_rates_ms_day else 1.0),
        ):
            ta = _diff_sig_for_derived(
                mean_a_3d=budget_wp.get(mean_key),
                var_a_3d=budget_wp.get(var_key),
                count_a_3d=budget_wp.get(count_key),
                mean_b_3d=budget_na.get(mean_key),
                var_b_3d=budget_na.get(var_key),
                count_b_3d=budget_na.get(count_key),
                lag_hours=lag_hours_diff,
                t_start=t_start, t_end=t_end,
                sigma_3d=sigma_3d, sigma_2d=sigma,
                scale=sc,
                baseline_lag=bl_for_anom,
            )
            if ta is not None:
                M[sig_key] = ta
                N[sig_key] = ta
            else:
                M[sig_key] = None
                N[sig_key] = None

        for D in (M, N):
            sa, sc2 = D.get("sig_awb"), D.get("sig_cwb")
            if sa is not None and sc2 is not None:
                D["sig_rwb"] = sa | sc2
            elif sa is not None:
                D["sig_rwb"] = sa
            elif sc2 is not None:
                D["sig_rwb"] = sc2
            else:
                D["sig_rwb"] = None

    nrows_per_basin = 4 if include_bottom_row else 3
    rel_lat_top = 25.0
    rel_lat_bot = float(rel_lat_ymin) if rel_lat_ymin is not None else -25.0
    title_in = 0.30
    hspace_in = 0.18

    if stacked_layout:
        nrows = nrows_per_basin * 2
        ncols = 3
        fig_w = 16.0
        panel_w_in = (0.82 - 0.08) * fig_w / 3.0
        panel_h_in = panel_w_in * (rel_lat_top - rel_lat_bot) / 150.0
        bottom_in = 0.50
        top_in = 0.50 if strat_column_titles else 0.30
        if suptitle:
            top_in += 0.40
        sep_frac = 0.06
        fig_h = (nrows * (panel_h_in + title_in + hspace_in)
                 + bottom_in + top_in + 1.5)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                                 squeeze=False)
        base_hspace = hspace_in / panel_h_in
        fig.subplots_adjust(
            left=0.08, right=0.82,
            top=1.0 - top_in / fig_h,
            bottom=bottom_in / fig_h,
            wspace=0.14,
            hspace=base_hspace * 1.2,
        )
        fig.canvas.draw()
        for r in range(nrows_per_basin, nrows):
            for c in range(ncols):
                pos = axes[r, c].get_position()
                axes[r, c].set_position(
                    [pos.x0, pos.y0 - sep_frac, pos.width, pos.height])
    else:
        nrows = nrows_per_basin
        ncols = 6
        fig_w = 26.0
        panel_w_in = (0.885 - 0.052) * fig_w / 6.0
        panel_h_in = panel_w_in * (rel_lat_top - rel_lat_bot) / 150.0
        bottom_in = 1.10
        top_in = 0.40 if strat_column_titles else 0.20
        if suptitle:
            top_in += 0.40
        fig_h = nrows * (panel_h_in + title_in + hspace_in) + bottom_in + top_in
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                                 squeeze=False)
        fig.subplots_adjust(
            left=0.052, right=0.885,
            top=1.0 - top_in / fig_h,
            bottom=bottom_in / fig_h,
            wspace=0.07,
            hspace=hspace_in / panel_h_in,
        )

    if suptitle and strat_column_titles is not None:
        col_title_y = 1.0 - 0.75 * top_in / fig_h
    else:
        col_title_y = 1.0 - 0.55 * top_in / fig_h

    if stacked_layout and strat_column_titles is not None:
        lt, rt = strat_column_titles
        fig.text(0.45, col_title_y, lt, ha="center",
                 fontsize=15, fontweight="semibold", transform=fig.transFigure)
    elif strat_column_titles is not None:
        lt, rt = strat_column_titles
        fig.text(0.28, col_title_y, lt, ha="center",
                 fontsize=11, fontweight="semibold", transform=fig.transFigure)
        fig.text(0.72, col_title_y, rt, ha="center",
                 fontsize=11, fontweight="semibold", transform=fig.transFigure)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold",
                     y=1.0 - 0.20 * top_in / fig_h)

    cf_budget_ref = None
    hbar_cf: dict[str, Any] = {}
    hbar_map = {(0, 0): "lwa", (0, 2): "fc"}
    if include_bottom_row:
        hbar_map[(3, 0)] = "pr"
        hbar_map[(3, 1)] = "rwb" if use_rwb_nonqg else "awb"

    def _basin_block(Mb: dict[str, Any], r0: int, c0: int, letters: list[str],
                     *, draw_bottom: bool, is_first: bool) -> None:
        nonlocal cf_budget_ref
        lon, lat = Mb["lon"], Mb["lat"]
        cs = Mb["coast_segs"] if show_coastlines else None
        qf = Mb["qgpv_f"] if show_qgpv else None
        mr, mlat = Mb["mean_rlon"], Mb["mean_rlat"]
        tv, l0 = Mb["track_valid"], Mb["lag0"]
        fl, fp = Mb["f_lambda_avg"], Mb["f_phi_avg"]

        def _b(row_rel: int, col_rel: int, field: np.ndarray | None,
               title: str, levels: np.ndarray, cmap: Any,
               sig: np.ndarray | None = None,
               flux: bool = False, xl: bool = False) -> Any:
            nonlocal cf_budget_ref
            row = r0 + row_rel
            col = c0 + col_rel
            fu = fl if (flux and show_flux_arrows) else None
            fvv = fp if (flux and show_flux_arrows) else None
            yl = (col == 0)
            cf = _fig5_draw_panel(
                ax=axes[row, col], lon=lon, lat=lat, field=field, title=title,
                levels=levels, cmap=cmap,
                coast_segs=cs, qgpv_f=qf, sig_mask=sig,
                flux_u=fu, flux_v=fvv,
                mean_rlon=mr, mean_rlat=mlat, track_valid=tv, lag0=l0,
                ylabel=yl, xlabel=xl, rel_lat_ymin=rel_lat_ymin)
            if cmap is qj3._BWOR_8 and levels is budget_levels:
                cf_budget_ref = cf
            if is_first and (row_rel, col_rel) in hbar_map:
                hbar_cf[hbar_map[row_rel, col_rel]] = cf
            return cf

        _b(0, 0, Mb["lwa_f"],
           f"({letters[0]}) LWA (m s$^{{-1}}$)",
           lwa_levels, "RdYlBu_r", sig=Mb["sig_lwa"],
           xl=False)
        if budget_rates_ms_day:
            tend_anom = (
                r" anomaly (T$-$pre)" if anomaly_lh_mst else ""
            )
            _b(0, 1, Mb["tend_f"],
               f"({letters[1]}) $\\partial A/\\partial t${tend_anom}",
               budget_levels, qj3._BWOR_8, sig=Mb["sig_tend"], xl=False)
        else:
            _b(0, 1, Mb["tend_f"],
               f"({letters[1]}) LWA tendency = A(t) $-$ A(T$_0$)",
               budget_levels, qj3._BWOR_8, sig=Mb["sig_tend"], xl=False)
        _b(0, 2, Mb["fc_f"],
           f"({letters[2]}) Carrying capacity $F_c$ (m$^2$ s$^{{-2}}$)",
           fc_levels, "YlOrRd", sig=Mb.get("sig_fc"), xl=False)
        if show_flux_arrows:
            arrow_tag = (
                r" arrows: $\Delta(F_\lambda,F_\phi)$"
                if anomaly_lh_mst
                else r" arrows: $(F_\lambda,F_\phi)$"
            )
        else:
            arrow_tag = ""
        if budget_rates_ms_day:
            t1_t = (f"({letters[3]}) Term I:"
                    f"$-\\partial F_\\lambda/\\partial x$;{arrow_tag}")
            t2_t = f"({letters[4]}) Term II: meridional flux"
            t3_t = f"({letters[5]}) Term III"
        else:
            t1_t = f"({letters[3]}) $\\int$ I dt;{arrow_tag}"
            t2_t = f"({letters[4]}) $\\int$ Term II dt"
            t3_t = f"({letters[5]}) $\\int$ Term III dt"
        _b(1, 0, Mb["t1_f"], t1_t,
           budget_levels, qj3._BWOR_8, sig=Mb["sig_t1"], flux=True,
           xl=False)
        _b(1, 1, Mb["t2_f"], t2_t,
           budget_levels, qj3._BWOR_8, sig=Mb["sig_t2"], xl=False)
        _b(1, 2, Mb["t3_f"], t3_t,
           budget_levels, qj3._BWOR_8, sig=Mb["sig_t3"], xl=False)
        anom_tag = (
            r" anomaly (T$-$pre)"
            if anomaly_lh_mst and budget_rates_ms_day
            else ""
        )
        if budget_rates_ms_day:
            r_t = f"({letters[6]}) Residual (diabatic);{arrow_tag}"
            lh_t_def = (
                f"({letters[7]}) LH-LWA (ERA5){anom_tag}")
            mst_t_def = (
                f"({letters[8]}) DTDTMST (MERRA-2){anom_tag}")
        else:
            r_t = f"({letters[6]}) Residual;{arrow_tag}"
            lh_t_def = f"({letters[7]}) $\\int$ LH-LWA dt (ERA5)"
            mst_t_def = (f"({letters[8]}) $\\int$ DTDTMST dt "
                         "(MERRA-2 latent heating)")
        lh_t = (lh_title % {"letter": letters[7]}) if lh_title else lh_t_def
        mst_t = (mst_title % {"letter": letters[8]}) if mst_title else mst_t_def
        _b(2, 0, Mb["res_f"], r_t,
           budget_levels, qj3._BWOR_8, sig=Mb["sig_res"], flux=True,
           xl=not draw_bottom)
        _b(2, 1, Mb["lh_f"], lh_t,
           budget_levels, qj3._BWOR_8, sig=Mb["sig_lh"], xl=not draw_bottom)
        if mst_panel_mode == "precip_diff":
            diff_t = (
                f"({letters[8]}) |\N{GREEK CAPITAL LETTER DELTA} precipitation| "
                r"(mm hr$^{-1}$)"
            )
            _b(2, 2, precip_diff_field, diff_t,
               precip_levels, "YlGnBu", sig=precip_diff_sig,
               xl=not draw_bottom)
        elif mst_panel_mode == "none":
            ax_blank = axes[r0 + 2, c0 + 2]
            ax_blank.set_axis_off()
        else:
            _b(2, 2, Mb["mst_f"], mst_t,
               budget_levels, qj3._BWOR_8, sig=Mb["sig_mst"],
               xl=not draw_bottom)
        if draw_bottom:
            if bottom_precip_title is not None:
                pr_title = (
                    bottom_precip_title % {"letter": letters[9]}
                    if "%(letter)s" in bottom_precip_title
                    else bottom_precip_title
                )
            else:
                lab = precip_label or "IMERG precipitation (mm hr$^{-1}$)"
                pr_title = f"({letters[9]}) {lab}"
            _b(3, 0, Mb["precip_f"],
               pr_title,
               precip_levels, "YlGnBu", sig=Mb.get("sig_precip"), xl=True)
            if use_rwb_nonqg:
                _b(3, 1, Mb["rwb_f"],
                   f"({letters[10]}) RWB frequency (AWB+CWB)",
                   awb_levels, rwb_freq_cmap, sig=Mb.get("sig_rwb"), xl=True)
                _b(3, 2, Mb["nonqg_f"],
                   f"({letters[11]}) Non-QG source ({nonqg_source_tag})",
                   budget_levels, qj3._BWOR_8, sig=Mb.get("sig_nonqg"),
                   xl=True)
            else:
                _b(3, 1, Mb["awb_f"],
                   f"({letters[10]}) AWB frequency",
                   awb_levels, rwb_freq_cmap, sig=Mb.get("sig_awb"), xl=True)
                _b(3, 2, Mb["cwb_f"],
                   f"({letters[11]}) CWB frequency",
                   awb_levels, rwb_freq_cmap, sig=Mb.get("sig_cwb"), xl=True)

    letters_wp = list("abcdefghijkl")
    letters_na = list("mnopqrstuvwx") if stacked_layout else letters_wp
    if stacked_layout:
        _basin_block(M, 0, 0, letters_wp, draw_bottom=include_bottom_row,
                     is_first=True)
        _basin_block(N, nrows_per_basin, 0, letters_na,
                     draw_bottom=include_bottom_row, is_first=False)
    else:
        _basin_block(M, 0, 0, letters_wp, draw_bottom=include_bottom_row,
                     is_first=True)
        _basin_block(N, 0, 3, letters_wp, draw_bottom=include_bottom_row,
                     is_first=False)

    fig.canvas.draw()

    if stacked_layout:
        if strat_column_titles is not None:
            lt, rt = strat_column_titles
            p_wp_bot = axes[nrows_per_basin - 1, 1].get_position()
            p_na_top = axes[nrows_per_basin, 1].get_position()
            gap_y = p_wp_bot.y0 - p_na_top.y1
            na_title_y = p_na_top.y1 + 0.55 * gap_y
            fig.text(0.45, na_title_y, rt, ha="center",
                     fontsize=15, fontweight="semibold",
                     transform=fig.transFigure)

    def _vbar(rect: tuple[float, float, float, float], cf: Any, label: str,
              ticks: Any = None, label_fs: int = 11, tick_fs: int = 10,
              extend: str = "both") -> None:
        axv = fig.add_axes(rect)
        cb = fig.colorbar(cf, cax=axv, orientation="vertical", extend=extend)
        cb.set_label(label, fontsize=label_fs)
        axv.tick_params(labelsize=tick_fs)
        if ticks is not None:
            cb.set_ticks(ticks)

    def _hbar(rect: tuple[float, float, float, float], cf: Any, label: str,
              ticks: Any = None, label_fs: int = 8, tick_fs: int = 7) -> None:
        axh = fig.add_axes(rect)
        cb = fig.colorbar(cf, cax=axh, orientation="horizontal")
        cb.set_label(label, fontsize=label_fs)
        axh.tick_params(labelsize=tick_fs)
        if ticks is not None:
            cb.set_ticks(ticks)

    if stacked_layout:
        cbar_x = 0.84
        cbar_w = 0.020
        cbar_h = 0.10
        if cf_budget_ref is not None:
            _vbar((cbar_x, 0.55, cbar_w, 0.35), cf_budget_ref,
                  r"Anom. (m s$^{-1}$ d$^{-1}$)" if budget_rates_ms_day
                  else r"$\Delta$ (m s$^{-1}$)",
                  ticks=budget_levels, label_fs=11, tick_fs=10)
        _vbar((cbar_x, 0.42, cbar_w, cbar_h), hbar_cf["lwa"],
              r"LWA (m s$^{-1}$)",
              ticks=[20, 35, 50, 65], label_fs=11, tick_fs=10, extend="both")
        _vbar((cbar_x, 0.29, cbar_w, cbar_h), hbar_cf["fc"],
              r"$F_c$ (m$^2$ s$^{-2}$)",
              ticks=[0, 50, 100, 150], label_fs=11, tick_fs=10, extend="max")
        if include_bottom_row:
            _vbar((cbar_x, 0.16, cbar_w, cbar_h), hbar_cf["pr"],
                  r"Precip (mm/hr)",
                  ticks=[0, 0.2, 0.4, 0.6], label_fs=11, tick_fs=10,
                  extend="max")
            rwb_key = "rwb" if use_rwb_nonqg else "awb"
            rwb_label = "RWB freq" if use_rwb_nonqg else "AWB/CWB"
            _vbar((cbar_x, 0.03, cbar_w, cbar_h), hbar_cf[rwb_key],
                  rwb_label,
                  ticks=[0, 0.1, 0.2], label_fs=11, tick_fs=10, extend="max")
    else:
        if cf_budget_ref is not None:
            cax_y0, cax_h = ((0.20, 0.58) if not include_bottom_row
                             else (0.26, 0.66))
            cax_b = fig.add_axes((0.902, cax_y0, 0.014, cax_h))
            cb_b = fig.colorbar(cf_budget_ref, cax=cax_b, extend="both",
                                ticks=budget_levels)
            cb_b.set_label(
                r"Anomaly (m s$^{-1}$ day$^{-1}$)" if budget_rates_ms_day
                else r"$\Delta$ over period (m s$^{-1}$)",
                fontsize=11 if budget_rates_ms_day else 10)
            cb_b.ax.tick_params(labelsize=8)
        _hbar((0.055, 0.028, 0.17, 0.022), hbar_cf["lwa"],
              r"LWA (m s$^{-1}$)",
              ticks=np.linspace(20, 65, 6),
              label_fs=9, tick_fs=8)
        _hbar((0.26, 0.028, 0.15, 0.020), hbar_cf["fc"],
              r"$F_c$ (m$^2$ s$^{-2}$)", ticks=np.linspace(0, 150, 4),
              label_fs=8, tick_fs=7)
        if include_bottom_row:
            _hbar((0.44, 0.028, 0.15, 0.020), hbar_cf["pr"],
                  r"mm hr$^{-1}$", ticks=np.linspace(0, 0.6, 4),
                  label_fs=8, tick_fs=7)
            rwb_key = "rwb" if use_rwb_nonqg else "awb"
            rwb_label = ("RWB frequency" if use_rwb_nonqg
                         else "AWB / CWB frequency")
            _hbar((0.60, 0.028, 0.24, 0.022), hbar_cf[rwb_key],
                  rwb_label,
                  ticks=np.linspace(0, 0.25, 6),
                  label_fs=9, tick_fs=8)

        p2 = axes[0, 2].get_position()
        p3 = axes[0, 3].get_position()
        gap = p3.x0 - p2.x1
        x = p2.x1 + 0.07 * gap if gap > 1e-4 else 0.5 * (p2.x1 + p3.x0)
        y0 = min(axes[r, c].get_position().y0
                 for r in range(nrows) for c in range(ncols))
        y1 = max(axes[r, c].get_position().y1
                 for r in range(nrows) for c in range(ncols))
        fig.add_artist(
            matplotlib.lines.Line2D(
                [x, x], [y0, y1], transform=fig.transFigure,
                color="black", linewidth=1.8, zorder=200, clip_on=False))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("  Saved: %s", out)
    return out


def plot_budget_diff_fig(
        *,
        data_wp: dict[str, Any], budget_wp: dict[str, Any],
        data_na: dict[str, Any], budget_na: dict[str, Any],
        out_path: Path,
        reference: str = "recurvature",
        t_start: int = 0,
        t_end: int = 144,
        sigma: float = 0.0,
        sigma_3d: Optional[Tuple[float, float, float]] = None,
        rel_lat_ymin: Optional[float] = -10.0,
        budget_rates_ms_day: bool = False,
        title: str = "Difference (Group 1 \N{MINUS SIGN} Group 2)",
        include_bottom_row: bool = True,
        anomaly_lh_mst: bool = True,
        baseline_lag: Tuple[float, float] = (-48.0, -12.0),
        show_qgpv: bool = True,
        qgpv_field_key: str = "era5_qgpv_10km",
        use_rwb_nonqg: bool = False,
        nonqg_source_tag: str = "ERA5",
        sig_from_diff_test: bool = True,
        skip_mst: bool = False,
        show_coastlines: bool = True) -> Path:
    lag_wp = data_wp["_lag_hours"]
    lag_na = data_na["_lag_hours"]
    if lag_wp.shape != lag_na.shape or not np.allclose(lag_wp, lag_na):
        raise ValueError(
            "Left and right composites must share identical lag_hours.")

    if budget_rates_ms_day:
        budget_levels = np.asarray(BUDGET_LEVELS, dtype=float)
    else:
        budget_levels = np.linspace(-4, 4, 9)

    M = _assemble_budget_map_fields(
        data=data_wp, budget=budget_wp, t_start=t_start, t_end=t_end,
        sigma_2d=sigma, sigma_3d=sigma_3d,
        anomaly_lh_mst=anomaly_lh_mst, baseline_lag=baseline_lag,
        qgpv_field_key=qgpv_field_key)
    N = _assemble_budget_map_fields(
        data=data_na, budget=budget_na, t_start=t_start, t_end=t_end,
        sigma_2d=sigma, sigma_3d=sigma_3d,
        anomaly_lh_mst=anomaly_lh_mst, baseline_lag=baseline_lag,
        qgpv_field_key=qgpv_field_key)

    if use_rwb_nonqg:
        for D in [M, N]:
            if "awb_f" in D and "cwb_f" in D:
                D["rwb_f"] = D["awb_f"] + D["cwb_f"]

    if sig_from_diff_test:
        lag_hours_diff = np.asarray(data_wp["_lag_hours"], dtype=float)
        bl_for_anom = baseline_lag if anomaly_lh_mst else None

        def _diff(alias_keys: tuple[str, ...], scale: float = 1.0,
                  bl: tuple[float, float] | None = None) -> np.ndarray | None:
            return _diff_sig_for_keys(
                data_a=data_wp, data_b=data_na, alias_keys=alias_keys,
                lag_hours=lag_hours_diff,
                t_start=t_start, t_end=t_end,
                sigma_3d=sigma_3d, sigma_2d=sigma,
                scale=scale, baseline_lag=bl)

        for sig_key, alias_keys, sc, bl in (
            ("sig_t1", ("era5_budget_termI", "budget_termI"),
             _SEC_PER_DAY, bl_for_anom),
            ("sig_t2", ("era5_budget_termII", "budget_termII"),
             _SEC_PER_DAY, bl_for_anom),
            ("sig_t3", ("era5_budget_termIII", "budget_termIII"),
             _SEC_PER_DAY, bl_for_anom),
            ("sig_lwa", ("era5_lwa", "lwa"), 1.0, None),
            ("sig_lh", ("era5_lh_lwa", "mpas_lh_lwa", "lh_lwa"),
             _SEC_PER_DAY, bl_for_anom),
            ("sig_nonqg", ("era5_nonqg_lwa", "mpas_nonqg_lwa", "nonqg_lwa"),
             _SEC_PER_DAY, bl_for_anom),
            ("sig_mst", ("merra2_heat_fortran_DTDTMST",),
             _SEC_PER_DAY, bl_for_anom),
            ("sig_awb", ("era5_rwb_awb", "mpas_rwb_awb"), 1.0, None),
            ("sig_cwb", ("era5_rwb_cwb", "mpas_rwb_cwb"), 1.0, None),
            ("sig_fc", ("era5_cc_Fc", "mpas_cc_Fc"), 1.0, None),
            ("sig_precip", ("imerg_precip", "mpas_precip"), 1.0, None),
        ):
            ta = _diff(alias_keys, scale=sc, bl=bl)
            if ta is not None:
                M[sig_key] = ta
        for sig_key, mean_key, var_key, count_key in (
            ("sig_tend", "tendency", "tendency_var", "tendency_count"),
            ("sig_res", "residual", "residual_var", "residual_count"),
        ):
            ta = _diff_sig_for_derived(
                mean_a_3d=budget_wp.get(mean_key),
                var_a_3d=budget_wp.get(var_key),
                count_a_3d=budget_wp.get(count_key),
                mean_b_3d=budget_na.get(mean_key),
                var_b_3d=budget_na.get(var_key),
                count_b_3d=budget_na.get(count_key),
                lag_hours=lag_hours_diff,
                t_start=t_start, t_end=t_end,
                sigma_3d=sigma_3d, sigma_2d=sigma,
                baseline_lag=bl_for_anom)
            if ta is not None:
                M[sig_key] = ta
        if use_rwb_nonqg:
            sa, sc3 = M.get("sig_awb"), M.get("sig_cwb")
            if sa is not None and sc3 is not None:
                M["sig_rwb"] = sa | sc3
            elif sa is not None:
                M["sig_rwb"] = sa
            elif sc3 is not None:
                M["sig_rwb"] = sc3
            else:
                M["sig_rwb"] = None

    lon, lat = M["lon"], M["lat"]
    cs = M["coast_segs"] if show_coastlines else None

    qgpv_avg = None
    if show_qgpv and M.get("qgpv_f") is not None and N.get("qgpv_f") is not None:
        qgpv_avg = (M["qgpv_f"] + N["qgpv_f"]) / 2.0

    mr = (M["mean_rlon"] + N["mean_rlon"]) / 2.0
    mlat = (M["mean_rlat"] + N["mean_rlat"]) / 2.0
    tv = M["track_valid"] & N["track_valid"]
    l0 = M["lag0"]

    nrows = 4 if include_bottom_row else 3
    ncols = 3
    fig_w = 14.0
    panel_w_in = (0.82 - 0.10) * fig_w / 3.0
    rel_lat_top = 25.0
    rel_lat_bot = float(rel_lat_ymin) if rel_lat_ymin is not None else -25.0
    panel_h_in = panel_w_in * (rel_lat_top - rel_lat_bot) / 150.0
    title_in = 0.35
    hspace_in = 0.20
    bottom_in = 0.50
    top_in = 0.60
    fig_h = nrows * (panel_h_in + title_in + hspace_in) + bottom_in + top_in
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             squeeze=False)
    fig.subplots_adjust(
        left=0.10, right=0.82,
        top=1.0 - top_in / fig_h,
        bottom=bottom_in / fig_h,
        wspace=0.14,
        hspace=hspace_in / panel_h_in,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold",
                 y=1.0 - 0.25 * top_in / fig_h)

    cf_budget_ref = None
    hbar_cf: dict[str, Any] = {}

    def _diff_avg(key: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        m_f, n_f = M.get(key), N.get(key)
        if m_f is None or n_f is None:
            return None, None
        return m_f - n_f, (m_f + n_f) / 2.0

    def _b(row: int, col: int, diff_f: np.ndarray | None,
           avg_f: np.ndarray | None, ttl: str, levels: np.ndarray, cmap: Any,
           sig: np.ndarray | None = None, xl: bool = False,
           avg_levs: list | None = None) -> Any:
        nonlocal cf_budget_ref
        yl = (col == 0)
        cf = _fig5_draw_panel(
            ax=axes[row, col], lon=lon, lat=lat, field=diff_f, title=ttl,
            levels=levels, cmap=cmap,
            coast_segs=cs, qgpv_f=qgpv_avg, sig_mask=sig,
            mean_rlon=mr, mean_rlat=mlat, track_valid=tv, lag0=l0,
            ylabel=yl, xlabel=xl, rel_lat_ymin=rel_lat_ymin,
            avg_field=avg_f, avg_levels=avg_levs)
        if cmap is qj3._BWOR_8 and levels is budget_levels:
            cf_budget_ref = cf
        return cf

    letters = list("abcdefghijkl")

    diff_budget_levels = budget_levels
    avg_budget_levels = list(budget_levels[budget_levels != 0])

    d_lwa, a_lwa = _diff_avg("lwa_f")
    lwa_diff_levels = np.linspace(-15, 15, 16)
    hbar_cf["lwa"] = _b(
        0, 0, d_lwa, a_lwa,
        f"({letters[0]}) \N{GREEK CAPITAL LETTER DELTA}LWA (m s$^{{-1}}$)",
        lwa_diff_levels, "RdBu_r", sig=M.get("sig_lwa"),
        avg_levs=[25, 35, 45, 55])

    d_tend, a_tend = _diff_avg("tend_f")
    _b(0, 1, d_tend, a_tend,
       (f"({letters[1]}) \N{GREEK CAPITAL LETTER DELTA}(\u2202A/\u2202t)"
        if budget_rates_ms_day
        else f"({letters[1]}) \N{GREEK CAPITAL LETTER DELTA}LWA tendency"),
       diff_budget_levels, qj3._BWOR_8, sig=M.get("sig_tend"),
       avg_levs=avg_budget_levels)

    d_fc, a_fc = _diff_avg("fc_f")
    fc_diff_levels = np.linspace(-40, 40, 17)
    hbar_cf["fc"] = _b(
        0, 2, d_fc, a_fc,
        f"({letters[2]}) \N{GREEK CAPITAL LETTER DELTA}$F_c$ (m$^2$ s$^{{-2}}$)",
        fc_diff_levels, "RdBu_r", sig=M.get("sig_fc"),
        avg_levs=[30, 60, 90, 120])

    d_t1, a_t1 = _diff_avg("t1_f")
    _b(1, 0, d_t1, a_t1,
       f"({letters[3]}) \N{GREEK CAPITAL LETTER DELTA}Term I",
       diff_budget_levels, qj3._BWOR_8,
       sig=M.get("sig_t1"), avg_levs=avg_budget_levels)

    d_t2, a_t2 = _diff_avg("t2_f")
    _b(1, 1, d_t2, a_t2,
       f"({letters[4]}) \N{GREEK CAPITAL LETTER DELTA}Term II",
       diff_budget_levels, qj3._BWOR_8,
       sig=M.get("sig_t2"), avg_levs=avg_budget_levels)

    d_t3, a_t3 = _diff_avg("t3_f")
    _b(1, 2, d_t3, a_t3,
       f"({letters[5]}) \N{GREEK CAPITAL LETTER DELTA}Term III",
       diff_budget_levels, qj3._BWOR_8,
       sig=M.get("sig_t3"), avg_levs=avg_budget_levels)

    d_res, a_res = _diff_avg("res_f")
    _b(2, 0, d_res, a_res,
       f"({letters[6]}) \N{GREEK CAPITAL LETTER DELTA}Residual",
       diff_budget_levels, qj3._BWOR_8,
       sig=M.get("sig_res"), xl=not include_bottom_row,
       avg_levs=avg_budget_levels)

    d_lh, a_lh = _diff_avg("lh_f")
    _b(2, 1, d_lh, a_lh,
       f"({letters[7]}) \N{GREEK CAPITAL LETTER DELTA}LH-LWA",
       diff_budget_levels, qj3._BWOR_8,
       sig=M.get("sig_lh"), xl=not include_bottom_row,
       avg_levs=avg_budget_levels)

    d_mst, a_mst = _diff_avg("mst_f")
    if skip_mst or d_mst is None or np.all(np.isnan(d_mst)):
        axes[2, 2].set_axis_off()
    else:
        _b(2, 2, d_mst, a_mst,
           f"({letters[8]}) \N{GREEK CAPITAL LETTER DELTA}MST",
           diff_budget_levels, qj3._BWOR_8,
           sig=M.get("sig_mst"), xl=not include_bottom_row,
           avg_levs=avg_budget_levels)

    if include_bottom_row:
        d_pr, a_pr = _diff_avg("precip_f")
        pr_diff_levels = np.linspace(-0.3, 0.3, 13)
        hbar_cf["pr"] = _b(
            3, 0, d_pr, a_pr,
            f"({letters[9]}) \N{GREEK CAPITAL LETTER DELTA}Precip (mm/hr)",
            pr_diff_levels, "BrBG", sig=M.get("sig_precip"), xl=True,
            avg_levs=[0.1, 0.2, 0.3, 0.4])

        if use_rwb_nonqg:
            d_rwb, a_rwb = _diff_avg("rwb_f")
            rwb_diff_levels = np.linspace(-0.1, 0.1, 11)
            hbar_cf["rwb"] = _b(
                3, 1, d_rwb, a_rwb,
                f"({letters[10]}) \N{GREEK CAPITAL LETTER DELTA}RWB freq",
                rwb_diff_levels, "PuOr_r", sig=M.get("sig_rwb"), xl=True,
                avg_levs=[0.05, 0.1, 0.15, 0.2])

            d_nq, a_nq = _diff_avg("nonqg_f")
            if d_nq is not None:
                _b(3, 2, d_nq, a_nq,
                   f"({letters[11]}) \N{GREEK CAPITAL LETTER DELTA}Non-QG source",
                   diff_budget_levels, qj3._BWOR_8, sig=M.get("sig_nonqg"),
                   xl=True,
                   avg_levs=avg_budget_levels)
            else:
                axes[3, 2].set_axis_off()
        else:
            d_awb, a_awb = _diff_avg("awb_f")
            d_cwb, a_cwb = _diff_avg("cwb_f")
            rwb_diff_levels = np.linspace(-0.1, 0.1, 11)
            hbar_cf["rwb"] = _b(
                3, 1, d_awb, a_awb,
                f"({letters[10]}) \N{GREEK CAPITAL LETTER DELTA}AWB freq",
                rwb_diff_levels, "PuOr_r", sig=M.get("sig_awb"), xl=True,
                avg_levs=[0.05, 0.1, 0.15])
            _b(3, 2, d_cwb, a_cwb,
               f"({letters[11]}) \N{GREEK CAPITAL LETTER DELTA}CWB freq",
               rwb_diff_levels, "PuOr_r", sig=M.get("sig_cwb"), xl=True,
               avg_levs=[0.05, 0.1, 0.15])

    def _vbar(rect: tuple[float, float, float, float], cf: Any, label: str,
              ticks: Any = None, label_fs: int = 11, tick_fs: int = 10,
              extend: str = "both") -> None:
        axv = fig.add_axes(rect)
        cb = fig.colorbar(cf, cax=axv, orientation="vertical", extend=extend)
        cb.set_label(label, fontsize=label_fs)
        axv.tick_params(labelsize=tick_fs)
        if ticks is not None:
            cb.set_ticks(ticks)

    cbar_x = 0.84
    cbar_w = 0.018
    cbar_h = 0.09
    if cf_budget_ref is not None:
        _vbar((cbar_x, 0.58, cbar_w, 0.32), cf_budget_ref,
              ("\N{GREEK CAPITAL LETTER DELTA} (m s$^{-1}$ d$^{-1}$)"
               if budget_rates_ms_day
               else "\N{GREEK CAPITAL LETTER DELTA} (m s$^{-1}$)"),
              ticks=[-4, -2, 0, 2, 4], label_fs=10, tick_fs=9)
    _vbar((cbar_x, 0.46, cbar_w, cbar_h), hbar_cf["lwa"],
          "\N{GREEK CAPITAL LETTER DELTA}LWA",
          ticks=[-10, 0, 10], label_fs=10, tick_fs=9)
    _vbar((cbar_x, 0.34, cbar_w, cbar_h), hbar_cf["fc"],
          "\N{GREEK CAPITAL LETTER DELTA}$F_c$",
          ticks=[-30, 0, 30], label_fs=10, tick_fs=9)
    if include_bottom_row:
        _vbar((cbar_x, 0.22, cbar_w, cbar_h), hbar_cf["pr"],
              "\N{GREEK CAPITAL LETTER DELTA}Precip",
              ticks=[-0.2, 0, 0.2], label_fs=10, tick_fs=9)
        if "rwb" in hbar_cf:
            _vbar((cbar_x, 0.10, cbar_w, cbar_h), hbar_cf["rwb"],
                  "\N{GREEK CAPITAL LETTER DELTA}RWB",
                  ticks=[-0.05, 0, 0.05], label_fs=10, tick_fs=9)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("  Saved: %s", out)
    return out


def _get_coastlines_shifted(*, mean_lat: float, mean_lon: float
                            ) -> list[tuple[np.ndarray, np.ndarray]]:
    coast = cartopy.feature.NaturalEarthFeature("physical", "coastline", "110m")
    segments = []
    for geom in coast.geometries():
        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        elif geom.geom_type == "LineString":
            lines = [geom]
        else:
            continue
        for line in lines:
            coords = np.array(line.coords)
            rel_lon = coords[:, 0] - mean_lon
            rel_lat = coords[:, 1] - mean_lat
            rel_lon = np.where(rel_lon > 180, rel_lon - 360, rel_lon)
            rel_lon = np.where(rel_lon < -180, rel_lon + 360, rel_lon)
            segments.append((rel_lon, rel_lat))
    return segments


def plot_budget_composite(*, data: dict[str, Any], budget: dict[str, Any],
                          basin: str, reference: str, outdir: Path,
                          t_start: int = 0, t_end: int = 168,
                          sigma: float = 0.0,
                          sigma_3d: tuple[float, float, float] | None = None,
                          fixed_vmax: float | None = None,
                          group: str | None = None) -> None:
    lag_hours = data["_lag_hours"]
    lat = data["_lat"]
    lon = data["_lon"]
    n = data["_n_storms"]
    tavg = _make_tavg(lag_hours=lag_hours, t_start=t_start, t_end=t_end,
                      sigma_3d=sigma_3d, sigma_2d=sigma)

    lwa_f = tavg(budget["lwa"])
    tend_f = tavg(budget["tendency"])
    t1_f = tavg(budget["termI"])
    t2_f = tavg(budget["termII"])
    t3_f = tavg(budget["termIII"])
    res_f = tavg(budget["residual"])
    lh_f = tavg(budget["lh_lwa"])

    have_merra2 = budget.get("merra2_mst") is not None
    mst_f = rad_f = ana_f = tot_f = None
    if have_merra2:
        mst_f = tavg(budget["merra2_mst"])
        rad_f = tavg(budget["merra2_rad"])
        ana_f = tavg(budget["merra2_ana"])
        tot_f = tavg(budget["merra2_tot"])

    sig_t1 = time_average_sig(sig_bool_3d=budget.get("termI_sig"),
                              lag_hours=lag_hours, t_start=t_start, t_end=t_end)
    sig_t2 = time_average_sig(sig_bool_3d=budget.get("termII_sig"),
                              lag_hours=lag_hours, t_start=t_start, t_end=t_end)
    sig_t3 = time_average_sig(sig_bool_3d=budget.get("termIII_sig"),
                              lag_hours=lag_hours, t_start=t_start, t_end=t_end)
    sig_awb = time_average_sig(sig_bool_3d=budget.get("awb_sig"),
                               lag_hours=lag_hours, t_start=t_start,
                               t_end=t_end)
    sig_cwb = time_average_sig(sig_bool_3d=budget.get("cwb_sig"),
                               lag_hours=lag_hours, t_start=t_start,
                               t_end=t_end)
    sig_lh = time_average_sig(sig_bool_3d=budget.get("lh_lwa_sig"),
                              lag_hours=lag_hours, t_start=t_start, t_end=t_end)
    sig_tend = time_average_sig(sig_bool_3d=budget.get("tendency_sig"),
                                lag_hours=lag_hours, t_start=t_start,
                                t_end=t_end)
    sig_res = time_average_sig(sig_bool_3d=budget.get("residual_sig"),
                               lag_hours=lag_hours, t_start=t_start,
                               t_end=t_end)
    sig_mst = (time_average_sig(sig_bool_3d=budget.get("merra2_mst_sig"),
                                lag_hours=lag_hours, t_start=t_start,
                                t_end=t_end)
               if have_merra2 else None)
    sig_rad = (time_average_sig(sig_bool_3d=budget.get("merra2_rad_sig"),
                                lag_hours=lag_hours, t_start=t_start,
                                t_end=t_end)
               if have_merra2 else None)
    sig_ana = (time_average_sig(sig_bool_3d=budget.get("merra2_ana_sig"),
                                lag_hours=lag_hours, t_start=t_start,
                                t_end=t_end)
               if have_merra2 else None)
    sig_tot = (time_average_sig(sig_bool_3d=budget.get("merra2_tot_sig"),
                                lag_hours=lag_hours, t_start=t_start,
                                t_end=t_end)
               if have_merra2 else None)

    awb_raw = data.get("mpas_rwb_awb")
    if awb_raw is None or not np.any(np.isfinite(awb_raw)):
        awb_raw = data.get("era5_rwb_awb", np.full_like(budget["lwa"], np.nan))
    cwb_raw = data.get("mpas_rwb_cwb")
    if cwb_raw is None or not np.any(np.isfinite(cwb_raw)):
        cwb_raw = data.get("era5_rwb_cwb", np.full_like(budget["lwa"], np.nan))
    fc_raw = data.get("mpas_cc_Fc")
    if fc_raw is None or not np.any(np.isfinite(fc_raw)):
        fc_raw = data.get("era5_cc_Fc", np.full_like(budget["lwa"], np.nan))
    qgpv_raw = data.get("era5_qgpv_10km")
    if qgpv_raw is None or not np.any(np.isfinite(qgpv_raw)):
        qgpv_raw = np.full_like(budget["lwa"], np.nan)
    precip_raw = data.get("mpas_precip")
    if precip_raw is None or not np.any(np.isfinite(precip_raw)):
        precip_raw = data.get(
            "imerg_precip", np.full_like(budget["lwa"], np.nan))
    awb_f = tavg(awb_raw)
    cwb_f = tavg(cwb_raw)
    fc_f = tavg(fc_raw, do_smooth=False)
    qgpv_f = tavg(qgpv_raw)
    precip_f = tavg(precip_raw, do_smooth=False)

    ua1_raw = (data.get("era5_ua1") if data.get("era5_ua1") is not None
               else data.get("ua1"))
    ua2_raw = (data.get("era5_ua2") if data.get("era5_ua2") is not None
               else data.get("ua2"))
    ep1_raw = (data.get("era5_ep1") if data.get("era5_ep1") is not None
               else data.get("ep1"))
    ep2a_raw = (data.get("era5_ep2a") if data.get("era5_ep2a") is not None
                else data.get("ep2a"))
    ep3a_raw = (data.get("era5_ep3a") if data.get("era5_ep3a") is not None
                else data.get("ep3a"))

    f_lambda_avg = None
    f_phi_avg = None

    if ua1_raw is not None:
        f_lambda = ua1_raw.copy()
        if ua2_raw is not None:
            f_lambda = f_lambda + ua2_raw
        if ep1_raw is not None:
            f_lambda = f_lambda + ep1_raw
        f_lambda_avg = tavg(f_lambda)

    if ep2a_raw is not None and ep3a_raw is not None:
        mean_abs_lat = data.get("_mean_abs_lat", 35.0)
        abs_lat = mean_abs_lat + lat
        cosphi = np.cos(np.deg2rad(abs_lat))
        cosphi = np.maximum(cosphi, 0.1)
        f_phi = 0.5 * (ep2a_raw + ep3a_raw) / cosphi[:, np.newaxis, np.newaxis]
        f_phi_avg = tavg(f_phi)

    mean_rlat = data.get("_mean_rel_lat", np.full(len(lag_hours), np.nan))
    mean_rlon = data.get("_mean_rel_lon", np.full(len(lag_hours), np.nan))
    track_mask = (lag_hours >= t_start - 24) & (lag_hours <= t_end + 24)
    track_valid = np.isfinite(mean_rlat) & np.isfinite(mean_rlon) & track_mask
    lag0 = int(np.argmin(np.abs(lag_hours)))

    qgpv_levels = [1e-4, 1.5e-4, 2e-4]

    coast_segs = None
    if "_mean_abs_lat" in data:
        coast_segs = _get_coastlines_shifted(
            mean_lat=data["_mean_abs_lat"], mean_lon=data["_mean_abs_lon"])

    if fixed_vmax is not None:
        vmax = fixed_vmax
    else:
        budget_fields: list[np.ndarray | None] = [
            tend_f, t1_f, t2_f, t3_f, res_f, lh_f]
        if have_merra2:
            budget_fields += [mst_f, rad_f, ana_f, tot_f]
        all_vals = np.concatenate([f.ravel() for f in budget_fields
                                   if f is not None
                                   and np.any(np.isfinite(f))])
        finite_vals = all_vals[np.isfinite(all_vals)]
        vmax = (max(np.nanpercentile(np.abs(finite_vals), 95), 1.0)
                if len(finite_vals) > 0 else 10.0)
    budget_levels = np.linspace(-vmax, vmax, 21)

    n_rows = 5 if have_merra2 else 4
    fig = plt.figure(figsize=(20, 5.8 * n_rows))
    gs = fig.add_gridspec(n_rows, 5, width_ratios=[1, 1, 1, 0.04, 0.04],
                          hspace=0.28, wspace=0.25)

    def _add_coastlines(ax: matplotlib.axes.Axes) -> None:
        if coast_segs is None:
            return
        for rl, ra in coast_segs:
            in_box = ((rl > lon[0] - 5) & (rl < lon[-1] + 5)
                      & (ra > lat[0] - 5) & (ra < lat[-1] + 5))
            if in_box.sum() > 1:
                ax.plot(rl[in_box], ra[in_box], color="0.3",
                        lw=0.8, alpha=0.7, zorder=2)

    def _plot(row: int, col: int, field: np.ndarray | None, title: str,
              levels: np.ndarray, cmap: Any,
              sig_mask: np.ndarray | None = None,
              flux_u: np.ndarray | None = None,
              flux_v: np.ndarray | None = None) -> Any:
        ax = fig.add_subplot(gs[row, col])
        _add_coastlines(ax)
        cf = ax.contourf(lon, lat, field, levels=levels,
                         cmap=cmap, extend="both")
        if sig_mask is not None:
            ax.contourf(lon, lat, sig_mask.astype(float),
                        levels=[0.5, 1.5], colors="none", hatches=[".."],
                        zorder=3)
        if np.any(np.isfinite(qgpv_f)):
            qgpv_smooth = scipy.ndimage.gaussian_filter(
                np.where(np.isfinite(qgpv_f), qgpv_f, 0.0), sigma=(1.5, 1.5))
            for lev in qgpv_levels:
                ax.contour(lon, lat, qgpv_smooth, levels=[lev],
                           colors="green", linewidths=0.9, linestyles="--",
                           alpha=0.85)
        if flux_u is not None:
            fv = flux_v if flux_v is not None else np.zeros_like(flux_u)
            skip = 3
            lon_q = lon[::skip]
            lat_q = lat[::skip]
            u_q = flux_u[::skip, ::skip]
            v_q = fv[::skip, ::skip]
            mag = np.sqrt(u_q**2 + v_q**2)
            ref_mag = max(np.nanpercentile(mag[np.isfinite(mag)], 90), 1.0)
            ax.quiver(lon_q, lat_q, u_q, v_q,
                      scale=ref_mag * 12, scale_units="width",
                      headwidth=4, headlength=5, headaxislength=4,
                      color="k", alpha=0.7, zorder=5,
                      minshaft=1.5, minlength=0.5)
        ax.plot(mean_rlon[track_valid], mean_rlat[track_valid], "k-", lw=2.5)
        if np.isfinite(mean_rlon[lag0]) and np.isfinite(mean_rlat[lag0]):
            ax.plot(mean_rlon[lag0], mean_rlat[lag0], "+", color="lime",
                    ms=14, mew=2.5, zorder=10)
        ax.set_xlim(lon[0], lon[-1])
        ax.set_ylim(lat[0], lat[-1])
        ax.axhline(0, color="k", lw=0.3, ls=":")
        ax.axvline(0, color="k", lw=0.3, ls=":")
        ax.set_title(title, fontsize=10)
        if col == 0:
            ax.set_ylabel("lat rel. to center (\u00b0)")
        if row == n_rows - 1:
            ax.set_xlabel("lon rel. to center (\u00b0)")
        return cf

    cf_b = _plot(0, 0, res_f,
                 r"(a) Residual; arrows: ($F_\lambda$, $F_\phi$)",
                 budget_levels, "RdBu_r", sig_mask=sig_res,
                 flux_u=f_lambda_avg, flux_v=f_phi_avg)
    _plot(0, 1, tend_f, r"(b) LWA tendency = A(t) $-$ A(T$_0$)",
          budget_levels, "RdBu_r", sig_mask=sig_tend)
    _plot(0, 2, lwa_f, "(c) LWA (m/s)",
          np.linspace(20, 65, 19), "RdYlBu_r")

    _plot(1, 0, t1_f,
          r"(d) $\int$ I dt ($-\partial F_\lambda/\partial x$); "
          r"arrows: ($F_\lambda$, $F_\phi$)",
          budget_levels, "RdBu_r", sig_mask=sig_t1,
          flux_u=f_lambda_avg, flux_v=f_phi_avg)
    _plot(1, 1, t2_f, r"(e) $\int$ Term II dt (meridional flux)",
          budget_levels, "RdBu_r", sig_mask=sig_t2)
    _plot(1, 2, t3_f, r"(f) $\int$ Term III dt (non-conservative)",
          budget_levels, "RdBu_r", sig_mask=sig_t3)

    _plot(2, 0, lh_f, r"(g) $\int$ LH-LWA dt (ERA5 latent-heating source)",
          budget_levels, "RdBu_r", sig_mask=sig_lh)
    cf_awb = _plot(2, 1, awb_f, "(h) AWB frequency",
                   np.linspace(0, 0.25, 11), "Oranges", sig_mask=sig_awb)
    cf_cwb = _plot(2, 2, cwb_f, "(i) CWB frequency",
                   np.linspace(0, 0.25, 11), "Blues", sig_mask=sig_cwb)

    if have_merra2:
        _plot(3, 0, mst_f,
              r"(j) $\int$ DTDTMST dt (MERRA-2 latent heating)",
              budget_levels, "RdBu_r", sig_mask=sig_mst)
        _plot(3, 1, rad_f,
              r"(k) $\int$ DTDTRAD dt (MERRA-2 radiation)",
              budget_levels, "RdBu_r", sig_mask=sig_rad)
        _plot(3, 2, ana_f,
              r"(l) $\int$ DTDTANA dt (MERRA-2 analysis increment)",
              budget_levels, "RdBu_r", sig_mask=sig_ana)
        bot_row = 4
        lbl_tot = "(m)"
        lbl_fc = "(n)"
        lbl_pr = "(o)"
    else:
        bot_row = 3
        lbl_tot = None
        lbl_fc = "(j)"
        lbl_pr = "(k)"

    if have_merra2:
        _plot(bot_row, 0, tot_f,
              f"{lbl_tot} $\\int$ DTDTTOT dt (MERRA-2 total diabatic)",
              budget_levels, "RdBu_r", sig_mask=sig_tot)
        fc_col, pr_col = 1, 2
    else:
        fc_col, pr_col = 0, 1

    cf_fc = _plot(bot_row, fc_col, fc_f,
                  f"{lbl_fc} Carrying capacity $F_c$ (m$^2$/s$^2$)",
                  np.linspace(0, 150, 16), "YlOrRd")
    cf_pr = _plot(bot_row, pr_col, precip_f,
                  f"{lbl_pr} IMERG precipitation (mm/hr)",
                  np.linspace(0, 0.6, 13), "YlGnBu")

    budget_row_end = 4 if have_merra2 else 3
    cbar_ax = fig.add_subplot(gs[0:budget_row_end, 3])
    fig.colorbar(cf_b, cax=cbar_ax, label=r"$\Delta$ over period (m/s)")

    cb_ax_awb = fig.add_axes((0.38, 0.02, 0.14, 0.010))
    fig.colorbar(cf_awb, cax=cb_ax_awb, orientation="horizontal",
                 label="AWB freq")

    cb_ax_cwb = fig.add_axes((0.56, 0.02, 0.14, 0.010))
    fig.colorbar(cf_cwb, cax=cb_ax_cwb, orientation="horizontal",
                 label="CWB freq")

    cb_ax_fc = fig.add_axes((0.06, 0.02, 0.14, 0.010))
    fig.colorbar(cf_fc, cax=cb_ax_fc, orientation="horizontal",
                 label=r"$F_c$ (m$^2$/s$^2$)")

    cb_ax_pr = fig.add_axes((0.76, 0.02, 0.14, 0.010))
    fig.colorbar(cf_pr, cax=cb_ax_pr, orientation="horizontal",
                 label="mm/hr")

    ref_name = "Recurvature" if reference == "recurvature" else "ET"
    group_label = f" [{group}]" if group else ""
    sig3 = _sigma_3d_for_maps(sigma_3d=sigma_3d)
    smooth_note = (
        f"3D \u03c3(lat,lon,lag)={sig3}"
        + (f", +2D \u03c3={sigma}\u00b0" if sigma > 0 else ""))
    fig.suptitle(
        f"TC-relative composite: {ref_name}, {basin}{group_label} (n={n})\n"
        f"Time-integrated budget T+{t_start}h to T+{t_end}h "
        f"({(t_end - t_start) / 24:.0f} days post-{ref_name.lower()})  "
        f"[{smooth_note}, "
        f"hatching = sig. (p<0.05), "
        f"green = QGPV, gray = coastlines]",
        fontsize=11, y=0.99)

    grp_suffix = f"_{group}" if group else ""
    fname = (f"budget_composite_{basin}_{reference}"
             f"_T{t_start:+d}to{t_end:+d}{grp_suffix}.png")
    path = Path(outdir) / fname
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("  Saved: %s", path)


def _compute_joint_vmax(*, data_list: list[dict[str, Any]],
                        budget_list: list[dict[str, Any]],
                        lag_hours: np.ndarray, t_start: float, t_end: float,
                        sigma_2d: float = 0.0,
                        sigma_3d: tuple[float, ...] | None = None) -> float:
    sig3 = _sigma_3d_for_maps(sigma_3d=sigma_3d)
    all_vals = []
    for budget in budget_list:
        for key in ["tendency", "termI", "termII", "termIII", "residual",
                    "lh_lwa",
                    "merra2_mst", "merra2_rad", "merra2_ana", "merra2_tot"]:
            f = budget.get(key)
            if f is None:
                continue
            raw = time_average(
                field_3d=_smooth_volume(field_3d=f, sigma_3d=sig3),
                lag_hours=lag_hours, t_start=t_start, t_end=t_end)
            if sigma_2d > 0:
                field = smooth(field=raw, sigma=sigma_2d)
            else:
                field = raw
            all_vals.append(field.ravel())
    cat = np.concatenate(all_vals)
    finite = cat[np.isfinite(cat)]
    return (max(np.nanpercentile(np.abs(finite), 95), 1.0)
            if len(finite) > 0 else 10.0)


def main(
        composites_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_<reference>_<basin>.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for figure outputs")],
        supp_directory: Annotated[Optional[Path], typer.Option(
            help="Directory with *_merra2src.nc supplementary "
                 "composites")] = None,
        basin: Annotated[Optional[list[str]], typer.Option()] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature, et, or both")] = "both",
        t_start: Annotated[int, typer.Option()] = 0,
        t_end: Annotated[int, typer.Option(
            help="Lag upper bound (hours) for time mean")] = 144,
        rel_lat_ymin: Annotated[float, typer.Option(
            help="With --wp-na-fig5: southern limit of rel. latitude "
                 "(deg)")] = -10.0,
        sigma: Annotated[float, typer.Option(
            help="Extra isotropic 2D Gaussian sigma (deg) after the time "
                 "mean")] = 0.0,
        volume_sigma: Annotated[Optional[list[float]], typer.Option(
            help="Override 3D Gaussian sigma (rel_lat, rel_lon, lag grid "
                 "units) before time mean (3 floats)")] = None,
        unified: Annotated[bool, typer.Option(
            help="Use unified colorbar across all basins")] = False,
        fixed_vmax: Annotated[Optional[float], typer.Option()] = None,
        group: Annotated[Optional[str], typer.Option(
            help="Stratification group label (e.g. highwb, lowwb)")] = None,
        wp_na_fig5: Annotated[bool, typer.Option(
            help="Single 4x6 figure: WP | NA (no suptitle), NASA Fig. 5 "
                 "layout")] = False,
        integrated_budget_maps: Annotated[bool, typer.Option(
            help="With --wp-na-fig5: use lag-integrated budget (m/s) "
                 "instead of rates (m/s/day)")] = False,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if basin is None:
        basin = ["WP"]
    vol_sigma = (tuple(volume_sigma) if volume_sigma is not None else None)
    if vol_sigma is not None and len(vol_sigma) != 3:
        raise typer.BadParameter("--volume-sigma needs exactly 3 floats")

    outdir = Path(output_directory)
    os.makedirs(outdir, exist_ok=True)

    refs = (["recurvature", "et"] if reference == "both" else [reference])

    if wp_na_fig5:
        for ref in refs:
            print(f"\n{'=' * 60}\n  Fig. 5 WP|NA composite: {ref}\n{'=' * 60}")
            data_wp = load_composite(basin="WP", reference=ref,
                                     composites_dir=composites_directory,
                                     group=group, supp_dir=supp_directory)
            data_na = load_composite(basin="NA", reference=ref,
                                     composites_dir=composites_directory,
                                     group=group, supp_dir=supp_directory)
            if data_wp is None or data_na is None:
                print("  Missing WP or NA composite, skipping.")
                continue
            use_rates = not integrated_budget_maps
            budget_wp = compute_budget(data=data_wp,
                                       strip_rate_units=use_rates)
            budget_na = compute_budget(data=data_na,
                                       strip_rate_units=use_rates)
            if budget_wp is None or budget_na is None:
                print("  Missing budget variables, skipping.")
                continue
            fvmax = fixed_vmax
            if not use_rates and fvmax is None:
                lag_hours = data_wp["_lag_hours"]
                fvmax = _compute_joint_vmax(
                    data_list=[data_wp, data_na],
                    budget_list=[budget_wp, budget_na],
                    lag_hours=lag_hours, t_start=t_start, t_end=t_end,
                    sigma_2d=sigma, sigma_3d=vol_sigma)
                print(f"  Unified vmax = {fvmax:.2f} m/s")
            grp = f"_{group}" if group else ""
            out_png = (outdir
                       / (f"budget_composite_wp_na_{ref}"
                          f"_T{t_start:+d}to{t_end:+d}{grp}.png"))
            mc_env_wp = load_mc_envelope_2d(
                basin="WP", reference=ref,
                composites_dir=composites_directory, out_tag=group or "")
            mc_env_na = load_mc_envelope_2d(
                basin="NA", reference=ref,
                composites_dir=composites_directory, out_tag=group or "")
            if mc_env_wp is not None or mc_env_na is not None:
                tag_wp = (f" WP n_iter={mc_env_wp['n_iter']}"
                          if mc_env_wp is not None else " WP env=missing")
                tag_na = (f" NA n_iter={mc_env_na['n_iter']}"
                          if mc_env_na is not None else " NA env=missing")
                print(f"  MC 2-D envelope hatching enabled "
                      f"({tag_wp}, {tag_na})")
            else:
                print("  MC 2-D envelope NOT FOUND for WP/NA - falling back "
                      "to per-pixel baseline-anomaly t-test")
            plot_budget_wp_na_fig5(
                data_wp=data_wp, budget_wp=budget_wp,
                data_na=data_na, budget_na=budget_na,
                reference=ref,
                t_start=t_start,
                t_end=t_end,
                sigma=sigma,
                sigma_3d=vol_sigma,
                fixed_vmax=fvmax,
                rel_lat_ymin=rel_lat_ymin,
                budget_rates_ms_day=use_rates,
                mc_env_left=mc_env_wp,
                mc_env_right=mc_env_na,
                out_path=out_png,
            )
        print("\nDone!", flush=True)
        return

    for ref in refs:
        loaded = {}
        budgets = {}
        for b in basin:
            print(f"\n{'=' * 60}")
            print(f"  Budget composite: {ref}, {b}")
            print(f"{'=' * 60}")
            data = load_composite(basin=b, reference=ref,
                                  composites_dir=composites_directory,
                                  group=group, supp_dir=supp_directory)
            if data is None:
                continue
            budget = compute_budget(data=data)
            if budget is None:
                print("  Missing budget variables, skipping.")
                continue
            loaded[b] = data
            budgets[b] = budget

        fvmax = fixed_vmax
        if fvmax is None and unified and len(budgets) > 1:
            lag_hours = list(loaded.values())[0]["_lag_hours"]
            fvmax = _compute_joint_vmax(
                data_list=list(loaded.values()),
                budget_list=list(budgets.values()),
                lag_hours=lag_hours, t_start=t_start, t_end=t_end,
                sigma_2d=sigma, sigma_3d=vol_sigma)
            print(f"  Unified vmax = {fvmax:.2f} m/s")

        for b in loaded:
            plot_budget_composite(
                data=loaded[b], budget=budgets[b], basin=b, reference=ref,
                outdir=outdir,
                t_start=t_start, t_end=t_end, sigma=sigma,
                sigma_3d=vol_sigma,
                fixed_vmax=fvmax, group=group)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    typer.run(main)
