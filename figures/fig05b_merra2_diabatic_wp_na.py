"""Paper Fig. 5b: WP | NA storm-relative MERRA-2 diabatic source
decomposition (DTDTMST, DTDTRAD, DTDTANA) plus the MERRA-2 LWA budget
residual, built from the supplemental merra2src composites."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.contour
import matplotlib.lines
import netCDF4 as nc
import numpy as np
import typer
from typing_extensions import Annotated

import downstream_et_lwa.plotting.budget_maps as budget_maps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)

BUDGET_LEVELS = np.array(
    [-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
)
BUDGET_CMAP = plt.get_cmap("RdBu_r")

MIN_TRACK_COUNT = 15


def _load_merra2_only(*, basin: str, reference: str,
                      supp_dir: Path) -> dict[str, Any]:
    path = Path(supp_dir) / f"composite_2d_{reference}_{basin}_merra2src.nc"
    if not path.exists():
        raise SystemExit(f"Missing supplemental composite: {path}")

    data: dict[str, Any] = {}
    with nc.Dataset(path, "r") as ds:
        budget_maps._read_composite_vars_into(ds=ds, data=data)
        data["_lat"] = np.array(ds["rel_lat"][:])
        data["_lon"] = np.array(ds["rel_lon"][:])
        data["_lag_hours"] = np.array(ds["lag_hours"][:])
        data["_n_storms"] = ds.dimensions["storm"].size

        if "track_rel_lat" in ds.variables:
            trl = ds["track_rel_lat"][:]
            trn = ds["track_rel_lon"][:]
            valid_per_lag = np.sum(np.isfinite(trl), axis=0)
            mean_rl = np.where(valid_per_lag >= MIN_TRACK_COUNT,
                               np.nanmean(trl, axis=0), np.nan)
            mean_rn = np.where(valid_per_lag >= MIN_TRACK_COUNT,
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


def _compute_merra2_budget(*, data: dict[str, Any]) -> dict[str, Any]:
    lwa = data.get("merra2_lwa")
    if lwa is None:
        raise SystemExit("merra2_lwa missing in supplemental composite")

    lag_hours = data["_lag_hours"]
    tendency = budget_maps._tendency_lwa_per_day(lwa=lwa,
                                                 lag_hours=lag_hours)

    tI = data.get("merra2_budget_termI")
    tII = data.get("merra2_budget_termII")
    tIII = data.get("merra2_budget_termIII")

    def _to_per_day(x: np.ndarray | None) -> np.ndarray:
        return (x * budget_maps._SEC_PER_DAY if x is not None
                else np.full_like(lwa, np.nan))

    termI = _to_per_day(tI)
    termII = _to_per_day(tII)
    termIII = _to_per_day(tIII)

    residual = tendency - termI - termII - termIII

    mst = _to_per_day(data.get("merra2_heat_fortran_DTDTMST"))
    rad = _to_per_day(data.get("merra2_heat_fortran_DTDTRAD"))
    ana = _to_per_day(data.get("merra2_heat_fortran_DTDTANA"))

    out: dict[str, Any] = {
        "lwa": lwa,
        "tendency": tendency,
        "termI": termI,
        "termII": termII,
        "termIII": termIII,
        "residual": residual,
        "merra2_mst": mst,
        "merra2_rad": rad,
        "merra2_ana": ana,
    }

    for nc_key, sig_key in [
        ("merra2_heat_fortran_DTDTMST", "merra2_mst_sig"),
        ("merra2_heat_fortran_DTDTRAD", "merra2_rad_sig"),
        ("merra2_heat_fortran_DTDTANA", "merra2_ana_sig"),
    ]:
        sumsq = data.get(f"{nc_key}__sumsq")
        cnt = data.get(f"{nc_key}__count_field")
        mean = data.get(nc_key)
        if sumsq is not None and cnt is not None and mean is not None:
            out[sig_key] = budget_maps._sig_mask_ttest(
                mean_3d=mean, sumsq_3d=sumsq, count_3d=cnt)
        else:
            out[sig_key] = None

    lwa_sumsq = data.get("merra2_lwa__sumsq")
    lwa_cnt = data.get("merra2_lwa__count_field")
    sig_res = None
    if lwa_sumsq is not None and lwa_cnt is not None:
        lwa_var, _, _ = budget_maps._get_var_from_sumsq(
            mean_3d=lwa, sumsq_3d=lwa_sumsq, count_3d=lwa_cnt)
        res_var = lwa_var.copy()
        res_n = lwa_cnt.astype(np.float64).copy()
        for nc_key in ("merra2_budget_termI", "merra2_budget_termII",
                       "merra2_budget_termIII"):
            sq = data.get(f"{nc_key}__sumsq")
            ct = data.get(f"{nc_key}__count_field")
            mn = data.get(nc_key)
            if sq is not None and ct is not None and mn is not None:
                term_var, _, _ = budget_maps._get_var_from_sumsq(
                    mean_3d=mn, sumsq_3d=sq, count_3d=ct)
                term_var = term_var * (budget_maps._SEC_PER_DAY ** 2)
                res_var = res_var + np.where(np.isfinite(term_var),
                                             term_var, 0.0)
                res_n = np.minimum(res_n, ct.astype(np.float64))
        sig_res = budget_maps._sig_from_var(
            mean_field=residual, var_field=res_var, n_field=res_n)
    out["residual_sig"] = sig_res

    return out


def _baseline_anomaly_field(*, field_3d: np.ndarray,
                            lag_hours: np.ndarray,
                            tavg: Any,
                            baseline_lag: Tuple[float, float]) -> np.ndarray:
    m = tavg(field_3d)
    base = budget_maps._pre_event_baseline(
        field_3d=field_3d, lag_hours=lag_hours,
        baseline_start=baseline_lag[0], baseline_end=baseline_lag[1])
    if base is None:
        return np.asarray(m)
    return np.asarray(m - base)


def _basin_block(*, reference: str, basin: str, supp_dir: Path,
                 t_start: int, t_end: int,
                 sigma_2d: float,
                 sigma_3d: Tuple[float, float, float],
                 baseline_lag: Tuple[float, float] = (-48.0, -12.0)
                 ) -> dict[str, Any]:
    data = _load_merra2_only(basin=basin, reference=reference,
                             supp_dir=supp_dir)
    budget = _compute_merra2_budget(data=data)

    lag_hours = data["_lag_hours"]
    lat = data["_lat"]
    lon = data["_lon"]
    n = data["_n_storms"]

    tavg = budget_maps._make_tavg(lag_hours=lag_hours, t_start=t_start,
                                  t_end=t_end, sigma_3d=sigma_3d,
                                  sigma_2d=sigma_2d)

    mst_f = _baseline_anomaly_field(
        field_3d=budget["merra2_mst"], lag_hours=lag_hours, tavg=tavg,
        baseline_lag=baseline_lag)
    rad_f = _baseline_anomaly_field(
        field_3d=budget["merra2_rad"], lag_hours=lag_hours, tavg=tavg,
        baseline_lag=baseline_lag)
    ana_f = _baseline_anomaly_field(
        field_3d=budget["merra2_ana"], lag_hours=lag_hours, tavg=tavg,
        baseline_lag=baseline_lag)
    res_f = _baseline_anomaly_field(
        field_3d=budget["residual"], lag_hours=lag_hours, tavg=tavg,
        baseline_lag=baseline_lag)

    sig_mst = budget_maps.time_average_sig(
        sig_bool_3d=budget.get("merra2_mst_sig"), lag_hours=lag_hours,
        t_start=t_start, t_end=t_end)
    sig_rad = budget_maps.time_average_sig(
        sig_bool_3d=budget.get("merra2_rad_sig"), lag_hours=lag_hours,
        t_start=t_start, t_end=t_end)
    sig_ana = budget_maps.time_average_sig(
        sig_bool_3d=budget.get("merra2_ana_sig"), lag_hours=lag_hours,
        t_start=t_start, t_end=t_end)
    sig_res = budget_maps.time_average_sig(
        sig_bool_3d=budget.get("residual_sig"), lag_hours=lag_hours,
        t_start=t_start, t_end=t_end)

    coast_segs = None
    if "_mean_abs_lat" in data:
        coast_segs = budget_maps._get_coastlines_shifted(
            mean_lat=data["_mean_abs_lat"], mean_lon=data["_mean_abs_lon"])

    mean_rlat = data.get("_mean_rel_lat", np.full(len(lag_hours), np.nan))
    mean_rlon = data.get("_mean_rel_lon", np.full(len(lag_hours), np.nan))
    track_mask = (lag_hours >= t_start - 24) & (lag_hours <= t_end + 24)
    track_valid = np.isfinite(mean_rlat) & np.isfinite(mean_rlon) & track_mask
    lag0 = int(np.argmin(np.abs(lag_hours)))

    return dict(
        basin=basin, n=n,
        lat=lat, lon=lon,
        coast_segs=coast_segs,
        mst_f=mst_f, rad_f=rad_f, ana_f=ana_f, res_f=res_f,
        sig_mst=sig_mst, sig_rad=sig_rad, sig_ana=sig_ana, sig_res=sig_res,
        mean_rlat=mean_rlat, mean_rlon=mean_rlon,
        track_valid=track_valid, lag0=lag0,
    )


def _draw_panel(*, ax: matplotlib.axes.Axes, panel: dict[str, Any],
                field_key: str, sig_key: str, title: str,
                ylabel: bool, xlabel: bool,
                rel_lat_ymin: float) -> matplotlib.contour.QuadContourSet:
    return budget_maps._fig5_draw_panel(
        ax=ax,
        lon=panel["lon"], lat=panel["lat"], field=panel[field_key],
        title=title,
        levels=BUDGET_LEVELS,
        cmap=BUDGET_CMAP,
        coast_segs=panel["coast_segs"],
        qgpv_f=None,
        sig_mask=panel.get(sig_key),
        mean_rlon=panel["mean_rlon"], mean_rlat=panel["mean_rlat"],
        track_valid=panel["track_valid"], lag0=panel["lag0"],
        ylabel=ylabel, xlabel=xlabel,
        rel_lat_ymin=rel_lat_ymin,
    )


def main(
        merra2_supplement_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_<reference>_<basin>"
                 "_merra2src.nc supplemental composites")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {reference} is substituted")]
        = "fig05b_merra2_diabatic_sources_wp_na_{reference}.png",
        reference: Annotated[str, typer.Option(
            help="recurvature or et")] = "recurvature",
        t_start: Annotated[int, typer.Option()] = 0,
        t_end: Annotated[int, typer.Option()] = 144,
        sigma: Annotated[float, typer.Option(
            help="Optional extra isotropic 2-D smooth after time "
                 "mean")] = 0.0,
        volume_sigma: Annotated[Optional[list[float]], typer.Option(
            help="3-D sigma (rel_lat, rel_lon, lag) before time mean "
                 "(3 floats; default 0 0 0)")] = None,
        rel_lat_ymin: Annotated[float, typer.Option()] = -10.0,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if reference not in ("recurvature", "et"):
        raise typer.BadParameter("reference must be recurvature or et")
    vol = volume_sigma if volume_sigma is not None else [0.0, 0.0, 0.0]
    if len(vol) != 3:
        raise typer.BadParameter("--volume-sigma needs exactly 3 floats")
    sigma_3d = (float(vol[0]), float(vol[1]), float(vol[2]))

    fig_dir = Path(output_directory)
    fig_dir.mkdir(parents=True, exist_ok=True)

    panel_wp = _basin_block(
        reference=reference, basin="WP",
        supp_dir=merra2_supplement_directory,
        t_start=t_start, t_end=t_end,
        sigma_2d=sigma, sigma_3d=sigma_3d,
    )
    panel_na = _basin_block(
        reference=reference, basin="NA",
        supp_dir=merra2_supplement_directory,
        t_start=t_start, t_end=t_end,
        sigma_2d=sigma, sigma_3d=sigma_3d,
    )

    rows = [
        ("mst_f", "sig_mst",
         "({letter}) MERRA-2 latent heating (DTDTMST)"),
        ("rad_f", "sig_rad",
         "({letter}) MERRA-2 radiation (DTDTRAD)"),
        ("ana_f", "sig_ana",
         "({letter}) MERRA-2 analysis increment (DTDTANA)"),
        ("res_f", "sig_res",
         r"({letter}) MERRA-2 LWA budget residual "
         r"$\partial A/\partial t - (\mathrm{{I+II+III}})$"),
    ]

    fig = plt.figure(figsize=(12.8, 11.6))
    gs = fig.add_gridspec(
        len(rows), 3,
        width_ratios=[1.0, 1.0, 0.05],
        hspace=0.32, wspace=0.18,
        left=0.06, right=0.95, top=0.96, bottom=0.07,
    )

    fig.text(0.295, 0.985,
             f"WP  (N = {panel_wp['n']})",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.685, 0.985,
             f"NA  (N = {panel_na['n']})",
             ha="center", va="top", fontsize=14, fontweight="bold")

    letters_wp = ["a", "c", "e", "g"]
    letters_na = ["b", "d", "f", "h"]

    for i, (fkey, skey, title_tmpl) in enumerate(rows):
        ax_wp = fig.add_subplot(gs[i, 0])
        ax_na = fig.add_subplot(gs[i, 1])
        _draw_panel(
            ax=ax_wp, panel=panel_wp, field_key=fkey, sig_key=skey,
            title=title_tmpl.format(letter=letters_wp[i]),
            ylabel=True,
            xlabel=(i == len(rows) - 1),
            rel_lat_ymin=rel_lat_ymin,
        )
        _draw_panel(
            ax=ax_na, panel=panel_na, field_key=fkey, sig_key=skey,
            title=title_tmpl.format(letter=letters_na[i]),
            ylabel=False,
            xlabel=(i == len(rows) - 1),
            rel_lat_ymin=rel_lat_ymin,
        )

    cax = fig.add_subplot(gs[:, 2])
    norm = matplotlib.colors.BoundaryNorm(BUDGET_LEVELS,
                                          ncolors=BUDGET_CMAP.N)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=BUDGET_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ticks=BUDGET_LEVELS, extend="both")
    cb.ax.tick_params(labelsize=10)
    cb.set_label(r"LWA source / residual (m s$^{-1}$ day$^{-1}$)",
                 fontsize=11.5)

    handles = [
        matplotlib.lines.Line2D(
            [0], [0], color="k", lw=2.5, label="Mean recurving track"),
        matplotlib.lines.Line2D(
            [0], [0], marker="+", color="lime", lw=0, ms=10, mew=2.0,
            label="Storm position at $T_0$"),
        matplotlib.lines.Line2D(
            [0], [0], color="0.3", lw=0.8,
            label="Coastlines (storm-relative)"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=3,
               frameon=False, fontsize=11)

    out = fig_dir / figure_name.format(reference=reference)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    typer.run(main)
