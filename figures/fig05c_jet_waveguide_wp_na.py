"""Paper Fig. 6 (revised manuscript): storm-relative jet, waveguide and
carrying capacity for WP and NA recurving TCs.

Two panels (WP, NA) in recurvature-relative coordinates, all fields
averaged over the T_0d .. T_+6d window (same window as the budget maps):

  shading        ERA5 250-hPa zonal wind (``era5_u250_1deg`` composite)
  black contours jet carrying capacity F_c (m^2 s^-2)
  green dashed   10-km QGPV (1e-4, 1.5e-4, 2e-4 s^-1)
  thick black    composite-mean TC track, lime cross = recurvature point
  magenta box    downstream-waveguide averaging box used for F_c^down in
                 Table 1: rel. lon [0, +40], rel. lat [+5, +20]

The u250 composite is produced with the composite engine, e.g.::

    python -m downstream_et_lwa.composites.run_composites \
        --variables era5_u250_1deg era5_v250_1deg --basins WP NA \
        --reference recurvature --output-dir <u250-directory>

Coastlines are drawn relative to the mean recurvature position of each
basin only to give a sense of geographic scale.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import netCDF4 as nc
import numpy as np
import typer
from typing_extensions import Annotated
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

import downstream_et_lwa.plotting.budget_maps as budget_maps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)

# Downstream-waveguide box of Table 1 (build_table_cases FC_DLAT/FC_DLON)
FC_BOX_LON = (0.0, 40.0)
FC_BOX_LAT = (5.0, 20.0)
QGPV_LEVELS = [1e-4, 1.5e-4, 2e-4]
FC_LEVELS = [10, 20, 30, 40, 50, 60]
U_LEVELS = np.arange(10, 50.1, 5.0)
REL_LAT_MIN = -10.0   # the jet/waveguide sits poleward of the storm


def _load_u250(*, basin: str, reference: str, u250_dir: Path,
               lag_hours: np.ndarray, t_start: int, t_end: int) -> np.ndarray:
    path = Path(u250_dir) / f"composite_2d_{reference}_{basin}.nc"
    with nc.Dataset(path) as ds:
        u = np.array(ds["era5_u250_1deg_mean"][:])
        lag = np.array(ds["lag_hours"][:])
    if not np.array_equal(lag, lag_hours):
        raise RuntimeError(
            f"lag axis mismatch between budget and u250 composites ({path})")
    return budget_maps.time_average(field_3d=u, lag_hours=lag_hours,
                                    t_start=t_start, t_end=t_end)


def _box_mean(*, field: np.ndarray, lon: np.ndarray, lat: np.ndarray,
              abs_lat0: float) -> float:
    """cos(lat)-weighted mean inside the Table-1 downstream box."""
    lo_m = (lon >= FC_BOX_LON[0]) & (lon <= FC_BOX_LON[1])
    la_m = (lat >= FC_BOX_LAT[0]) & (lat <= FC_BOX_LAT[1])
    sub = field[np.ix_(la_m, lo_m)]
    w = np.cos(np.deg2rad(abs_lat0 + lat[la_m]))[:, None] * np.ones_like(sub)
    ok = np.isfinite(sub)
    return float(np.sum(sub[ok] * w[ok]) / np.sum(w[ok]))


def _draw_panel(*, ax, data, u_f, fc_f, qgpv_f, title, coast_segs,
                t_start, t_end, ylabel):
    lon, lat = data["_lon"], data["_lat"]
    lag_hours = data["_lag_hours"]
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_under("white")
    cf = ax.contourf(lon, lat, u_f, levels=U_LEVELS, cmap=cmap, extend="both")
    if coast_segs is not None:
        for rl, ra in coast_segs:
            in_box = ((rl > lon[0] - 5) & (rl < lon[-1] + 5)
                      & (ra > lat[0] - 5) & (ra < lat[-1] + 5))
            if in_box.sum() > 1:
                ax.plot(rl[in_box], ra[in_box], color="0.35", lw=0.7,
                        alpha=0.8, zorder=2)
    cs = ax.contour(lon, lat, fc_f, levels=FC_LEVELS, colors="k",
                    linewidths=1.1, zorder=3)
    ax.clabel(cs, fmt="%d", fontsize=8, inline=True)
    if qgpv_f is not None and np.any(np.isfinite(qgpv_f)):
        qs = gaussian_filter(np.where(np.isfinite(qgpv_f), qgpv_f, 0.0),
                             sigma=(1.5, 1.5))
        ax.contour(lon, lat, qs, levels=QGPV_LEVELS, colors="green",
                   linewidths=1.0, linestyles="--", alpha=0.9, zorder=3)
    mrl = data.get("_mean_rel_lat")
    mrn = data.get("_mean_rel_lon")
    if mrl is not None:
        m = ((lag_hours >= t_start - 24) & (lag_hours <= t_end + 24)
             & np.isfinite(mrl) & np.isfinite(mrn))
        ax.plot(mrn[m], mrl[m], "k-", lw=2.5, zorder=4)
        lag0 = int(np.argmin(np.abs(lag_hours)))
        ax.plot(mrn[lag0], mrl[lag0], "+", color="lime", ms=14, mew=2.5,
                zorder=10)
    ax.add_patch(Rectangle((FC_BOX_LON[0], FC_BOX_LAT[0]),
                           FC_BOX_LON[1] - FC_BOX_LON[0],
                           FC_BOX_LAT[1] - FC_BOX_LAT[0],
                           fill=False, ec="magenta", lw=1.8, ls="--",
                           zorder=5))
    y0 = max(float(lat[0]), REL_LAT_MIN)
    ax.set_xlim(lon[0], lon[-1])
    ax.set_ylim(y0, lat[-1])
    ax.set_box_aspect((lat[-1] - y0) / (lon[-1] - lon[0]))
    ax.axhline(0, color="k", lw=0.3, ls=":")
    ax.axvline(0, color="k", lw=0.3, ls=":")
    ax.set_title(title, fontsize=11, pad=3)
    ax.set_xlabel("rel. lon (\N{DEGREE SIGN})", fontsize=10)
    if ylabel:
        ax.set_ylabel("rel. lat (\N{DEGREE SIGN})", fontsize=10)
    ax.tick_params(labelsize=9)
    return cf


def main(
        composites_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_<reference>_<basin>.nc "
                 "(must contain era5_cc_Fc and era5_qgpv_10km)")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        u250_directory: Annotated[Optional[Path], typer.Option(
            help="Directory with the era5_u250_1deg composites (default: "
                 "--composites-directory)")] = None,
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {reference} is substituted")]
        = "fig5c_jet_waveguide_fc_wp_na_{reference}.png",
        reference: Annotated[str, typer.Option(
            help="recurvature or et")] = "recurvature",
        t_start: Annotated[int, typer.Option()] = 0,
        t_end: Annotated[int, typer.Option()] = 144,
        sigma: Annotated[float, typer.Option(
            help="2-D Gaussian sigma (deg) applied to the plotted u250/F_c "
                 "fields")] = 1.5,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if reference not in ("recurvature", "et"):
        raise typer.BadParameter("reference must be recurvature or et")
    u250_dir = (Path(u250_directory) if u250_directory is not None
                else Path(composites_directory))
    fig_dir = Path(output_directory)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    cf = None
    for ax, basin, letter in zip(axes, ("WP", "NA"), ("a", "b")):
        data = budget_maps.load_composite(
            basin=basin, reference=reference,
            composites_dir=composites_directory)
        if data is None:
            raise SystemExit(f"Missing composite for {basin}.")
        lag = data["_lag_hours"]
        u_f = budget_maps.smooth(
            field=_load_u250(basin=basin, reference=reference,
                             u250_dir=u250_dir, lag_hours=lag,
                             t_start=t_start, t_end=t_end),
            sigma=sigma)
        fc_f = budget_maps.smooth(
            field=budget_maps.time_average(
                field_3d=data["era5_cc_Fc"], lag_hours=lag,
                t_start=t_start, t_end=t_end),
            sigma=sigma)
        qgpv_f = budget_maps.time_average(
            field_3d=data["era5_qgpv_10km"], lag_hours=lag,
            t_start=t_start, t_end=t_end)
        coast = budget_maps._get_coastlines_shifted(
            mean_lat=data["_mean_abs_lat"], mean_lon=data["_mean_abs_lon"])
        fc_box = _box_mean(field=fc_f, lon=data["_lon"], lat=data["_lat"],
                           abs_lat0=data["_mean_abs_lat"])
        u_box = _box_mean(field=u_f, lon=data["_lon"], lat=data["_lat"],
                          abs_lat0=data["_mean_abs_lat"])
        n = int(data["_n_storms"])
        mlon = data["_mean_abs_lon"] % 360.0
        lon_txt = (f"{mlon:.0f}\N{DEGREE SIGN}E" if mlon <= 180
                   else f"{360 - mlon:.0f}\N{DEGREE SIGN}W")
        title = (f"({letter}) {basin}  (N={n}; mean recurvature "
                 f"{data['_mean_abs_lat']:.0f}\N{DEGREE SIGN}N, {lon_txt})")
        cf = _draw_panel(ax=ax, data=data, u_f=u_f, fc_f=fc_f, qgpv_f=qgpv_f,
                         title=title, coast_segs=coast,
                         t_start=t_start, t_end=t_end,
                         ylabel=(basin == "WP"))
        print(f"{basin}: N={n}, box-mean Fc={fc_box:.2f}, "
              f"box-mean u250={u_box:.2f}, "
              f"mean recurv lat={data['_mean_abs_lat']:.2f}", flush=True)
    cb = fig.colorbar(cf, ax=axes.tolist(), orientation="horizontal",
                      fraction=0.06, pad=0.2, shrink=0.45, aspect=40)
    cb.set_label("250-hPa zonal wind (m s$^{-1}$), T$_0$\N{EN DASH}T$_{+6d}$ mean",
                 fontsize=10)
    cb.ax.tick_params(labelsize=9)
    out = fig_dir / figure_name.format(reference=reference)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    typer.run(main)
