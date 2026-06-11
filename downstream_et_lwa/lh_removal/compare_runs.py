"""Per-event diagnostic figure comparing observed, CTRL, NoLH, and NoR+ LWA runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors
import netCDF4 as nc
import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

import downstream_et_lwa.lh_removal.advection_kernel as advection_kernel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)

LAT_MIN = 20.0
LAT_MAX = 80.0


def merid_mean(*, arr_t_lat_lon: np.ndarray, lat_min: float = LAT_MIN,
               lat_max: float = LAT_MAX) -> np.ndarray:
    sel = (advection_kernel.LATS >= lat_min) & (advection_kernel.LATS <= lat_max)
    w = advection_kernel.COSPHI[sel]
    return (arr_t_lat_lon[:, sel, :] * w[None, :, None]).sum(axis=1) / w.sum()


def domain_mean(*, arr_t_lat_lon: np.ndarray, lat_min: float = LAT_MIN,
                lat_max: float = LAT_MAX) -> np.ndarray:
    sel = (advection_kernel.LATS >= lat_min) & (advection_kernel.LATS <= lat_max)
    w = advection_kernel.COSPHI[sel][:, None]
    a = arr_t_lat_lon[:, sel, :]
    return (a * w[None, :, :]).sum(axis=(1, 2)) / (w.sum() * a.shape[2])


def load_run(*, storm_id: str, mode: str, run_directory: Path) -> dict[str, Any]:
    safe_mode = mode.replace("+", "Pos")
    p = run_directory / f"{storm_id}_{safe_mode}.nc"
    with nc.Dataset(p, "r") as d:
        out = dict(
            t=np.array(d["time"][:], dtype=np.float64),
            A=np.array(d["A"][:], dtype=np.float64),
            A_obs=np.array(d["A_obs"][:], dtype=np.float64),
            attrs={k: getattr(d, k) for k in d.ncattrs()},
        )
    return out


def hovmoller_panel(*, ax: matplotlib.axes.Axes, lon: np.ndarray,
                    lag_d: np.ndarray, field: np.ndarray, levels: np.ndarray,
                    cmap: matplotlib.colors.Colormap, title: str,
                    cbar_label: str, star_lon: float | None = None,
                    star_lag: float = 0.0) -> Any:
    norm = matplotlib.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="both")
    im = ax.contourf(lon, lag_d, field, levels=levels, cmap=cmap,
                     norm=norm, extend="both")
    ax.axhline(0.0, color="k", lw=0.5, ls=":")
    if star_lon is not None:
        ax.plot([star_lon], [star_lag], marker="*", ms=14,
                mec="k", mfc="yellow", mew=1.0, zorder=5)
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("lag (days)")
    ax.set_title(title, fontsize=11)
    return im


def make_figure(*, storm_id: str, tracks_file: Path, run_directory: Path,
                output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    ctrl = load_run(storm_id=storm_id, mode="CTRL", run_directory=run_directory)
    nolh = load_run(storm_id=storm_id, mode="NoLH", run_directory=run_directory)
    norpos = load_run(storm_id=storm_id, mode="NoR+", run_directory=run_directory)
    for r in (nolh, norpos):
        if not np.array_equal(ctrl["t"], r["t"]):
            raise RuntimeError("Run time axes differ")
    times_s = ctrl["t"]
    A_obs = ctrl["A_obs"]
    A_ctrl = ctrl["A"]
    A_nolh = nolh["A"]
    A_norp = norpos["A"]

    t0_iso = ctrl["attrs"]["src_window_start"]
    recurv_iso = ctrl["attrs"]["src_recurv_time"]
    t0 = pd.Timestamp(t0_iso)
    t_recurv = pd.Timestamp(recurv_iso)
    lag_d = (times_s + (t0 - t_recurv).total_seconds()) / 86400.0
    lon = advection_kernel.LONS

    s_obs = merid_mean(arr_t_lat_lon=A_obs)
    s_ctrl = merid_mean(arr_t_lat_lon=A_ctrl)
    s_nolh = merid_mean(arr_t_lat_lon=A_nolh)
    s_norp = merid_mean(arr_t_lat_lon=A_norp)
    d_lh = s_ctrl - s_nolh
    d_rpos = s_ctrl - s_norp

    fig = plt.figure(figsize=(16.0, 11.5))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32,
                          left=0.06, right=0.98, top=0.94, bottom=0.06)

    cmap_lwa = plt.get_cmap("YlOrBr")
    cmap_diff = plt.get_cmap("RdBu_r")

    lvl_lwa = np.linspace(0.0, 60.0, 13)
    lvl_close = np.array([-12, -9, -6, -3, -1, 0, 1, 3, 6, 9, 12], dtype=float)
    lvl_dlh = np.array([-8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8], dtype=float)
    lvl_drpos = np.array([-12, -9, -6, -3, -1, 0, 1, 3, 6, 9, 12], dtype=float)

    df = pd.read_csv(
        tracks_file,
        parse_dates=["recurv_time", "et_time"],
        keep_default_na=False, na_values=[""])
    row = df[df["storm_id"] == storm_id].iloc[0]
    star_lon = float(row["recurv_lon"]) % 360
    name = row["name"] if isinstance(row["name"], str) else ""

    ax = fig.add_subplot(gs[0, 0])
    im = hovmoller_panel(ax=ax, lon=lon, lag_d=lag_d, field=s_obs,
                         levels=lvl_lwa, cmap=cmap_lwa,
                         title="(a) Observed LWA  20-80N mean",
                         cbar_label="m s-1", star_lon=star_lon)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"LWA (m s$^{-1}$)")
    ax = fig.add_subplot(gs[0, 1])
    im = hovmoller_panel(ax=ax, lon=lon, lag_d=lag_d, field=s_ctrl,
                         levels=lvl_lwa, cmap=cmap_lwa,
                         title="(b) CTRL reconstruction (all sources ON)",
                         cbar_label="m s-1", star_lon=star_lon)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"LWA (m s$^{-1}$)")
    ax = fig.add_subplot(gs[0, 2])
    im = hovmoller_panel(ax=ax, lon=lon, lag_d=lag_d, field=s_ctrl - s_obs,
                         levels=lvl_close, cmap=cmap_diff,
                         title="(c) CTRL - Obs   (closure error)",
                         cbar_label="m s-1", star_lon=star_lon)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\Delta$ LWA (m s$^{-1}$)")

    R_deg = ctrl["attrs"].get("tc_radius_deg", None)
    R_tag = (f"local R={float(R_deg):g}\N{DEGREE SIGN}" if R_deg is not None
             else "local")
    ax = fig.add_subplot(gs[1, 0])
    im = hovmoller_panel(ax=ax, lon=lon, lag_d=lag_d, field=s_norp,
                         levels=lvl_lwa, cmap=cmap_lwa,
                         title=f"(d) NoR$^+$ reconstruction "
                               f"(LH+S$_{{other}}$=0, {R_tag})",
                         cbar_label="m s-1", star_lon=star_lon)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"LWA (m s$^{-1}$)")
    ax = fig.add_subplot(gs[1, 1])
    im = hovmoller_panel(ax=ax, lon=lon, lag_d=lag_d, field=d_rpos,
                         levels=lvl_drpos, cmap=cmap_diff,
                         title=f"(e) $\\Delta$A = CTRL - NoR$^+$   "
                               f"(TC R$^+$ impact, {R_tag})",
                         cbar_label="m s-1", star_lon=star_lon)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\Delta$ LWA (m s$^{-1}$)")
    ax = fig.add_subplot(gs[1, 2])
    dm_obs = domain_mean(arr_t_lat_lon=A_obs)
    dm_ctrl = domain_mean(arr_t_lat_lon=A_ctrl)
    dm_nolh = domain_mean(arr_t_lat_lon=A_nolh)
    dm_norp = domain_mean(arr_t_lat_lon=A_norp)
    ax.plot(lag_d, dm_obs, color="k", lw=2.0, label="Obs")
    ax.plot(lag_d, dm_ctrl, color="C0", lw=1.8, label="CTRL")
    ax.plot(lag_d, dm_nolh, color="C3", lw=1.8, label="NoLH")
    ax.plot(lag_d, dm_norp, color="C2", lw=1.8, label="NoR$^+$")
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.axvline(0.0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("lag (days)")
    ax.set_ylabel(r"LWA 20-80N mean (m s$^{-1}$)")
    ax.set_title("(f) Domain-mean LWA")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 0])
    im = hovmoller_panel(ax=ax, lon=lon, lag_d=lag_d, field=s_nolh,
                         levels=lvl_lwa, cmap=cmap_lwa,
                         title=f"(g) NoLH reconstruction (LH=0, {R_tag})",
                         cbar_label="m s-1", star_lon=star_lon)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"LWA (m s$^{-1}$)")
    ax = fig.add_subplot(gs[2, 1])
    im = hovmoller_panel(ax=ax, lon=lon, lag_d=lag_d, field=d_lh,
                         levels=lvl_dlh, cmap=cmap_diff,
                         title=f"(h) $\\Delta$A = CTRL - NoLH   "
                               f"(TC LH impact, {R_tag})",
                         cbar_label="m s-1", star_lon=star_lon)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\Delta$ LWA (m s$^{-1}$)")
    ax = fig.add_subplot(gs[2, 2])
    s_diff_R = (d_rpos).mean(axis=1)
    s_diff_L = (d_lh).mean(axis=1)
    ax.plot(lag_d, s_diff_R, color="C2", lw=2.0, label="$\\Delta$A(R$^+$)")
    ax.plot(lag_d, s_diff_L, color="C3", lw=2.0, label="$\\Delta$A(LH)")
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.axvline(0.0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("lag (days)")
    ax.set_ylabel(r"zonal-mean $\Delta$A, 20-80N (m s$^{-1}$)")
    ax.set_title("(i) Domain $\\Delta$A from each removal")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    title = (f"2-D LH/R$^+$ removal experiment - storm {storm_id} ({name}, basin "
             f"{row['basin']}, recurvature {recurv_iso[:10]})")
    fig.suptitle(title, fontsize=13)

    out = output_directory / f"event_{storm_id}_LHRpos_removal_diagnostic.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main(
    storm_id: Annotated[str, typer.Option()],
    tracks_file: Annotated[Path, typer.Option(help="Recurving NH tracks CSV.")],
    run_directory: Annotated[Path, typer.Option(
        help="Directory with <storm_id>_<MODE>.nc run files.")],
    output_directory: Annotated[Path, typer.Option(
        help="Directory for the diagnostic figure.")],
) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    out = make_figure(storm_id=storm_id, tracks_file=tracks_file,
                      run_directory=run_directory,
                      output_directory=output_directory)
    print(out)


if __name__ == "__main__":
    typer.run(main)
