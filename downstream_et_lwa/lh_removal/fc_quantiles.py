"""LH / R+ removal composites stratified by downstream jet carrying capacity F_c."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.cm
import matplotlib.colors
import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

import downstream_et_lwa.lh_removal._composite_helpers as composite_helpers
import downstream_et_lwa.lh_removal.composite_figure as composite_figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)

DEFAULT_DLAT_LO = 0.0
DEFAULT_DLAT_HI = 25.0
DEFAULT_DLON_LO = 0.0
DEFAULT_DLON_HI = 25.0


def _load_fc(*, capacity_params_file: Path
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not capacity_params_file.exists():
        raise FileNotFoundError(
            f"Carrying-capacity NPZ not found: {capacity_params_file}")
    with np.load(capacity_params_file) as d:
        Fc = np.array(d["Fc"], dtype=np.float64)
        lat = np.array(d["latitude"], dtype=np.float64)
        lon = np.array(d["longitude"], dtype=np.float64)
    return Fc, lat, lon


def _box_mask(*, lat: np.ndarray, lon: np.ndarray,
              recurv_lat: float, recurv_lon: float,
              dlat_lo: float, dlat_hi: float,
              dlon_lo: float, dlon_hi: float
              ) -> tuple[np.ndarray, np.ndarray]:
    lat_mask = ((lat >= recurv_lat + dlat_lo)
                & (lat <= recurv_lat + dlat_hi))
    lo = (recurv_lon + dlon_lo) % 360.0
    hi = (recurv_lon + dlon_hi) % 360.0
    if hi > lo:
        lon_mask = (lon >= lo) & (lon <= hi)
    else:
        lon_mask = (lon >= lo) | (lon <= hi)
    return lat_mask, lon_mask


def fc_box_mean(*, Fc: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                month_idx: int, recurv_lat: float, recurv_lon: float,
                dlat_lo: float = DEFAULT_DLAT_LO,
                dlat_hi: float = DEFAULT_DLAT_HI,
                dlon_lo: float = DEFAULT_DLON_LO,
                dlon_hi: float = DEFAULT_DLON_HI) -> float:
    lat_mask, lon_mask = _box_mask(lat=lat, lon=lon,
                                   recurv_lat=recurv_lat,
                                   recurv_lon=recurv_lon,
                                   dlat_lo=dlat_lo, dlat_hi=dlat_hi,
                                   dlon_lo=dlon_lo, dlon_hi=dlon_hi)
    if not lat_mask.any() or not lon_mask.any():
        return float(np.nan)
    sub = Fc[month_idx][np.ix_(lat_mask, lon_mask)]
    sub_lat = lat[lat_mask]
    w = np.cos(np.deg2rad(sub_lat))[:, None]
    w = np.broadcast_to(w, sub.shape)
    finite = np.isfinite(sub)
    if not finite.any():
        return float(np.nan)
    num = np.nansum(np.where(finite, sub * w, 0.0))
    den = np.sum(np.where(finite, w, 0.0))
    return float(num / den) if den > 0 else float(np.nan)


def attach_fc(*, df: pd.DataFrame, capacity_params_file: Path,
              dlat_lo: float = DEFAULT_DLAT_LO,
              dlat_hi: float = DEFAULT_DLAT_HI,
              dlon_lo: float = DEFAULT_DLON_LO,
              dlon_hi: float = DEFAULT_DLON_HI) -> pd.DataFrame:
    Fc, lat, lon = _load_fc(capacity_params_file=capacity_params_file)
    out = df.copy()
    vals = []
    for _, r in out.iterrows():
        rt = pd.Timestamp(r["recurv_time"]).to_pydatetime()
        m = rt.month - 1
        v = fc_box_mean(Fc=Fc, lat=lat, lon=lon, month_idx=m,
                        recurv_lat=float(r["recurv_lat"]),
                        recurv_lon=float(r["recurv_lon"]),
                        dlat_lo=dlat_lo, dlat_hi=dlat_hi,
                        dlon_lo=dlon_lo, dlon_hi=dlon_hi)
        vals.append(v)
    out["Fc_box"] = vals
    return out


def split_by_quantile(*, df: pd.DataFrame, frac: float
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df.dropna(subset=["Fc_box"]).copy()
    if len(sub) == 0:
        return sub.iloc[:0], sub.iloc[:0]
    q_lo = float(np.quantile(sub["Fc_box"], frac))
    q_hi = float(np.quantile(sub["Fc_box"], 1.0 - frac))
    lo = sub[sub["Fc_box"] <= q_lo].copy()
    hi = sub[sub["Fc_box"] >= q_hi].copy()
    return lo, hi


def composite_delta_panels(*, df_sub: pd.DataFrame, strip_directory: Path
                           ) -> tuple[np.ndarray, np.ndarray, int]:
    ctrl, n = composite_figure.composite_basin(
        df_basin=df_sub, var="A_CTRL", strip_directory=strip_directory)
    nrp, _ = composite_figure.composite_basin(
        df_basin=df_sub, var="A_NoRPos", strip_directory=strip_directory)
    nlh, _ = composite_figure.composite_basin(
        df_basin=df_sub, var="A_NoLH", strip_directory=strip_directory)
    dRp = ctrl - nrp
    dLH = ctrl - nlh
    return (composite_figure._safe_smooth(field=dRp),
            composite_figure._safe_smooth(field=dLH), n)


def make_figure(*,
                df_pop: pd.DataFrame,
                output_path: Path,
                strip_directory: Path,
                capacity_params_file: Path,
                tracks_file: Path,
                individual_tracks_directory: Path,
                tc_radius_deg: float = 10.0,
                cbar_abs_max: float = 5.0,
                quantile_frac: float = 1.0 / 3.0,
                dlat_lo: float = DEFAULT_DLAT_LO,
                dlat_hi: float = DEFAULT_DLAT_HI,
                dlon_lo: float = DEFAULT_DLON_LO,
                dlon_hi: float = DEFAULT_DLON_HI,
                suptitle: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _LOG.info("Computing F_c box mean (dphi in [%g,%g], dlambda in [%g,%g]) "
              "for %d storms", dlat_lo, dlat_hi, dlon_lo, dlon_hi, len(df_pop))
    df_pop = attach_fc(df=df_pop, capacity_params_file=capacity_params_file,
                       dlat_lo=dlat_lo, dlat_hi=dlat_hi,
                       dlon_lo=dlon_lo, dlon_hi=dlon_hi)

    blocks: list[dict[str, Any]] = []
    for basin in ("WP", "NA"):
        df_b = df_pop[df_pop["basin"] == basin]
        lo, hi = split_by_quantile(df=df_b, frac=quantile_frac)
        blocks.append(dict(label="low F_c", basin=basin, df=lo))
        blocks.append(dict(label="high F_c", basin=basin, df=hi))

    _LOG.info("Compositing %d sub-populations (frac=%.3f)",
              len(blocks), quantile_frac)
    for p in blocks:
        dRp, dLH, n = composite_delta_panels(df_sub=p["df"],
                                             strip_directory=strip_directory)
        fc_mean = (float(p["df"]["Fc_box"].mean())
                   if len(p["df"]) else float("nan"))
        p.update(dict(dRp=dRp, dLH=dLH, n=n, fc_mean=fc_mean))
        _LOG.info("%2s %-8s N = %3d mean F_c = %6.2f m^2 s^-2",
                  p["basin"], p["label"], n, fc_mean)

    _LOG.info("Loading full track database for mean-track overlays")
    full_tracks: dict[str, pd.DataFrame] | None
    try:
        _, full_tracks = composite_helpers.load_track_database(
            tracks_file=tracks_file,
            individual_tracks_directory=individual_tracks_directory)
    except Exception:
        _LOG.exception("Could not load track database; track skipped")
        full_tracks = None

    for p in blocks:
        if len(p["df"]) == 0:
            p.update(dict(track_lag=None, track_lon=None,
                          lon_range=None, recurv_mean=None))
            continue
        tlag, tlon = composite_helpers._mean_track(
            storms_df=p["df"], basin=p["basin"], reference="recurvature",
            full_tracks=full_tracks)
        lon_vals = p["df"]["recurv_lon"].to_numpy() % 360.0
        p.update(dict(
            track_lag=tlag, track_lon=tlon,
            lon_range=(float(np.nanmin(lon_vals)),
                       float(np.nanmax(lon_vals))),
            recurv_mean=float(np.nanmean(lon_vals)),
        ))

    fig = plt.figure(figsize=(20.0, 11.0))
    gs = fig.add_gridspec(
        nrows=4, ncols=4,
        height_ratios=[0.45, 3.0, 3.0, 0.32],
        hspace=0.22, wspace=0.18,
        left=0.05, right=0.985, top=0.96, bottom=0.06,
    )

    shared_levels = np.linspace(-float(cbar_abs_max), float(cbar_abs_max),
                                9, dtype=float)

    quantile_pct = int(round(quantile_frac * 100))
    row_letters = ("abcd", "efgh")
    row_var = ("dRp", "dLH")
    row_titles = (
        r"$\Delta$A = CTRL $-$ NoR$^+$  (local TC R$^+$, R = "
        f"{tc_radius_deg:g}\N{DEGREE SIGN})",
        r"$\Delta$A = CTRL $-$ NoLH  (local TC LH, R = "
        f"{tc_radius_deg:g}\N{DEGREE SIGN})",
    )

    for c, p in enumerate(blocks):
        ax_m = fig.add_subplot(gs[0, c])
        composite_helpers._draw_minimap(ax=ax_m)
        ax_m.set_title(
            f"{p['basin']} recurving TCs ({p['label']})\n"
            f"N={p['n']}, F$_c$ mean={p['fc_mean']:.1f} m$^2$ s$^{{-2}}$",
            fontsize=10, loc="center", pad=2.0,
        )

    for r, (var, letters, rtitle) in enumerate(
            zip(row_var, row_letters, row_titles)):
        for c, p in enumerate(blocks):
            ax_h = fig.add_subplot(gs[1 + r, c])
            field = p[var]
            title = (f"({letters[c]}) {p['basin']}, {p['label']} - {rtitle}"
                     if c == 0 else
                     f"({letters[c]}) {p['basin']}, {p['label']}")
            composite_helpers._hovmoller_panel(
                ax=ax_h, data=field,
                mask_lo=np.zeros_like(field),
                mask_hi=np.zeros_like(field),
                levels=shared_levels, cmap=composite_helpers._BWOR_8,
                title=title, cbar_label="",
                mean_track_lag=p["track_lag"],
                mean_track_lon=p["track_lon"],
                lon_range=p["lon_range"],
                recurv_lon_mean=p["recurv_mean"],
                y_days=composite_figure.LAGS_D,
                y_lim=(-1.5, 8.5),
                show_lon_extent_hline=True,
                significance=False,
                with_colorbar=False,
                tick_labelsize=10,
                title_fontsize=10,
                show_xlabel=(r == 1),
                show_ylabel=(c == 0),
            )

    cb_outer = fig.add_subplot(gs[3, :])
    cb_outer.set_axis_off()
    bb = cb_outer.get_position()
    cax = fig.add_axes((
        bb.x0 + 0.18 * bb.width,
        bb.y0 + 0.20 * bb.height,
        0.64 * bb.width,
        0.55 * bb.height,
    ))
    norm = matplotlib.colors.BoundaryNorm(shared_levels,
                                          ncolors=composite_helpers._BWOR_8.N)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=composite_helpers._BWOR_8)
    sm.set_array([])
    cb = fig.colorbar(
        sm, cax=cax, orientation="horizontal",
        ticks=shared_levels, extend="both",
    )
    cb.set_label(
        r"$\Delta\langle$LWA$\rangle$  (20-80N, m s$^{-1}$); "
        f"shared scale \N{PLUS-MINUS SIGN}{cbar_abs_max:g}",
        fontsize=11,
    )
    cb.ax.tick_params(labelsize=9)

    if suptitle:
        yr_lo = int(df_pop["recurv_time"].dt.year.min())
        yr_hi = int(df_pop["recurv_time"].dt.year.max())
        fig.suptitle(
            "Diabatic-source removal stratified by downstream jet "
            "carrying capacity F$_c$    "
            f"(box: dphi in [{dlat_lo:g},{dlat_hi:g}], "
            f"dlambda in [{dlon_lo:g},{dlon_hi:g}]; cos-lat-weighted; "
            f"quantile = {quantile_pct}/{100 - quantile_pct})\n"
            f"WP and NA recurving TCs, {yr_lo}-{yr_hi}  "
            f"(absolute longitude; BARO_N ERA5; LN24 + NHN22; "
            f"local TC mask R = {tc_radius_deg:g}\N{DEGREE SIGN}; "
            f"Gaussian sigma=(6 h, 2.5\N{DEGREE SIGN}))",
            fontsize=11, y=0.985,
        )

    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main(
    tracks_file: Annotated[Path, typer.Option(help="Recurving NH tracks CSV.")],
    individual_tracks_directory: Annotated[Path, typer.Option(
        help="Directory of per-storm track CSVs (<storm_id>.csv).")],
    strip_directory: Annotated[Path, typer.Option(
        help="Directory with strip_<storm_id>.nc files.")],
    capacity_params_file: Annotated[Path, typer.Option(
        help="era5_monthly_clim_params.npz with Fc/latitude/longitude.")],
    output_directory: Annotated[Path, typer.Option(
        help="Directory for the composite figure.")],
    figure_name: Annotated[str, typer.Option()] =
        "fig5b_lh_Rpos_removal_fc_quantile_wp_na_recurvature.png",
    year_start: Annotated[int, typer.Option()] = 2000,
    year_end: Annotated[int, typer.Option()] = 2022,
    basins: Annotated[list[str], typer.Option()] = ["WP", "NA"],
    tc_radius_deg: Annotated[float, typer.Option(
        help="Figure title only; the actual radius is whatever was used "
             "when building the catalogs.")] = 10.0,
    cbar_max: Annotated[float, typer.Option()] = 5.0,
    quantile_frac: Annotated[float, typer.Option(
        help="Tail fraction per group (0.333 = terciles, "
             "0.5 = median split).")] = 1.0 / 3.0,
    dlat_lo: Annotated[float, typer.Option(
        help="Storm-relative phi lower bound (deg).")] = DEFAULT_DLAT_LO,
    dlat_hi: Annotated[float, typer.Option(
        help="Storm-relative phi upper bound (deg).")] = DEFAULT_DLAT_HI,
    dlon_lo: Annotated[float, typer.Option(
        help="Storm-relative lambda lower bound (deg).")] = DEFAULT_DLON_LO,
    dlon_hi: Annotated[float, typer.Option(
        help="Storm-relative lambda upper bound (deg).")] = DEFAULT_DLON_HI,
    suptitle: Annotated[bool, typer.Option(
        help="Draw figure suptitle (off by default).")] = False,
) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    df = pd.read_csv(tracks_file,
                     parse_dates=["recurv_time", "et_time"],
                     keep_default_na=False, na_values=[""])
    yr = df["recurv_time"].dt.year
    sel = ((yr >= year_start) & (yr <= year_end)
           & df["basin"].isin(basins))
    df = df[sel].copy()
    have_strip = df["storm_id"].apply(
        lambda sid: (strip_directory / f"strip_{sid}.nc").exists()
    )
    df = df[have_strip].copy()
    print(f"[info] {len(df)} storms with strips in basins {basins} "
          f"({year_start}-{year_end})")

    out = make_figure(
        df_pop=df,
        output_path=output_directory / figure_name,
        strip_directory=strip_directory,
        capacity_params_file=capacity_params_file,
        tracks_file=tracks_file,
        individual_tracks_directory=individual_tracks_directory,
        tc_radius_deg=tc_radius_deg,
        cbar_abs_max=cbar_max,
        quantile_frac=quantile_frac,
        dlat_lo=dlat_lo, dlat_hi=dlat_hi,
        dlon_lo=dlon_lo, dlon_hi=dlon_hi,
        suptitle=suptitle,
    )
    print(out)


if __name__ == "__main__":
    typer.run(main)
