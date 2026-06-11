"""IBTrACS + MPAS current/future track maps (Quinting & Jones 2016 style):
gray individual tracks, month-colored reference markers, gold mean track."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import cartopy.crs as ccrs
import cartopy.feature
import matplotlib
import matplotlib.axes
import matplotlib.lines
import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.composites.mpas_composites as mpas_composites
import downstream_et_lwa.tracks as tracks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)

BASIN_EXTENT = {
    "WP": [100, 180, 5, 55],
    "NA": [-100, -10, 10, 55],
}

GRAY_TRACK_COLOR = "0.38"
GRAY_TRACK_LW = 0.6
GRAY_TRACK_ALPHA = 0.5

MEAN_STROKE_COLOR = "#0d0d0d"
MEAN_STROKE_LW = 6.0
MEAN_FILL_COLOR = "#f5c200"
MEAN_LINE_LW = 4.0
MEAN_STAR_COLOR = "#f5c200"
MEAN_STAR_MEC = "k"
MEAN_STAR_MSW = 1.0
MEAN_STAR_MS = 20

MONTH_HEX = {
    12: "#0d3b66",
    1: "#1f618d",
    2: "#4a8fc4",
    3: "#0b5345",
    4: "#1d8348",
    5: "#52c878",
    6: "#4a0e0e",
    7: "#9a1212",
    8: "#c41e1e",
    9: "#4a0072",
    10: "#6a1b9a",
    11: "#9575cd",
}
MONTH_ABBREV = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

SEASONS_CAPTION = (
    "Recurvature month by season: "
    "DJF (blues), MAM (greens), JJA (red), SON (violet, distinct from JJA)."
)


def reference_month_from_row(*, storm: pd.Series,
                             reference: str = "recurvature") -> int:
    col = "et_time" if reference == "et" else "recurv_time"
    try:
        t = storm[col]
    except (TypeError, KeyError):
        t = np.nan
    if t is None or pd.isna(t):
        return 1
    m = int(pd.Timestamp(t).month)
    if m < 1 or m > 12:
        return 1
    return m


def recurvature_month_from_row(*, storm: pd.Series) -> int:
    return reference_month_from_row(storm=storm, reference="recurvature")


def month_color(*, month: int) -> str:
    m = int(month)
    if not (1 <= m <= 12):
        m = 1
    return MONTH_HEX.get(m, "#333333")


def _normalize_lons(*, lons: Any, basin: str) -> np.ndarray:
    arr = np.array(lons, dtype=float)
    if basin == "NA":
        arr = np.where(arr > 180, arr - 360, arr)
    else:
        arr = arr % 360
    return arr


def plot_basin_panel(
        *,
        ax: Any,
        storms_df: pd.DataFrame,
        full_tracks: dict[str, pd.DataFrame],
        basin: str,
        title: str,
        fontsize: float = 12.0,
        reference: str = "recurvature",
) -> None:
    if reference not in ("recurvature", "et"):
        raise ValueError("reference must be 'recurvature' or 'et'")
    if basin not in ("WP", "NA"):
        raise ValueError("basin must be WP or NA")
    ext = BASIN_EXTENT[basin]
    ax.set_extent(ext, crs=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, facecolor="#e8e4e0",
                   edgecolor="none")
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.35, color="0.45")
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.25, alpha=0.35, color="gray",
        linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    for _, storm in storms_df.iterrows():
        sid = storm["storm_id"]
        if sid not in full_tracks:
            continue
        track = full_tracks[sid]
        lats = track["lat"].values
        lons = _normalize_lons(lons=track["lon"].values, basin=basin)
        valid = np.isfinite(lats) & np.isfinite(lons)
        if valid.sum() < 2:
            continue
        ax.plot(
            lons[valid], lats[valid],
            color=GRAY_TRACK_COLOR, linewidth=GRAY_TRACK_LW,
            alpha=GRAY_TRACK_ALPHA,
            solid_capstyle="round", transform=ccrs.PlateCarree(), zorder=2,
        )

    lat_col = "et_lat" if reference == "et" else "recurv_lat"
    lon_col = "et_lon" if reference == "et" else "recurv_lon"
    for _, storm in storms_df.iterrows():
        rlat = float(storm[lat_col])
        rlon = float(storm[lon_col])
        if not np.isfinite(rlat) or not np.isfinite(rlon):
            continue
        rlon = _normalize_lons(lons=[rlon], basin=basin)[0]
        m = reference_month_from_row(storm=storm, reference=reference)
        color = month_color(month=m)
        ax.plot(
            rlon, rlat, "o", color=color, ms=4.5, alpha=0.9,
            markeredgecolor="white", markeredgewidth=0.4,
            transform=ccrs.PlateCarree(), zorder=5,
        )

    all_lats, all_lons = [], []
    for _, storm in storms_df.iterrows():
        sid = storm["storm_id"]
        if sid not in full_tracks:
            continue
        lats, lons, _, _ = tracks.interpolate_track_to_lags(
            track_df=full_tracks[sid], reference=reference)
        all_lats.append(lats)
        all_lons.append(_normalize_lons(lons=lons, basin=basin))

    if all_lats:
        mean_lat = np.nanmean(all_lats, axis=0)
        mean_lon = np.nanmean(all_lons, axis=0)
        mask = ((composite_config.LAG_HOURS >= -48)
                & (composite_config.LAG_HOURS <= 96))
        valid = np.isfinite(mean_lat) & np.isfinite(mean_lon) & mask
        mx = mean_lon[valid]
        my = mean_lat[valid]
        ax.plot(
            mx, my, color=MEAN_STROKE_COLOR, linewidth=MEAN_STROKE_LW,
            solid_capstyle="round", transform=ccrs.PlateCarree(), zorder=7,
        )
        ax.plot(
            mx, my, color=MEAN_FILL_COLOR, linewidth=MEAN_LINE_LW,
            solid_capstyle="round", transform=ccrs.PlateCarree(), zorder=8,
        )
        lag0 = int(np.argmin(np.abs(composite_config.LAG_HOURS)))
        ax.plot(
            mean_lon[lag0], mean_lat[lag0], "*", color=MEAN_STAR_COLOR,
            ms=MEAN_STAR_MS, markeredgecolor=MEAN_STAR_MEC,
            markeredgewidth=MEAN_STAR_MSW,
            transform=ccrs.PlateCarree(), zorder=10,
        )

    n = len(storms_df)
    ax.set_title(
        f"{title} (n = {n})",
        fontsize=fontsize, fontweight="bold", loc="left",
    )


def legend_handles_months_and_mean() -> list[matplotlib.lines.Line2D]:
    handles = []
    for mi in range(1, 13):
        c = month_color(month=mi)
        handles.append(
            matplotlib.lines.Line2D(
                [], [], marker="o", color="none",
                markerfacecolor=c, markeredgecolor="0.3",
                markeredgewidth=0.3,
                ms=6, label=MONTH_ABBREV[mi - 1],
            )
        )
    handles.append(
        matplotlib.lines.Line2D(
            [], [], color=MEAN_FILL_COLOR, lw=3.2, solid_capstyle="round",
            label="Mean track",
        )
    )
    return handles


def _plot_panels(*, axes: np.ndarray, panels: Sequence[tuple],
                 fontsize: float = 10.5) -> None:
    bas = ["WP", "NA"]
    bnames = ["WNP", "NA"]
    for row, col0, df, trk, dlab, letters, ref in panels:
        for c, (b, bn) in enumerate(zip(bas, bnames)):
            plot_basin_panel(
                ax=axes[row, col0 + c],
                storms_df=df[df["basin"] == b],
                full_tracks=trk,
                basin=b,
                title=f"({letters[c]}) {bn} \N{EM DASH} {dlab}",
                fontsize=fontsize,
                reference=ref,
            )


def plot_track_map_combined(
        *,
        tracks_directory: Path,
        output_path: Path,
        layout: str = "3x4",
        all_tracks_root: Path | None = None,
        et_track_dir: Path | None = None,
        use_all_tracks: bool = True,
        build_et_ibtracs: bool = False,
        ibtracs_path: Path | None = None,
) -> Path:
    bkw: dict[str, Any] = {}
    if use_all_tracks and all_tracks_root is not None:
        bkw["all_tracks_root"] = all_tracks_root
    if not use_all_tracks:
        bkw["et_track_dir"] = et_track_dir

    ib, ib_trk = tracks.load_track_database(tracks_directory=tracks_directory)
    cur, cur_t = mpas_composites.build_mpas_tracks(scenario="current", **bkw)
    fut, fut_t = mpas_composites.build_mpas_tracks(scenario="future", **bkw)

    yr = (f"{composite_config.YEAR_START}"
          f"\N{EN DASH}{composite_config.YEAR_END}")

    if layout == "3x2":
        fig, axes = plt.subplots(
            3, 2, figsize=(16, 14.5),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        panels = [
            (0, 0, ib, ib_trk, f"IBTrACS {yr} (recurving)", ["a", "b"],
             "recurvature"),
            (1, 0, cur, cur_t, "MPAS current (recurving)", ["c", "d"],
             "recurvature"),
            (2, 0, fut, fut_t, "MPAS future (recurving)", ["e", "f"],
             "recurvature"),
        ]
    else:
        ib_et, ib_et_trk = tracks.load_et_nh_track_database(
            tracks_directory=tracks_directory,
            build_if_missing=build_et_ibtracs,
            ibtracs_path=ibtracs_path,
        )
        bkw_et = {**bkw, "et_only": True}
        cur_et, cur_et_t = mpas_composites.build_mpas_tracks(
            scenario="current", **bkw_et)
        fut_et, fut_et_t = mpas_composites.build_mpas_tracks(
            scenario="future", **bkw_et)
        fig, axes = plt.subplots(
            3, 4, figsize=(32, 14.5),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        panels = [
            (0, 0, ib, ib_trk, f"IBTrACS {yr} (recurving)", ["a", "b"],
             "recurvature"),
            (0, 2, ib_et, ib_et_trk, f"IBTrACS {yr} (ET)", ["g", "h"], "et"),
            (1, 0, cur, cur_t, "MPAS current (recurving)", ["c", "d"],
             "recurvature"),
            (1, 2, cur_et, cur_et_t, "MPAS current (ET)", ["i", "j"], "et"),
            (2, 0, fut, fut_t, "MPAS future (recurving)", ["e", "f"],
             "recurvature"),
            (2, 2, fut_et, fut_et_t, "MPAS future (ET)", ["k", "l"], "et"),
        ]

    _plot_panels(axes=axes, panels=panels)

    handles = legend_handles_months_and_mean()
    fig.legend(
        handles=handles, loc="lower center", ncol=7, fontsize=8.5,
        framealpha=0.95, edgecolor="0.7", bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _LOG.info("Saved: %s", out)
    return out


def main(
        tracks_directory: Annotated[Path, typer.Option(
            help="Directory with recurving_nh_tracks.csv and individual/")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        layout: Annotated[str, typer.Option(help="3x2 or 3x4")] = "3x4",
        all_tracks_root: Annotated[Optional[Path], typer.Option(
            help="Root of extracted MPAS ettrack .txt files")] = None,
        et_track_dir: Annotated[Optional[Path], typer.Option(
            help="MPAS ET-track .dat directory (used with --no-all-tracks)"
        )] = None,
        no_all_tracks: Annotated[bool, typer.Option(
            "--no-all-tracks", help="Use .dat trajectories instead of the "
            "extracted all-tracks root")] = False,
        build_et_ibtracs: Annotated[bool, typer.Option(
            help="Build et_nh_tracks.csv from IBTrACS if missing")] = False,
        ibtracs_path: Annotated[Optional[Path], typer.Option(
            help="IBTrACS CSV (needed with --build-et-ibtracs)")] = None,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if layout not in ("3x2", "3x4"):
        raise typer.BadParameter("layout must be 3x2 or 3x4")
    output_path = Path(output_directory) / (
        f"track_map_ibtracs_and_mpas_{layout}.png")
    out = plot_track_map_combined(
        tracks_directory=tracks_directory,
        output_path=output_path,
        layout=layout,
        all_tracks_root=all_tracks_root,
        et_track_dir=et_track_dir,
        use_all_tracks=not no_all_tracks,
        build_et_ibtracs=build_et_ibtracs,
        ibtracs_path=ibtracs_path,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    typer.run(main)
