"""WP RW vs WP no-RW stratified composites of the LH / R+ removal experiment."""

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


def make_rw_strat_figure(*,
                         df_pop_wp: pd.DataFrame,
                         output_path: Path,
                         classification_file: Path,
                         strip_directory: Path,
                         climatology_file: Path,
                         tracks_file: Path,
                         individual_tracks_directory: Path,
                         cbar_abs_max: float = 5.0) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cls = pd.read_csv(classification_file)
    cls = cls[["storm_id", "wb_group"]].rename(columns={"wb_group": "rw_group"})
    df = df_pop_wp.merge(cls, on="storm_id", how="left")

    df_rw = df[df["rw_group"] == "rwcase"].copy()
    df_no = df[df["rw_group"] == "norwcase"].copy()
    _LOG.info("RW=%d storms; no-RW=%d storms (out of %d WP storms)",
              len(df_rw), len(df_no), len(df))

    if len(df_rw) == 0 or len(df_no) == 0:
        raise SystemExit("no RW or no-RW storms after merge.")

    _LOG.info("Loading JJASON LWA climatology")
    lwa_clim_1d = composite_figure.load_lwa_climatology(
        climatology_file=climatology_file)

    _LOG.info("Compositing CTRL")
    ctrl_rw, n_rw = composite_figure.composite_basin(
        df_basin=df_rw, var="A_CTRL", strip_directory=strip_directory)
    ctrl_no, n_no = composite_figure.composite_basin(
        df_basin=df_no, var="A_CTRL", strip_directory=strip_directory)
    _LOG.info("CTRL RW n=%d, no-RW n=%d", n_rw, n_no)

    _LOG.info("Compositing NoR+")
    nrp_rw, _ = composite_figure.composite_basin(
        df_basin=df_rw, var="A_NoRPos", strip_directory=strip_directory)
    nrp_no, _ = composite_figure.composite_basin(
        df_basin=df_no, var="A_NoRPos", strip_directory=strip_directory)

    _LOG.info("Compositing NoLH")
    nlh_rw, _ = composite_figure.composite_basin(
        df_basin=df_rw, var="A_NoLH", strip_directory=strip_directory)
    nlh_no, _ = composite_figure.composite_basin(
        df_basin=df_no, var="A_NoLH", strip_directory=strip_directory)

    ctrl_rw_anom = ctrl_rw - lwa_clim_1d[None, :]
    ctrl_no_anom = ctrl_no - lwa_clim_1d[None, :]
    dRp_rw = ctrl_rw - nrp_rw
    dRp_no = ctrl_no - nrp_no
    dLH_rw = ctrl_rw - nlh_rw
    dLH_no = ctrl_no - nlh_no

    ctrl_rw_s = composite_figure._safe_smooth(field=ctrl_rw_anom)
    ctrl_no_s = composite_figure._safe_smooth(field=ctrl_no_anom)
    dRp_rw_s = composite_figure._safe_smooth(field=dRp_rw)
    dRp_no_s = composite_figure._safe_smooth(field=dRp_no)
    dLH_rw_s = composite_figure._safe_smooth(field=dLH_rw)
    dLH_no_s = composite_figure._safe_smooth(field=dLH_no)

    _LOG.info("Loading WP track for mean-track overlay")
    full_tracks: dict[str, pd.DataFrame] | None
    try:
        _, full_tracks = composite_helpers.load_track_database(
            tracks_file=tracks_file,
            individual_tracks_directory=individual_tracks_directory)
    except Exception:
        _LOG.exception("Track DB load failed; track skipped")
        full_tracks = None

    track_lag_rw, track_lon_rw = composite_helpers._mean_track(
        storms_df=df_rw, basin="WP", reference="recurvature",
        full_tracks=full_tracks)
    track_lag_no, track_lon_no = composite_helpers._mean_track(
        storms_df=df_no, basin="WP", reference="recurvature",
        full_tracks=full_tracks)

    def _recurv_stats(*, df_b: pd.DataFrame
                      ) -> tuple[tuple[float, float] | None, float | None]:
        if "recurv_lon" not in df_b.columns or len(df_b) == 0:
            return None, None
        v = df_b["recurv_lon"].to_numpy() % 360.0
        return ((float(np.nanmin(v)), float(np.nanmax(v))),
                float(np.nanmean(v)))

    lon_rng_rw, recurv_rw = _recurv_stats(df_b=df_rw)
    lon_rng_no, recurv_no = _recurv_stats(df_b=df_no)

    fig = plt.figure(figsize=(15.2, 16.8))
    gs = fig.add_gridspec(
        nrows=4, ncols=3,
        width_ratios=[1.0, 1.0, 0.08],
        height_ratios=[0.34, 1.85, 1.85, 1.85],
        hspace=0.34, wspace=0.20,
        left=0.06, right=0.94, top=0.98, bottom=0.06,
    )

    shared_levels = np.linspace(-float(cbar_abs_max), float(cbar_abs_max),
                                9, dtype=float)

    common_kw: dict[str, Any] = dict(
        y_days=composite_figure.LAGS_D,
        y_lim=(-2, 7),
        show_lon_extent_hline=True,
        significance=False,
        with_colorbar=False,
        tick_labelsize=14,
        title_fontsize=18,
    )

    def _block(*, map_row: int, hov_row: int,
               panels: tuple[np.ndarray, np.ndarray],
               titles: tuple[str, str]) -> None:
        if map_row == 0:
            for c in range(2):
                ax_m = fig.add_subplot(gs[map_row, c])
                composite_helpers._draw_minimap(ax=ax_m)
                ax_m.set_title(
                    "WP RW case (top quintile)" if c == 0
                    else "WP no-RW case (bottom quintile)",
                    fontsize=20, fontweight="bold", loc="center", pad=2.0,
                )
        ax_left = fig.add_subplot(gs[hov_row, 0])
        composite_helpers._hovmoller_panel(
            ax=ax_left, data=panels[0],
            mask_lo=np.zeros_like(panels[0]),
            mask_hi=np.zeros_like(panels[0]),
            levels=shared_levels, cmap=composite_helpers._BWOR_8,
            title=titles[0], cbar_label="",
            mean_track_lag=track_lag_rw, mean_track_lon=track_lon_rw,
            lon_range=lon_rng_rw,
            recurv_lon_mean=recurv_rw,
            show_xlabel=(hov_row == 5), show_ylabel=True,
            **common_kw,
        )
        ax_right = fig.add_subplot(gs[hov_row, 1])
        composite_helpers._hovmoller_panel(
            ax=ax_right, data=panels[1],
            mask_lo=np.zeros_like(panels[1]),
            mask_hi=np.zeros_like(panels[1]),
            levels=shared_levels, cmap=composite_helpers._BWOR_8,
            title=titles[1], cbar_label="",
            mean_track_lag=track_lag_no, mean_track_lon=track_lon_no,
            lon_range=lon_rng_no,
            recurv_lon_mean=recurv_no,
            show_xlabel=(hov_row == 5), show_ylabel=False,
            **common_kw,
        )

    _block(
        map_row=0, hov_row=1,
        panels=(ctrl_rw_s, ctrl_no_s),
        titles=("(a) WP RW - CTRL reconstruction LWA anomaly",
                "(b) WP no-RW - CTRL reconstruction LWA anomaly"),
    )
    _block(
        map_row=-1, hov_row=2,
        panels=(dRp_rw_s, dRp_no_s),
        titles=(
            "(c) WP RW - $\\Delta A$ = CTRL $-$ NoR$^+$ (local TC R$^+$)",
            "(d) WP no-RW - $\\Delta A$ = CTRL $-$ NoR$^+$ (local TC R$^+$)",
        ),
    )
    _block(
        map_row=-1, hov_row=3,
        panels=(dLH_rw_s, dLH_no_s),
        titles=(
            "(e) WP RW - $\\Delta A$ = CTRL $-$ NoLH (local TC LH)",
            "(f) WP no-RW - $\\Delta A$ = CTRL $-$ NoLH (local TC LH)",
        ),
    )

    cax = fig.add_subplot(gs[1:, 2])
    norm = matplotlib.colors.BoundaryNorm(shared_levels,
                                          ncolors=composite_helpers._BWOR_8.N)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=composite_helpers._BWOR_8)
    sm.set_array([])
    cb = fig.colorbar(
        sm, cax=cax, orientation="vertical",
        ticks=shared_levels, extend="both",
    )
    cb.set_label("LWA", fontsize=17, labelpad=8)
    cb.ax.tick_params(labelsize=14)

    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main(
    tracks_file: Annotated[Path, typer.Option(help="Recurving NH tracks CSV.")],
    individual_tracks_directory: Annotated[Path, typer.Option(
        help="Directory of per-storm track CSVs (<storm_id>.csv).")],
    strip_directory: Annotated[Path, typer.Option(
        help="Directory with strip_<storm_id>.nc files.")],
    climatology_file: Annotated[Path, typer.Option(
        help="JJASON cos-lat 20-80N LWA climatology NetCDF.")],
    classification_file: Annotated[Path, typer.Option(
        help="WP RW classification CSV (storm_id, wb_group).")],
    output_directory: Annotated[Path, typer.Option(
        help="Directory for the composite figure.")],
    figure_name: Annotated[str, typer.Option()] =
        "fig5_lh_Rpos_removal_composites_WP_RW_strat.png",
    year_start: Annotated[int, typer.Option()] = 2000,
    year_end: Annotated[int, typer.Option()] = 2022,
    cbar_max: Annotated[float, typer.Option()] = 5.0,
) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    df = pd.read_csv(tracks_file,
                     parse_dates=["recurv_time", "et_time"],
                     keep_default_na=False, na_values=[""])
    yr = df["recurv_time"].dt.year
    sel = ((yr >= year_start) & (yr <= year_end)
           & (df["basin"] == "WP"))
    df = df[sel].copy()
    have_strip = df["storm_id"].apply(
        lambda sid: (strip_directory / f"strip_{sid}.nc").exists()
    )
    df = df[have_strip].copy()
    print(f"[info] {len(df)} WP storms with strips "
          f"({year_start}-{year_end})")

    out = make_rw_strat_figure(
        df_pop_wp=df,
        output_path=output_directory / figure_name,
        classification_file=classification_file,
        strip_directory=strip_directory,
        climatology_file=climatology_file,
        tracks_file=tracks_file,
        individual_tracks_directory=individual_tracks_directory,
        cbar_abs_max=cbar_max,
    )
    print(out)


if __name__ == "__main__":
    typer.run(main)
