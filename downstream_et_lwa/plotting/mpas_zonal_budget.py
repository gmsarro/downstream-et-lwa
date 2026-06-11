"""MPAS current vs future zonal-mean LWA-budget Hovmoller figure (Fig. 10
layout): WP+NA pooled or single-basin blocks with anomaly significance."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.gridspec
import matplotlib.lines
import numpy as np
import scipy.ndimage
import typer
from typing_extensions import Annotated

import downstream_et_lwa.composites.mpas_composites as mpas_composites
import downstream_et_lwa.plotting.budget_hovmoller as budget_hovmoller
import downstream_et_lwa.plotting.budget_maps as budget_maps
import downstream_et_lwa.plotting.mpas_wpna_io as mpas_wpna_io
import downstream_et_lwa.plotting.qj_hovmoller as qj3

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LOG = logging.getLogger(__name__)

BUDGET_LEVELS = budget_hovmoller.BUDGET_LEVELS
HOVMOLLER_GAUSSIAN_SIGMA = budget_hovmoller.HOVMOLLER_GAUSSIAN_SIGMA
PANEL_CBAR = budget_hovmoller.PANEL_CBAR

_SEC_PER_DAY = 86400.0


def _meridional_mean_var(*, var_lat_lon_lag: np.ndarray,
                         lat_idx: Any) -> np.ndarray:
    return np.nanmean(var_lat_lon_lag[lat_idx, :, :], axis=0).T


def _zonal_mean_rate_cube(*, lwa: np.ndarray,
                          tI: np.ndarray | None,
                          tII: np.ndarray | None,
                          tIII: np.ndarray | None,
                          lag_hours: np.ndarray,
                          lh: np.ndarray | None = None,
                          rel_lat: np.ndarray | None = None,
                          rel_lat_band: tuple[float, float] | None = None,
                          subtract_pre_baseline: bool = True,
                          baseline_lag: tuple[float, float] = (-48.0, -12.0),
                          data: dict | None = None
                          ) -> tuple[dict[str, np.ndarray],
                                     dict[str, np.ndarray | None],
                                     dict[str, np.ndarray | None]]:
    tendency = budget_maps._tendency_lwa_per_day(lwa=lwa, lag_hours=lag_hours)
    a = tI * _SEC_PER_DAY if tI is not None else np.full_like(lwa, np.nan)
    b = tII * _SEC_PER_DAY if tII is not None else np.full_like(lwa, np.nan)
    c = tIII * _SEC_PER_DAY if tIII is not None else np.full_like(lwa, np.nan)
    residual = tendency - a - b - c
    lh_term = (lh * _SEC_PER_DAY if lh is not None
               else np.full_like(lwa, np.nan))

    lat_idx: Any
    if rel_lat is None:
        lat_idx = slice(None)
    else:
        lo, hi = (-10.0, 25.0) if rel_lat_band is None else rel_lat_band
        lat_idx = ((np.asarray(rel_lat, dtype=float) >= float(lo))
                   & (np.asarray(rel_lat, dtype=float) <= float(hi)))

    def zm(x: np.ndarray) -> np.ndarray:
        return np.nanmean(x[lat_idx, :, :], axis=0).T

    out = {
        "tendency": zm(tendency),
        "termI": zm(a),
        "termII": zm(b),
        "termIII": zm(c),
        "residual": zm(residual),
        "lh_lwa": zm(lh_term),
        "lwa": zm(lwa),
    }

    var_out: dict[str, np.ndarray | None] = {k: None for k in out}
    n_out: dict[str, np.ndarray | None] = {k: None for k in out}
    if data is not None:
        def _per_pixel_var(name: str) -> tuple[np.ndarray | None,
                                               np.ndarray | None]:
            mn = data.get(name)
            sq = data.get(f"{name}__sumsq")
            ct = data.get(f"{name}__count_field")
            if mn is None or sq is None or ct is None:
                return None, None
            v, n, _ = budget_maps._get_var_from_sumsq(
                mean_3d=mn.astype(np.float64),
                sumsq_3d=sq.astype(np.float64),
                count_3d=ct.astype(np.float64))
            return v, n

        v_lwa, n_lwa = _per_pixel_var("lwa")
        var_terms: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        for term_name, key in (("budget_termI", "termI"),
                               ("budget_termII", "termII"),
                               ("budget_termIII", "termIII")):
            v_t, n_t = _per_pixel_var(term_name)
            if v_t is not None:
                var_terms[key] = (v_t * (_SEC_PER_DAY ** 2), n_t)

        v_lh, n_lh = _per_pixel_var("mpas_lh_lwa")
        if v_lh is not None:
            var_terms["lh_lwa"] = (v_lh * (_SEC_PER_DAY ** 2), n_lh)

        if v_lwa is not None and len(lag_hours) >= 3:
            dt_h = float(lag_hours[1] - lag_hours[0])
            scale = (24.0 / (2.0 * dt_h)) ** 2
            v_tend = np.full_like(v_lwa, np.nan)
            v_tend[..., 1:-1] = scale * (v_lwa[..., 2:] + v_lwa[..., :-2])
            var_terms["tendency"] = (v_tend, n_lwa)

        if "tendency" in var_terms:
            v_t, n_t = var_terms["tendency"]
            var_res = np.copy(v_t)
            for k in ("termI", "termII", "termIII"):
                if k in var_terms:
                    var_res = var_res + var_terms[k][0]
            var_terms["residual"] = (var_res, n_t)

        for key, (v_field, n_field) in var_terms.items():
            var_out[key] = _meridional_mean_var(var_lat_lon_lag=v_field,
                                                lat_idx=lat_idx)
            assert n_field is not None
            n_out[key] = np.floor(
                np.nanmean(n_field[lat_idx, :, :], axis=0)).T.astype(np.int32)

    if subtract_pre_baseline:
        lag_arr = np.asarray(lag_hours, dtype=float)
        bm = ((lag_arr >= float(baseline_lag[0]))
              & (lag_arr <= float(baseline_lag[1])))
        if bm.any():
            for k, v in list(out.items()):
                if k == "lwa":
                    continue
                base = np.nanmean(v[bm, :], axis=0)
                out[k] = v - base[None, :]
    return out, var_out, n_out


def _title_reletter(*, letter: str, rest: str) -> str:
    return f"({letter}){rest}"


def _add_scenario_column_separator(
        *,
        fig: matplotlib.figure.Figure,
        data_axes_left: np.ndarray,
        data_axes_right: np.ndarray,
        n_data_rows: int = 3,
        outer_col_gap: tuple[float, float] | None = None,
) -> None:
    if outer_col_gap is not None:
        x0, x1 = float(outer_col_gap[0]), float(outer_col_gap[1])
        gap = x1 - x0
        x = 0.5 * (x0 + x1) if gap <= 1e-4 else x0 + 0.30 * gap
    else:
        p1 = data_axes_left[0, 1].get_position()
        p2 = data_axes_right[0, 0].get_position()
        gap = p2.x0 - p1.x1
        x = 0.5 * (p1.x1 + p2.x0) if gap <= 1e-4 else p1.x1 + 0.30 * gap
    y0 = min(
        data_axes_left[r, c].get_position().y0
        for r in range(n_data_rows) for c in range(2))
    y0 = min(
        y0,
        min(data_axes_right[r, c].get_position().y0
            for r in range(n_data_rows) for c in range(2)),
    )
    y1 = max(
        data_axes_left[r, c].get_position().y1
        for r in range(n_data_rows) for c in range(2))
    y1 = max(
        y1,
        max(data_axes_right[r, c].get_position().y1
            for r in range(n_data_rows) for c in range(2)),
    )
    fig.add_artist(
        matplotlib.lines.Line2D(
            [x, x], [y0, y1], transform=fig.transFigure,
            color="black", linewidth=1.8, zorder=200, clip_on=False))


def _load_wpna_zm(*, composite_dir: Path, scenario: str, reference: str,
                  basin: str = "WPNA"
                  ) -> tuple[dict[str, Any], dict[str, np.ndarray],
                             dict[str, np.ndarray | None],
                             dict[str, np.ndarray | None],
                             np.ndarray, np.ndarray]:
    if str(basin).upper() == "WPNA":
        data = mpas_wpna_io.load_mpas_wpna_pooled(
            composite_dir=composite_dir, reference=reference,
            scenario=scenario)
    else:
        path = Path(composite_dir) / (
            f"composite_2d_{reference}_mpas_{scenario}_{basin}.nc")
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing single-basin MPAS composite: {path}")
        data = mpas_composites.load_mpas_composite(
            path=path, prefix=f"mpas_{scenario}")
    lag = np.asarray(data["_lag_hours"])
    lwa = data["lwa"]
    rel_lat = np.asarray(data["_lat"], dtype=float)
    zm, var_zm, n_zm = _zonal_mean_rate_cube(
        lwa=lwa,
        tI=data.get("budget_termI"),
        tII=data.get("budget_termII"),
        tIII=data.get("budget_termIII"),
        lag_hours=lag,
        lh=data.get("mpas_lh_lwa"),
        rel_lat=rel_lat,
        rel_lat_band=(-10.0, 25.0),
        subtract_pre_baseline=True,
        data=data,
    )
    return data, zm, var_zm, n_zm, np.asarray(data["_lon"], dtype=float), lag


def _mean_track_relative(*, data: dict[str, Any], lag_h: np.ndarray,
                         lag_lim_days: tuple[float, float] | None = (-2.0, 4.0)
                         ) -> tuple[np.ndarray | None, np.ndarray | None]:
    rlat = data.get("_mean_rel_lat")
    rlon = data.get("_mean_rel_lon")
    if rlon is None:
        return None, None
    rlon = np.asarray(rlon, dtype=float)
    rlat_arr = np.asarray(rlat, dtype=float) if rlat is not None else None
    lag_days = np.asarray(lag_h, dtype=float) / 24.0
    ok = np.isfinite(rlon)
    if rlat_arr is not None:
        ok = ok & np.isfinite(rlat_arr)
    if lag_lim_days is not None:
        lo, hi = float(lag_lim_days[0]), float(lag_lim_days[1])
        ok = ok & (lag_days >= lo) & (lag_days <= hi)
    if not ok.any():
        return None, None
    return lag_days[ok], rlon[ok]


def _hovmoller_anomaly_sig(*, field_lon_lag: np.ndarray,
                           var_lon_lag: np.ndarray | None,
                           n_lon_lag: np.ndarray | None,
                           lag_hours: np.ndarray,
                           baseline_lag: tuple[float, float],
                           smooth_sigma: tuple[float, float],
                           alpha: float = 0.05) -> np.ndarray | None:
    if var_lon_lag is None or n_lon_lag is None:
        return None
    lag_arr = np.asarray(lag_hours, dtype=float)
    bm = ((lag_arr >= float(baseline_lag[0]))
          & (lag_arr <= float(baseline_lag[1])))
    if not bm.any():
        return None
    var_smooth = scipy.ndimage.gaussian_filter(
        np.where(np.isfinite(var_lon_lag), var_lon_lag, 0.0)
        .astype(np.float64),
        sigma=smooth_sigma,
    )
    base_var = np.nanmean(var_smooth[bm, :], axis=0)
    se_var = var_smooth + base_var[None, :]
    n_arr = np.asarray(n_lon_lag, dtype=np.float64)
    return budget_maps._sig_from_var(
        mean_field=np.asarray(field_lon_lag, dtype=np.float64),
        var_field=se_var,
        n_field=n_arr,
        alpha=alpha,
    )


def _render_block(
        *,
        fig: matplotlib.figure.Figure,
        outer: matplotlib.gridspec.SubplotSpec,
        zm_bundle: tuple,
        scenario_label: str,
        letter_index0: int,
        smooth_sigma: tuple[float, float],
        n_storms: int,
        is_left: bool,
        baseline_lag: tuple[float, float] = (-48.0, -12.0),
        x_lim: tuple[float, float] | None = None,
) -> tuple[np.ndarray, list]:
    data, zm, var_zm, n_zm, x_lon, lag_h = zm_bundle
    gs_data = outer.subgridspec(3, 2, hspace=0.34, wspace=0.18)

    data_ax = np.empty((3, 2), dtype=object)
    for r in range(3):
        data_ax[r, 0] = fig.add_subplot(gs_data[r, 0])
        data_ax[r, 1] = fig.add_subplot(gs_data[r, 1], sharey=data_ax[r, 0])
        data_ax[r, 1].tick_params(labelleft=False)

    panel_keys = ("tendency", "termI", "termII", "termIII", "residual",
                  "lh_lwa")
    cfg_title = {
        "tendency": r"$\partial A / \partial t$",
        "termI": r"Term I: $-\partial F_\lambda/\partial x$",
        "termII": r"Term II: meridional flux",
        "termIII": r"Term III",
        "residual": r"Residual (diabatic closure)",
        "lh_lwa": r"MPAS LH-LWA source",
    }
    letters = "abcdefghijklmnopqrstuvwxyz"
    li = letter_index0
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    Y = lag_h / 24.0
    y0 = float(np.min(Y))
    y1 = float(np.max(Y))

    track_lag, track_lon = _mean_track_relative(data=data, lag_h=lag_h)

    for (pr, pc), key in zip(positions, panel_keys):
        ax = data_ax[pr, pc]
        raw = zm[key]
        fld = scipy.ndimage.gaussian_filter(
            np.where(np.isfinite(raw), raw, 0.0).astype(np.float64),
            sigma=smooth_sigma,
        ).astype(np.float32)
        sig_mask = _hovmoller_anomaly_sig(
            field_lon_lag=fld,
            var_lon_lag=var_zm.get(key) if var_zm is not None else None,
            n_lon_lag=n_zm.get(key) if n_zm is not None else None,
            lag_hours=lag_h,
            baseline_lag=baseline_lag,
            smooth_sigma=smooth_sigma,
        )
        levels = BUDGET_LEVELS
        cmap = qj3._BWOR_8
        title = _title_reletter(
            letter=letters[li],
            rest=(
                f" MPAS {scenario_label} \N{EM DASH} {cfg_title[key]} "
                f"(N={n_storms})"
                if (pr == 0 and pc == 0)
                else f" MPAS {scenario_label} \N{EM DASH} {cfg_title[key]}"),
        )
        li += 1
        lo = np.full_like(fld, -1.0e6)
        hi = np.full_like(fld, 1.0e6)
        cbar_lbl = PANEL_CBAR.get(key, r"Anomaly (m s$^{-1}$ day$^{-1}$)")
        qj3._hovmoller_panel(
            ax=ax, data=fld, mask_lo=lo, mask_hi=hi,
            levels=levels, cmap=cmap,
            title=title,
            cbar_label=cbar_lbl,
            with_colorbar=False,
            title_fontsize=11,
            tick_labelsize=9,
            title_pad=7,
            show_xlabel=(pr == 2),
            show_ylabel=(pc == 0 and is_left),
            sig_field=None,
            sig_mask=sig_mask,
            significance=(sig_mask is not None),
            show_lon_extent_hline=False,
            x_lon=x_lon,
            y_days=Y,
            y_lim=(y0, y1),
            mean_track_lag=track_lag,
            mean_track_lon=track_lon,
            recurv_lon_mean=0.0,
        )
        if x_lim is not None:
            ax.set_xlim(*x_lim)

    fig.align_xlabels([data_ax[2, c] for c in range(2)])
    return data_ax, []


def plot_mpas_fig10_zonal_budget(
        *,
        composite_dir: Path,
        out_path: Path,
        reference: str = "recurvature",
        smooth_sigma: tuple[float, float] | None = None,
        basin: str = "WPNA",
) -> Path:
    if reference != "recurvature":
        raise ValueError(
            "MPAS composites use recurvature reference in this workflow.")
    sig = smooth_sigma or HOVMOLLER_GAUSSIAN_SIGMA

    zm_cur = _load_wpna_zm(composite_dir=composite_dir, scenario="current",
                           reference=reference, basin=basin)
    zm_fut = _load_wpna_zm(composite_dir=composite_dir, scenario="future",
                           reference=reference, basin=basin)
    data_cur = zm_cur[0]
    data_fut = zm_fut[0]
    n_cur = int(data_cur.get("_n_storms", 0))
    n_fut = int(data_fut.get("_n_storms", 0))

    x_lim = (-25.0, 125.0)

    fig = plt.figure(figsize=(20.0, 11.0))
    mega = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 1.0],
        wspace=0.10,
        left=0.06, right=0.985, top=0.97, bottom=0.13,
    )

    ax_cur, _ = _render_block(
        fig=fig, outer=mega[0, 0], zm_bundle=zm_cur,
        scenario_label="current",
        letter_index0=0, smooth_sigma=sig, n_storms=n_cur, is_left=True,
        x_lim=x_lim)

    ax_fut, _ = _render_block(
        fig=fig, outer=mega[0, 1], zm_bundle=zm_fut,
        scenario_label="future",
        letter_index0=6, smooth_sigma=sig, n_storms=n_fut, is_left=False,
        x_lim=x_lim)

    norm_b = matplotlib.colors.BoundaryNorm(BUDGET_LEVELS,
                                            ncolors=qj3._BWOR_8.N)
    sm_b = matplotlib.cm.ScalarMappable(norm=norm_b, cmap=qj3._BWOR_8)
    sm_b.set_array(np.array([0.0]))
    cax_b = fig.add_axes((0.12, 0.055, 0.76, 0.024))
    cb_b = fig.colorbar(
        sm_b, cax=cax_b, orientation="horizontal",
        ticks=BUDGET_LEVELS, extend="both",
    )
    cb_b.set_label(
        r"Anomaly (m s$^{-1}$ day$^{-1}$)",
        fontsize=13,
        labelpad=6,
    )
    cb_b.ax.tick_params(labelsize=11)

    fig.canvas.draw()
    pos_l = mega[0, 0].get_position(fig)
    pos_r = mega[0, 1].get_position(fig)
    _add_scenario_column_separator(
        fig=fig, data_axes_left=ax_cur, data_axes_right=ax_fut, n_data_rows=3,
        outer_col_gap=(pos_l.x1, pos_r.x0))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    _LOG.info("  wrote %s", out)
    return out


def main(
        composite_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_<reference>_mpas_<scenario>_"
                 "<basin>.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        basin: Annotated[str, typer.Option(
            help="WPNA, WP, or NA")] = "WPNA",
        reference: Annotated[str, typer.Option()] = "recurvature",
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if basin not in ("WPNA", "WP", "NA"):
        raise typer.BadParameter("basin must be WPNA, WP, or NA")
    out_path = Path(output_directory) / (
        f"fig10_mpas_lwa_budget_zonal_{basin.lower()}_{reference}.png")
    out = plot_mpas_fig10_zonal_budget(
        composite_dir=composite_directory,
        out_path=out_path,
        reference=reference,
        basin=basin,
    )
    print(f"  wrote {out}")


if __name__ == "__main__":
    typer.run(main)
