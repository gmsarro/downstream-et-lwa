"""Paper Fig. 5: WP | NA storm-relative budget-composite maps (NASA Fig. 5
layout), rate-based budget panels with Welch diff-test hatching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

import downstream_et_lwa.plotting.budget_maps as budget_maps

_LOG = logging.getLogger(__name__)


def main(
        composites_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_<reference>_<basin>.nc")],
        merra2_supplement_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_*_merra2src.nc supplementary "
                 "composites")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {reference} and {group} are "
                 "substituted")] = "fig5_budget_composite_wp_na_"
                                   "{reference}{group}.png",
        reference: Annotated[str, typer.Option(
            help="recurvature, et, or both")] = "recurvature",
        t_start: Annotated[int, typer.Option()] = 0,
        t_end: Annotated[int, typer.Option()] = 144,
        rel_lat_ymin: Annotated[float, typer.Option(
            help="Southern limit of rel. latitude axis (deg)")] = -10.0,
        sigma: Annotated[float, typer.Option(
            help="Extra per-storm isotropic 2D Gaussian sigma (deg)")] = 2.0,
        volume_sigma: Annotated[Optional[list[float]], typer.Option(
            help="3-D Gaussian sigma (rel_lat, rel_lon, lag) before time "
                 "mean (3 floats; default 0 0 0)")] = None,
        integrated_budget_maps: Annotated[bool, typer.Option(
            help="Use lag-integrated budget anomalies (m/s) with joint "
                 "vmax instead of rates (m/s/day)")] = False,
        fixed_vmax: Annotated[Optional[float], typer.Option(
            help="With --integrated-budget-maps: override joint RdBu "
                 "limit (m/s)")] = None,
        group: Annotated[Optional[str], typer.Option(
            help="Stratification group label")] = None,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    vol_sigma = (tuple(volume_sigma) if volume_sigma is not None
                 else (0.0, 0.0, 0.0))
    if len(vol_sigma) != 3:
        raise typer.BadParameter("--volume-sigma needs exactly 3 floats")
    use_rates = not integrated_budget_maps

    refs = (["recurvature", "et"] if reference == "both" else [reference])
    fig_dir = Path(output_directory)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for ref in refs:
        data_wp = budget_maps.load_composite(
            basin="WP", reference=ref,
            composites_dir=composites_directory, group=group,
            supp_dir=merra2_supplement_directory)
        data_na = budget_maps.load_composite(
            basin="NA", reference=ref,
            composites_dir=composites_directory, group=group,
            supp_dir=merra2_supplement_directory)
        if data_wp is None or data_na is None:
            raise SystemExit(
                f"Missing composite for WP or NA (reference={ref}).")
        budget_wp = budget_maps.compute_budget(
            data=data_wp, strip_rate_units=use_rates)
        budget_na = budget_maps.compute_budget(
            data=data_na, strip_rate_units=use_rates)
        if budget_wp is None or budget_na is None:
            raise SystemExit("Missing budget fields in composite NetCDF.")

        fv = fixed_vmax
        if not use_rates and fv is None:
            lag = data_wp["_lag_hours"]
            fv = budget_maps._compute_joint_vmax(
                data_list=[data_wp, data_na],
                budget_list=[budget_wp, budget_na],
                lag_hours=lag, t_start=t_start, t_end=t_end,
                sigma_2d=sigma, sigma_3d=vol_sigma)

        grp = f"_{group}" if group else ""
        out = fig_dir / figure_name.format(reference=ref, group=grp)
        n_wp = int(data_wp.get("_n_storms", 0))
        n_na = int(data_na.get("_n_storms", 0))
        budget_maps.plot_budget_wp_na_fig5(
            data_wp=data_wp, budget_wp=budget_wp,
            data_na=data_na, budget_na=budget_na,
            reference=ref,
            t_start=t_start,
            t_end=t_end,
            sigma=sigma,
            sigma_3d=vol_sigma,
            fixed_vmax=fv,
            rel_lat_ymin=rel_lat_ymin,
            budget_rates_ms_day=use_rates,
            show_flux_arrows=False,
            strat_column_titles=(
                f"WP   N = {n_wp}",
                f"NA   N = {n_na}",
            ),
            compare_with_diff_test=True,
            use_rwb_nonqg=True,
            stacked_layout=True,
            out_path=out,
        )
        print(f"Paper copy: {out}")


if __name__ == "__main__":
    typer.run(main)
