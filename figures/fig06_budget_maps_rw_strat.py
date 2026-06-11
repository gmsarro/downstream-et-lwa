"""Paper Fig. 6 (RW stratification): budget-composite maps for RW vs no-RW
cases (Quinting & Jones 2016 quintiles), WP-only sample."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

import downstream_et_lwa.plotting.budget_maps as budget_maps

_LOG = logging.getLogger(__name__)

POOL = "WP"


def main(
        composites_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_<reference>_WP_rwcase.nc and "
                 "composite_2d_<reference>_WP_norwcase.nc")],
        merra2_supplement_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_*_merra2src.nc supplementary "
                 "composites")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {reference} is substituted")]
        = "fig06_budget_wpna_rw_strat_{reference}.png",
        reference: Annotated[str, typer.Option(
            help="recurvature, et, or both")] = "recurvature",
        t_start: Annotated[int, typer.Option()] = 0,
        t_end: Annotated[int, typer.Option()] = 144,
        rel_lat_ymin: Annotated[float, typer.Option()] = -10.0,
        sigma: Annotated[float, typer.Option(
            help="Extra per-storm 2D Gaussian sigma (deg)")] = 2.0,
        volume_sigma: Annotated[Optional[list[float]], typer.Option(
            help="3-D Gaussian sigma (rel_lat, rel_lon, lag) before time "
                 "mean (3 floats; default 0 0 0)")] = None,
        integrated_budget_maps: Annotated[bool, typer.Option()] = False,
        fixed_vmax: Annotated[Optional[float], typer.Option()] = None,
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
        data_rw = budget_maps.load_composite(
            basin=POOL, reference=ref,
            composites_dir=composites_directory, group="rwcase",
            supp_dir=merra2_supplement_directory)
        data_no = budget_maps.load_composite(
            basin=POOL, reference=ref,
            composites_dir=composites_directory, group="norwcase",
            supp_dir=merra2_supplement_directory)
        if data_rw is None or data_no is None:
            raise SystemExit(
                f"Missing WP-only RW-stratified composite (reference={ref}).")
        n_rw = int(data_rw["_n_storms"])
        n_no = int(data_no["_n_storms"])
        titles = (
            f"WP   RW case (top quintile, Q&J 2016 score)   N = {n_rw}",
            f"WP   no-RW case (bottom quintile)   N = {n_no}",
        )
        budget_rw = budget_maps.compute_budget(
            data=data_rw, strip_rate_units=use_rates)
        budget_no = budget_maps.compute_budget(
            data=data_no, strip_rate_units=use_rates)
        if budget_rw is None or budget_no is None:
            raise SystemExit("Missing budget fields in composite NetCDF.")

        fv = fixed_vmax
        if not use_rates and fv is None:
            lag = data_rw["_lag_hours"]
            fv = budget_maps._compute_joint_vmax(
                data_list=[data_rw, data_no],
                budget_list=[budget_rw, budget_no],
                lag_hours=lag, t_start=t_start, t_end=t_end,
                sigma_2d=sigma, sigma_3d=vol_sigma)

        out = fig_dir / figure_name.format(reference=ref)
        budget_maps.plot_budget_wp_na_fig5(
            data_wp=data_rw, budget_wp=budget_rw,
            data_na=data_no, budget_na=budget_no,
            reference=ref,
            t_start=t_start,
            t_end=t_end,
            sigma=sigma,
            sigma_3d=vol_sigma,
            fixed_vmax=fv,
            rel_lat_ymin=rel_lat_ymin,
            budget_rates_ms_day=use_rates,
            strat_column_titles=titles,
            include_bottom_row=True,
            show_flux_arrows=False,
            compare_with_diff_test=True,
            use_rwb_nonqg=True,
            stacked_layout=True,
            out_path=out,
        )
        print(f"Wrote {out}")


if __name__ == "__main__":
    typer.run(main)
