"""Paper Fig. 6 (RWB stratification): budget-composite maps for high vs low
downstream RWB with WP + NA storms pooled (WPNA bucket)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

import downstream_et_lwa.plotting.budget_maps as budget_maps

_LOG = logging.getLogger(__name__)

POOL = "WPNA"


def main(
        composites_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_<reference>_WPNA_highwb.nc "
                 "and composite_2d_<reference>_WPNA_lowwb.nc")],
        merra2_supplement_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_*_merra2src.nc supplementary "
                 "composites")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {reference} is substituted")]
        = "fig06_budget_wpna_rwb_strat_{reference}.png",
        reference: Annotated[str, typer.Option(
            help="recurvature, et, or both")] = "recurvature",
        t_start: Annotated[int, typer.Option()] = 0,
        t_end: Annotated[int, typer.Option()] = 144,
        rel_lat_ymin: Annotated[float, typer.Option()] = -10.0,
        sigma: Annotated[float, typer.Option(
            help="Extra per-storm 2D Gaussian sigma (deg)")] = 2.0,
        volume_sigma: Annotated[Optional[list[float]], typer.Option(
            help="Optional 3-D Gaussian before time mean (3 floats; "
                 "default 0 0 0)")] = None,
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
        data_hi = budget_maps.load_composite(
            basin=POOL, reference=ref,
            composites_dir=composites_directory, group="highwb",
            supp_dir=merra2_supplement_directory)
        data_lo = budget_maps.load_composite(
            basin=POOL, reference=ref,
            composites_dir=composites_directory, group="lowwb",
            supp_dir=merra2_supplement_directory)
        if data_hi is None or data_lo is None:
            raise SystemExit(
                f"Missing pooled WPNA stratified composite "
                f"(reference={ref}).")
        n_hi = int(data_hi["_n_storms"])
        n_lo = int(data_lo["_n_storms"])
        titles = (
            f"WP+NA \N{EM DASH} high downstream RWB (top quintile), "
            f"N = {n_hi}",
            f"WP+NA \N{EM DASH} low downstream RWB (bottom quintile), "
            f"N = {n_lo}",
        )
        budget_hi = budget_maps.compute_budget(
            data=data_hi, strip_rate_units=use_rates)
        budget_lo = budget_maps.compute_budget(
            data=data_lo, strip_rate_units=use_rates)
        if budget_hi is None or budget_lo is None:
            raise SystemExit("Missing budget fields in composite NetCDF.")

        fv = fixed_vmax
        if not use_rates and fv is None:
            lag = data_hi["_lag_hours"]
            fv = budget_maps._compute_joint_vmax(
                data_list=[data_hi, data_lo],
                budget_list=[budget_hi, budget_lo],
                lag_hours=lag, t_start=t_start, t_end=t_end,
                sigma_2d=sigma, sigma_3d=vol_sigma)

        out = fig_dir / figure_name.format(reference=ref)
        budget_maps.plot_budget_wp_na_fig5(
            data_wp=data_hi, budget_wp=budget_hi,
            data_na=data_lo, budget_na=budget_lo,
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
            out_path=out,
        )
        print(f"Wrote {out}")


if __name__ == "__main__":
    typer.run(main)
