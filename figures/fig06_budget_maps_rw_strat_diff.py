"""Paper Fig. 6 difference version: RW case - no-RW case (WP only)."""

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
            help="Directory with composite_2d_<reference>_WP_rwcase.nc and "
                 "composite_2d_<reference>_WP_norwcase.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {reference} is substituted")]
        = "fig6_diff_rw_minus_norw_{reference}.png",
        merra2_supplement_directory: Annotated[Optional[Path], typer.Option(
            help="Directory with composite_2d_*_merra2src.nc supplementary "
                 "composites")] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature or et")] = "recurvature",
        t_start: Annotated[int, typer.Option()] = 0,
        t_end: Annotated[int, typer.Option()] = 144,
        sigma: Annotated[float, typer.Option()] = 2.0,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if reference not in ("recurvature", "et"):
        raise typer.BadParameter("reference must be recurvature or et")

    fig_dir = Path(output_directory)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_rw = budget_maps.load_composite(
        basin="WP", reference=reference,
        composites_dir=composites_directory, group="rwcase",
        supp_dir=merra2_supplement_directory)
    data_no = budget_maps.load_composite(
        basin="WP", reference=reference,
        composites_dir=composites_directory, group="norwcase",
        supp_dir=merra2_supplement_directory)
    if data_rw is None or data_no is None:
        raise SystemExit(
            f"Missing WP-only RW-stratified composite "
            f"(reference={reference}).")
    budget_rw = budget_maps.compute_budget(data=data_rw,
                                           strip_rate_units=True)
    budget_no = budget_maps.compute_budget(data=data_no,
                                           strip_rate_units=True)
    if budget_rw is None or budget_no is None:
        raise SystemExit("Missing budget fields in composite NetCDF.")

    n_rw = int(data_rw.get("_n_storms", 0))
    n_no = int(data_no.get("_n_storms", 0))

    out = fig_dir / figure_name.format(reference=reference)
    budget_maps.plot_budget_diff_fig(
        data_wp=data_rw, budget_wp=budget_rw,
        data_na=data_no, budget_na=budget_no,
        reference=reference,
        t_start=t_start,
        t_end=t_end,
        sigma=sigma,
        budget_rates_ms_day=True,
        title=(f"WP: RW case \N{MINUS SIGN} no-RW case   "
               f"(N={n_rw} vs N={n_no})"),
        use_rwb_nonqg=True,
        out_path=out,
    )
    print(f"Diff figure: {out}")


if __name__ == "__main__":
    typer.run(main)
