"""Paper Fig. 5 difference version: WP - NA budget-composite maps with the
(WP + NA)/2 average as contours and Welch diff-test hatching."""

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
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {reference} is substituted")]
        = "fig5_diff_wp_minus_na_{reference}.png",
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

    data_wp = budget_maps.load_composite(
        basin="WP", reference=reference,
        composites_dir=composites_directory,
        supp_dir=merra2_supplement_directory)
    data_na = budget_maps.load_composite(
        basin="NA", reference=reference,
        composites_dir=composites_directory,
        supp_dir=merra2_supplement_directory)
    if data_wp is None or data_na is None:
        raise SystemExit(
            f"Missing composite for WP or NA (reference={reference}).")
    budget_wp = budget_maps.compute_budget(data=data_wp,
                                           strip_rate_units=True)
    budget_na = budget_maps.compute_budget(data=data_na,
                                           strip_rate_units=True)
    if budget_wp is None or budget_na is None:
        raise SystemExit("Missing budget fields in composite NetCDF.")

    n_wp = int(data_wp.get("_n_storms", 0))
    n_na = int(data_na.get("_n_storms", 0))

    out = fig_dir / figure_name.format(reference=reference)
    budget_maps.plot_budget_diff_fig(
        data_wp=data_wp, budget_wp=budget_wp,
        data_na=data_na, budget_na=budget_na,
        reference=reference,
        t_start=t_start,
        t_end=t_end,
        sigma=sigma,
        budget_rates_ms_day=True,
        title=(f"WP \N{MINUS SIGN} NA   (N={n_wp} vs N={n_na})"),
        use_rwb_nonqg=True,
        out_path=out,
    )
    print(f"Diff figure: {out}")


if __name__ == "__main__":
    typer.run(main)
