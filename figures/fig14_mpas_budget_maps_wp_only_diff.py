"""MPAS future minus current LWA-budget difference maps, WP storms only
(paper Fig. 14, difference version)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

import downstream_et_lwa.composites.mpas_composites as mpas_composites
import downstream_et_lwa.plotting.budget_maps as budget_maps
import downstream_et_lwa.plotting.mpas_wpna_io as mpas_wpna_io


def main(
        mpas_composites_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_<reference>_mpas_<scenario>_"
                 "WP.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        reference: Annotated[str, typer.Option()] = "recurvature",
        t_start: Annotated[int, typer.Option()] = 0,
        t_end: Annotated[int, typer.Option()] = 144,
        sigma: Annotated[float, typer.Option()] = 2.0,
        figure_name: Annotated[str, typer.Option()] = (
            "fig14_diff_mpas_future_minus_current_wp_only_recurvature.png"),
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if reference != "recurvature":
        raise typer.BadParameter("reference must be recurvature")

    cdir = Path(mpas_composites_directory)
    basin = "WP"
    path_c = cdir / f"composite_2d_{reference}_mpas_current_{basin}.nc"
    path_f = cdir / f"composite_2d_{reference}_mpas_future_{basin}.nc"
    if not path_c.is_file() or not path_f.is_file():
        raise SystemExit(f"Missing composite(s): {path_c} and/or {path_f}")

    d_c = mpas_wpna_io.alias_mpas_to_era5_keys(
        data_mp=mpas_composites.load_mpas_composite(
            path=path_c, prefix="mpas_current"))
    d_f = mpas_wpna_io.alias_mpas_to_era5_keys(
        data_mp=mpas_composites.load_mpas_composite(
            path=path_f, prefix="mpas_future"))
    b_c = budget_maps.compute_budget(data=d_c, strip_rate_units=True)
    b_f = budget_maps.compute_budget(data=d_f, strip_rate_units=True)
    if b_c is None or b_f is None:
        raise SystemExit("compute_budget returned None.")

    n_c = int(d_c.get("_n_storms", 0))
    n_f = int(d_f.get("_n_storms", 0))

    out = Path(output_directory) / figure_name
    out.parent.mkdir(parents=True, exist_ok=True)
    budget_maps.plot_budget_diff_fig(
        data_wp=d_f, budget_wp=b_f, data_na=d_c, budget_na=b_c,
        reference=reference,
        t_start=t_start,
        t_end=t_end,
        sigma=sigma,
        budget_rates_ms_day=True,
        title=f"MPAS: Future \N{MINUS SIGN} Current (WP only)   "
              f"(N={n_f} vs N={n_c})",
        use_rwb_nonqg=True,
        nonqg_source_tag="MPAS",
        qgpv_field_key="mpas_qgpv_10km",
        skip_mst=True,
        out_path=out,
    )
    print(f"Diff figure: {out}")


if __name__ == "__main__":
    typer.run(main)
