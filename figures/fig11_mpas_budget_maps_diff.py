"""MPAS future minus current LWA-budget difference maps, WP+NA pooled
(paper Fig. 11, difference version)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

import downstream_et_lwa.plotting.budget_maps as budget_maps
import downstream_et_lwa.plotting.mpas_wpna_io as mpas_wpna_io


def main(
        mpas_composites_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_<reference>_mpas_<scenario>_"
                 "<basin>.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        reference: Annotated[str, typer.Option(
            help="recurvature or et")] = "recurvature",
        t_start: Annotated[int, typer.Option()] = 0,
        t_end: Annotated[int, typer.Option()] = 144,
        sigma: Annotated[float, typer.Option()] = 2.0,
        figure_name: Annotated[Optional[str], typer.Option(
            help="Defaults to fig11_diff_mpas_future_minus_current_"
                 "<reference>.png")] = None,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if reference not in ("recurvature", "et"):
        raise typer.BadParameter("reference must be recurvature or et")

    cdir = Path(mpas_composites_directory)
    d_c_raw = mpas_wpna_io.load_mpas_wpna_pooled(
        composite_dir=cdir, reference=reference, scenario="current")
    d_f_raw = mpas_wpna_io.load_mpas_wpna_pooled(
        composite_dir=cdir, reference=reference, scenario="future")

    d_c = mpas_wpna_io.alias_mpas_to_era5_keys(data_mp=d_c_raw)
    d_f = mpas_wpna_io.alias_mpas_to_era5_keys(data_mp=d_f_raw)
    b_c = budget_maps.compute_budget(data=d_c, strip_rate_units=True)
    b_f = budget_maps.compute_budget(data=d_f, strip_rate_units=True)
    if b_c is None or b_f is None:
        raise SystemExit(
            "compute_budget returned None (missing era5_lwa alias?).")

    n_c = int(d_c.get("_n_storms", 0))
    n_f = int(d_f.get("_n_storms", 0))

    name = figure_name or (
        f"fig11_diff_mpas_future_minus_current_{reference}.png")
    out = Path(output_directory) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    budget_maps.plot_budget_diff_fig(
        data_wp=d_f, budget_wp=b_f, data_na=d_c, budget_na=b_c,
        reference=reference,
        t_start=t_start,
        t_end=t_end,
        sigma=sigma,
        budget_rates_ms_day=True,
        anomaly_lh_mst=True,
        title=f"MPAS: Future \N{MINUS SIGN} Current (WP+NA)   "
              f"(N={n_f} vs N={n_c})",
        use_rwb_nonqg=True,
        nonqg_source_tag="MPAS",
        qgpv_field_key="mpas_qgpv_10km",
        skip_mst=True,
        show_coastlines=False,
        out_path=out,
    )
    print(f"Diff figure: {out}")


if __name__ == "__main__":
    typer.run(main)
