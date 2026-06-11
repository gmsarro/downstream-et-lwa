"""MPAS current vs future LWA-budget composite maps in the Fig. 5 layout,
WP+NA pooled by default (paper Fig. 11)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from typing_extensions import Annotated

import downstream_et_lwa.composites.mpas_composites as mpas_composites
import downstream_et_lwa.plotting.budget_maps as budget_maps
import downstream_et_lwa.plotting.mpas_wpna_io as mpas_wpna_io


def main(
        mpas_composites_directory: Annotated[Path, typer.Option(
            help="Directory with composite_2d_<reference>_mpas_<scenario>_"
                 "<basin>.nc")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure output")],
        basin: Annotated[str, typer.Option(
            help="WPNA = pooled WP+NA (default); else single-basin "
                 "file")] = "WPNA",
        reference: Annotated[str, typer.Option()] = "recurvature",
        t_start: Annotated[int, typer.Option()] = 0,
        t_end: Annotated[int, typer.Option()] = 144,
        figure_name: Annotated[Optional[str], typer.Option(
            help="Defaults to fig11_mpas_budget_maps_cf_<basin>_"
                 "<reference>.png")] = None,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if basin not in ("WPNA", "WP", "NA"):
        raise typer.BadParameter("basin must be WPNA, WP, or NA")
    if reference != "recurvature":
        raise typer.BadParameter("reference must be recurvature")

    cdir = Path(mpas_composites_directory)
    if basin == "WPNA":
        d_c_raw = mpas_wpna_io.load_mpas_wpna_pooled(
            composite_dir=cdir, reference=reference, scenario="current")
        d_f_raw = mpas_wpna_io.load_mpas_wpna_pooled(
            composite_dir=cdir, reference=reference, scenario="future")
    else:
        path_c = cdir / f"composite_2d_{reference}_mpas_current_{basin}.nc"
        path_f = cdir / f"composite_2d_{reference}_mpas_future_{basin}.nc"
        if not path_c.is_file() or not path_f.is_file():
            raise SystemExit(
                f"Missing composite(s): {path_c} and/or {path_f}")
        d_c_raw = mpas_composites.load_mpas_composite(
            path=path_c, prefix="mpas_current")
        d_f_raw = mpas_composites.load_mpas_composite(
            path=path_f, prefix="mpas_future")

    d_c = mpas_wpna_io.alias_mpas_to_era5_keys(data_mp=d_c_raw)
    d_f = mpas_wpna_io.alias_mpas_to_era5_keys(data_mp=d_f_raw)
    b_c = budget_maps.compute_budget(data=d_c, strip_rate_units=True)
    b_f = budget_maps.compute_budget(data=d_f, strip_rate_units=True)
    if b_c is None or b_f is None:
        raise SystemExit(
            "compute_budget returned None (missing era5_lwa alias?).")

    n_c = int(d_c.get("_n_storms", 0))
    n_f = int(d_f.get("_n_storms", 0))

    has_mpas_pr = (
        d_c.get("mpas_precip") is not None
        and np.any(np.isfinite(d_c["mpas_precip"]))
        and d_f.get("mpas_precip") is not None
        and np.any(np.isfinite(d_f["mpas_precip"]))
    )
    if has_mpas_pr:
        pr_note = "(%(letter)s) MPAS precipitation (mm hr$^{-1}$)"
    else:
        pr_note = (
            "(%(letter)s) MPAS precipitation (mm hr$^{-1}$) — missing "
            "``mpas_precip`` in composite (rebuild with map diagnostics)"
        )

    has_mpas_lh = (
        d_c.get("era5_lh_lwa") is not None
        and np.any(np.isfinite(d_c["era5_lh_lwa"]))
        and d_f.get("era5_lh_lwa") is not None
        and np.any(np.isfinite(d_f["era5_lh_lwa"]))
    )
    if has_mpas_lh:
        lh_title = "(%(letter)s) MPAS LH-LWA anomaly (T$-$pre)"
    else:
        lh_title = "(%(letter)s) MPAS LH-LWA — missing ``mpas_lh_lwa``"

    name = figure_name or (
        f"fig11_mpas_budget_maps_cf_{basin.lower()}_{reference}.png")
    out = Path(output_directory) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    budget_maps.plot_budget_wp_na_fig5(
        data_wp=d_c, budget_wp=b_c, data_na=d_f, budget_na=b_f,
        reference=reference,
        t_start=t_start,
        t_end=t_end,
        budget_rates_ms_day=True,
        strat_column_titles=(
            f"MPAS current "
            f"({'WP+NA pooled' if basin == 'WPNA' else basin}, N={n_c})",
            f"MPAS future "
            f"({'WP+NA pooled' if basin == 'WPNA' else basin}, N={n_f})",
        ),
        allow_missing_mst=True,
        include_bottom_row=True,
        bottom_precip_title=pr_note,
        lh_title=lh_title,
        show_qgpv=True,
        qgpv_field_key="mpas_qgpv_10km",
        mst_panel_mode="none",
        show_flux_arrows=False,
        compare_with_diff_test=True,
        use_rwb_nonqg=True,
        nonqg_source_tag="MPAS",
        sigma=2.0,
        stacked_layout=True,
        show_coastlines=False,
        out_path=out,
    )
    print(f"Paper copy: {out}")


if __name__ == "__main__":
    typer.run(main)
