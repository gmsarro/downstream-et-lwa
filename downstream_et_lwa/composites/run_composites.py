"""CLI: build storm-relative composites for all variables across all storms."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from typing_extensions import Annotated

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.composites.engine as engine
import downstream_et_lwa.composites.io as composite_io
import downstream_et_lwa.data_registry as data_registry
import downstream_et_lwa.tracks as tracks

_LOG = logging.getLogger(__name__)

CORE_FAST_VARS = [
    "era5_lwa", "era5_ua1", "era5_ua2", "era5_ep1",
    "era5_ep2a", "era5_ep3a", "era5_ep4", "era5_Ub", "era5_Urefb",
    "merra2_lwa", "merra2_ua1", "merra2_ua2", "merra2_ep1",
    "merra2_ep2a", "merra2_ep3a", "merra2_ep4", "merra2_Ub", "merra2_Urefb",
    "era5_budget_termI", "era5_budget_termII", "era5_budget_termIII",
    "merra2_budget_termI", "merra2_budget_termII", "merra2_budget_termIII",
    "era5_qgpv_10km", "era5_rwb_awb", "era5_rwb_cwb",
    "era5_lh_lwa", "era5_nonqg_lwa",
    "era5_cc_Fc", "era5_cc_Ac", "era5_cc_C",
    "merra2_cc_Fc", "merra2_cc_Ac", "merra2_cc_C",
    "imerg_precip",
]

BUDGET_ONLY_VARS = [
    "era5_lwa",
    "era5_budget_termI", "era5_budget_termII", "era5_budget_termIII",
    "era5_lh_lwa", "era5_nonqg_lwa",
    "era5_qgpv_10km", "era5_rwb_awb", "era5_rwb_cwb",
    "era5_cc_Fc",
    "imerg_precip",
]

CORE_SLOW_VARS = [
    "era5_u_250hPa", "era5_v_250hPa",
    "era5_u_500hPa", "era5_v_500hPa",
    "era5_u_850hPa", "era5_v_850hPa",
    "era5_t_250hPa", "era5_t_500hPa", "era5_t_850hPa",
]

CORE_VARS = CORE_FAST_VARS + CORE_SLOW_VARS


def resolve_var_keys(
        *,
        variables: list[str] | None,
        category: str | None,
        preset: str,
) -> list[str]:
    if variables is not None:
        return list(variables)
    if category:
        return data_registry.list_keys(category=category)
    if preset == "all":
        return data_registry.list_keys()
    if preset == "budget":
        return [v for v in BUDGET_ONLY_VARS if data_registry.get(key=v) is not None]
    if preset == "core_slow":
        return [v for v in CORE_SLOW_VARS if data_registry.get(key=v) is not None]
    if preset == "core":
        return [v for v in CORE_VARS if data_registry.get(key=v) is not None]
    return [v for v in CORE_FAST_VARS if data_registry.get(key=v) is not None]


def main(
        data_config: Annotated[Path, typer.Option(
            help="JSON file mapping data-source keys to root directories")],
        tracks_directory: Annotated[Path, typer.Option(
            help="Directory with recurving_nh_tracks.csv and individual/")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for composite NetCDF output")],
        basins: Annotated[Optional[list[str]], typer.Option()] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature, et, or both")] = "both",
        variables: Annotated[Optional[list[str]], typer.Option(
            help="Registry keys")] = None,
        category: Annotated[Optional[str], typer.Option(
            help="Only variables of this category")] = None,
        preset: Annotated[str, typer.Option(
            help="core_fast, core_slow, core, all, budget")] = "core_fast",
        workers: Annotated[int, typer.Option(
            help="Parallel workers per reference")] = 12,
        filter_csv: Annotated[Optional[Path], typer.Option(
            help="CSV from classification CLIs to filter storms")] = None,
        group: Annotated[Optional[str], typer.Option(
            help="WB group to select (e.g. highwb, lowwb)")] = None,
        bake_volume_smoothing: Annotated[bool, typer.Option()] = False,
        premean_volume_sigma: Annotated[Optional[list[float]], typer.Option(
            help="Gaussian sigma (rel_lat, rel_lon, lag); 0 0 0 disables")] = None,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())

    cfg = data_registry.load_data_config(path=data_config)
    data_registry.register_all(data_config=cfg)

    if basins is None:
        basins = ["WP", "NA", "EP", "NI"]

    var_keys = resolve_var_keys(
        variables=variables, category=category, preset=preset)

    references = (["recurvature", "et"] if reference in ("both",)
                  else [reference])

    print(f"Variables: {len(var_keys)}")
    print(f"Basins: {basins}")
    print(f"References: {references}")
    print(f"Output: {output_directory}")
    if filter_csv:
        print(f"Filter CSV: {filter_csv}, group: {group}")

    os.makedirs(output_directory, exist_ok=True)

    print(f"Tracks: {tracks_directory}")
    storms_df, full_tracks = tracks.load_track_database(
        tracks_directory=tracks_directory)
    print(f"Loaded {len(storms_df)} storms")

    filter_df = None
    if filter_csv:
        filter_df = pd.read_csv(filter_csv, keep_default_na=False, na_values=[""])
        if group:
            filter_df = filter_df[filter_df["wb_group"] == group]
        print(f"Filter CSV: {len(filter_df)} storms after group={group}")

    if premean_volume_sigma is not None:
        premean_sigma = tuple(float(x) for x in premean_volume_sigma)
    else:
        premean_sigma = tuple(
            float(x) for x in composite_config.COMPOSITE_PREMEAN_VOLUME_SIGMA_3D)
    print(f"Premean volume sigma (rel_lat, rel_lon, lag): {premean_sigma}",
          flush=True)

    for ref in references:
        print(f"\n{'=' * 60}")
        print(f"  Building {ref}-relative composites ({workers} workers)")
        print(f"{'=' * 60}")

        run_storms = storms_df
        if filter_df is not None:
            ref_filter = filter_df[filter_df["reference"] == ref]
            keep_ids = set(ref_filter["storm_id"])
            run_storms = storms_df[storms_df["storm_id"].isin(keep_ids)]
            print(f"  Filtered to {len(run_storms)} storms for {ref}")

        accum = engine.build_composites_parallel(
            storms_df=run_storms, full_tracks=full_tracks,
            var_keys=var_keys, basins=basins, reference=ref,
            n_workers=workers, premean_volume_sigma_3d=premean_sigma,
        )

        composite_io.save_composites(
            accum=accum, reference=ref, output_directory=output_directory,
            group=group, bake_volume_smoothing=bake_volume_smoothing,
            premean_volume_sigma_3d=premean_sigma)


if __name__ == "__main__":
    typer.run(main)
