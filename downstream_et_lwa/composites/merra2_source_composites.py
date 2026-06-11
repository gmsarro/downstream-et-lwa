"""CLI: supplementary MERRA-2 LWA-source-decomposition composites (_merra2src files)."""

from __future__ import annotations

import logging
import os
import shutil
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

VARS = [
    "merra2_lwa",
    "merra2_heat_fortran_DTDTMST",
    "merra2_heat_fortran_DTDTRAD",
    "merra2_heat_fortran_DTDTANA",
    "merra2_heat_fortran_DTDTTOT",
    "merra2_budget_termI",
    "merra2_budget_termII",
    "merra2_budget_termIII",
]


def main(
        data_config: Annotated[Path, typer.Option(
            help="JSON file mapping data-source keys to root directories")],
        tracks_directory: Annotated[Path, typer.Option(
            help="Directory with recurving_nh_tracks.csv and individual/")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for MERRA-2 source composite NetCDFs")],
        basins: Annotated[Optional[list[str]], typer.Option()] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature, et, or both")] = "both",
        workers: Annotated[int, typer.Option()] = 10,
        filter_csv: Annotated[Optional[Path], typer.Option()] = None,
        group: Annotated[Optional[str], typer.Option(
            help="Stratification label (e.g. highwb, lowwb)")] = None,
        premean_volume_sigma: Annotated[Optional[list[float]], typer.Option(
            help="Per-storm volume Gaussian before ensemble mean")] = None,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())

    cfg = data_registry.load_data_config(path=data_config)
    data_registry.register_all(data_config=cfg)

    if basins is None:
        basins = ["WP", "NA"]

    premian = (
        tuple(float(x) for x in premean_volume_sigma)
        if premean_volume_sigma is not None
        else tuple(float(x)
                   for x in composite_config.COMPOSITE_PREMEAN_VOLUME_SIGMA_3D)
    )

    os.makedirs(output_directory, exist_ok=True)
    print(f"Variables: {VARS}")
    print(f"Basins: {basins}")
    print(f"Output: {output_directory}")

    refs = (["recurvature", "et"] if reference == "both" else [reference])

    print(f"Tracks: {tracks_directory}", flush=True)
    storms_df, full_tracks = tracks.load_track_database(
        tracks_directory=tracks_directory)
    print(f"Loaded {len(storms_df)} storms")

    filter_df = None
    if filter_csv:
        filter_df = pd.read_csv(filter_csv, keep_default_na=False,
                                na_values=[""])
        if group:
            filter_df = filter_df[filter_df["wb_group"] == group]

    for ref in refs:
        print(f"\n{'=' * 60}\n  Building {ref}-relative MERRA-2 source composites"
              f"\n{'=' * 60}")

        run_storms = storms_df
        if filter_df is not None:
            ref_filter = filter_df[filter_df["reference"] == ref]
            keep_ids = set(ref_filter["storm_id"])
            run_storms = storms_df[storms_df["storm_id"].isin(keep_ids)]
            print(f"  Filtered to {len(run_storms)} storms for {ref}")

        accum = engine.build_composites_parallel(
            storms_df=run_storms, full_tracks=full_tracks,
            var_keys=VARS, basins=basins, reference=ref,
            n_workers=workers, premean_volume_sigma_3d=premian,
        )
        composite_io.save_composites(
            accum=accum, reference=ref, output_directory=output_directory,
            group=group, premean_volume_sigma_3d=premian)

        for basin in basins:
            base = f"composite_2d_{ref}_{basin}"
            if group:
                base += f"_{group}"
            old = Path(output_directory) / f"{base}.nc"
            new = Path(output_directory) / f"{base}_merra2src.nc"
            if old.exists():
                shutil.move(str(old), str(new))
                print(f"  -> {new.name}")


if __name__ == "__main__":
    typer.run(main)
