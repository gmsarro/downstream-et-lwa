"""CLI: budget + MERRA-2 source composites for highwb / lowwb strata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

import downstream_et_lwa.composites.merra2_source_composites as merra2_source_composites
import downstream_et_lwa.composites.run_composites as run_composites

_LOG = logging.getLogger(__name__)


def main(
        data_config: Annotated[Path, typer.Option(
            help="JSON file mapping data-source keys to root directories")],
        classification_csv: Annotated[Path, typer.Option(
            help="CSV with storm_id, reference, wb_group from classification.wb")],
        tracks_directory: Annotated[Path, typer.Option(
            help="Directory with recurving_nh_tracks.csv and individual/")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for composite NetCDF output")],
        merra2_output_directory: Annotated[Optional[Path], typer.Option(
            help="Directory for MERRA-2 source composites "
                 "(default: output_directory)")] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature, et, or both")] = "recurvature",
        basins: Annotated[Optional[list[str]], typer.Option()] = None,
        workers: Annotated[int, typer.Option()] = 12,
        skip_merra2: Annotated[bool, typer.Option(
            help="Skip the MERRA-2 source supplemental composites")] = False,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    if basins is None:
        basins = ["WPNA"]

    if not classification_csv.is_file():
        raise SystemExit(f"Classification CSV not found: {classification_csv}")

    for group in ("highwb", "lowwb"):
        print("=" * 60, flush=True)
        print(f"  Budget composites  group={group}", flush=True)
        run_composites.main(
            data_config=data_config,
            tracks_directory=tracks_directory,
            output_directory=output_directory,
            basins=list(basins),
            reference=reference,
            preset="budget",
            workers=workers,
            filter_csv=classification_csv,
            group=group,
            log_level=log_level,
        )

        if skip_merra2:
            continue
        print("=" * 60, flush=True)
        print(f"  MERRA-2 source composites  group={group}", flush=True)
        merra2_source_composites.main(
            data_config=data_config,
            tracks_directory=tracks_directory,
            output_directory=merra2_output_directory or output_directory,
            basins=list(basins),
            reference=reference,
            workers=workers,
            filter_csv=classification_csv,
            group=group,
            log_level=log_level,
        )


if __name__ == "__main__":
    typer.run(main)
