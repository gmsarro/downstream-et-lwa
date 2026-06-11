"""CLI: build recurving + ET-only IBTrACS track CSV databases."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.tracks as tracks

_LOG = logging.getLogger(__name__)


def main(
        ibtracs_file: Annotated[Path, typer.Option(
            help="IBTrACS ALL NetCDF file")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for track CSV databases")],
        year_start: Annotated[int, typer.Option()] = composite_config.YEAR_START,
        year_end: Annotated[int, typer.Option()] = composite_config.YEAR_END,
        et_only: Annotated[bool, typer.Option(
            help="Also build the ET-only NH database")] = True,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())

    if not ibtracs_file.is_file():
        raise FileNotFoundError(f"IBTrACS not found: {ibtracs_file}")

    print(f"IBTrACS: {ibtracs_file}", flush=True)
    print(f"Years: {year_start}-{year_end}", flush=True)
    data = tracks.load_ibtracs(path=ibtracs_file)

    storms_df, full_tracks = tracks.build_track_database(
        ibtracs_data=data, year_start=year_start, year_end=year_end)
    tracks.save_track_database(
        storms_df=storms_df, full_tracks=full_tracks,
        output_directory=output_directory)

    print("\n=== Summary by basin and decade ===")
    for basin in sorted(storms_df["basin"].unique()):
        sub = storms_df[storms_df["basin"] == basin]
        decades = sub["season"].apply(lambda y: f"{(y // 10) * 10}s")
        print(f"\n{basin}:")
        print(decades.value_counts().sort_index().to_string())

    if et_only:
        storms_et, tracks_et = tracks.build_et_nh_track_database(
            ibtracs_data=data, year_start=year_start, year_end=year_end)
        tracks.save_et_nh_track_database(
            storms_df=storms_et, full_tracks=tracks_et,
            output_directory=output_directory)


if __name__ == "__main__":
    typer.run(main)
