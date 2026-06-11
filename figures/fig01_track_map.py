"""Paper Fig. 1: IBTrACS + MPAS current/future track maps (3x2 and 3x4
layouts), rebuilding the IBTrACS track CSV databases first."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.plotting.track_maps as track_maps
import downstream_et_lwa.tracks as tracks

_LOG = logging.getLogger(__name__)


def _rebuild_ibtracs(*, ibtracs_file: Path, tracks_directory: Path,
                     year_start: int, year_end: int) -> None:
    data = tracks.load_ibtracs(path=ibtracs_file)
    storms_df, full_tracks = tracks.build_track_database(
        ibtracs_data=data, year_start=year_start, year_end=year_end)
    tracks.save_track_database(
        storms_df=storms_df, full_tracks=full_tracks,
        output_directory=tracks_directory)
    storms_et, tracks_et = tracks.build_et_nh_track_database(
        ibtracs_data=data, year_start=year_start, year_end=year_end)
    tracks.save_et_nh_track_database(
        storms_df=storms_et, full_tracks=tracks_et,
        output_directory=tracks_directory)


def main(
        ibtracs_file: Annotated[Path, typer.Option(
            help="IBTrACS ALL NetCDF file")],
        tracks_directory: Annotated[Path, typer.Option(
            help="Track database directory (rebuilt here, then read)")],
        mpas_all_tracks_root: Annotated[Path, typer.Option(
            help="Root of extracted MPAS ettrack .txt files")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the figure outputs")],
        figure_name: Annotated[str, typer.Option(
            help="Output filename; {layout} is substituted")]
        = "track_map_ibtracs_and_mpas_{layout}.png",
        layouts: Annotated[Optional[list[str]], typer.Option(
            help="Layouts to render (default: 3x2 and 3x4)")] = None,
        year_start: Annotated[int, typer.Option()]
        = composite_config.YEAR_START,
        year_end: Annotated[int, typer.Option()] = composite_config.YEAR_END,
        rebuild_tracks: Annotated[bool, typer.Option(
            help="Rebuild the IBTrACS track CSVs before plotting")] = True,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())
    if layouts is None:
        layouts = ["3x2", "3x4"]
    for layout in layouts:
        if layout not in ("3x2", "3x4"):
            raise typer.BadParameter("layouts must be 3x2 or 3x4")
    if not ibtracs_file.is_file():
        raise FileNotFoundError(f"IBTrACS NetCDF not found: {ibtracs_file}")
    if not mpas_all_tracks_root.is_dir():
        raise FileNotFoundError(
            f"MPAS ettrack .txt root not found: {mpas_all_tracks_root}")

    if rebuild_tracks:
        print(f"Rebuilding IBTrACS track databases in {tracks_directory}",
              flush=True)
        _rebuild_ibtracs(ibtracs_file=ibtracs_file,
                         tracks_directory=tracks_directory,
                         year_start=year_start, year_end=year_end)

    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    for layout in layouts:
        out = out_dir / figure_name.format(layout=layout)
        result = track_maps.plot_track_map_combined(
            tracks_directory=tracks_directory,
            output_path=out,
            layout=layout,
            all_tracks_root=mpas_all_tracks_root,
            use_all_tracks=True,
            build_et_ibtracs=False,
            ibtracs_path=ibtracs_file,
        )
        print(f"Saved: {result}")


if __name__ == "__main__":
    typer.run(main)
