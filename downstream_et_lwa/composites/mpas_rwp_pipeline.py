"""Orchestrate the MPAS RWP pipeline (current/future): 1-deg v250 extraction,
RWP envelope months, seasonal climatology (via python -m subprocess calls into
downstream_et_lwa.preprocessing / downstream_et_lwa.rwp), and recurving-track
databases.  Figure plotting lives in the separate figure scripts."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

import downstream_et_lwa.composites.mpas_composites as mpas_composites
import downstream_et_lwa.tracks as tracks

_LOG = logging.getLogger(__name__)

YEAR_START_DEFAULT = 1988
YEAR_END_DEFAULT = 2016


def _run(*, cmd: list[str], dry: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if dry:
        return
    subprocess.run(cmd, check=True)


def main(
        mpas_combined_directory: Annotated[Path, typer.Option(
            help="Root of MPAS combined daily NH subset files "
                 "(per-scenario subdirectories)")],
        output_directory: Annotated[Path, typer.Option(
            help="Root for 1deg_extracted/, rwp/, and tracks/ outputs")],
        scenario: Annotated[str, typer.Option(
            help="current, future, or both")] = "both",
        year_start: Annotated[int, typer.Option()] = YEAR_START_DEFAULT,
        year_end: Annotated[int, typer.Option()] = YEAR_END_DEFAULT,
        et_track_directory: Annotated[Optional[Path], typer.Option(
            help="Directory with traj_et_mpas_avg_*_{scenario}.dat")] = None,
        all_tracks_root: Annotated[Optional[Path], typer.Option(
            help="Directory walked for ettrack_{scenario}_*.txt")] = None,
        n_workers: Annotated[int, typer.Option()] = 8,
        extract_workers: Annotated[int, typer.Option()] = 8,
        force: Annotated[bool, typer.Option(
            help="Rebuild 1-deg, RWP, climatology even if present")] = False,
        skip_extract: Annotated[bool, typer.Option()] = False,
        skip_rwp: Annotated[bool, typer.Option()] = False,
        skip_clim: Annotated[bool, typer.Option()] = False,
        skip_tracks: Annotated[bool, typer.Option()] = False,
        dry_run: Annotated[bool, typer.Option()] = False,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())

    scenarios = (["current", "future"] if scenario == "both" else [scenario])
    py = sys.executable

    extract_base = output_directory / "1deg_extracted"
    rwp_base = output_directory / "rwp"
    tracks_root = output_directory / "tracks"

    for scen in scenarios:
        one_deg = extract_base / scen
        rwp_dir = rwp_base / scen
        clim_path = rwp_dir / "rwp_climatology.nc"
        tracks_base = tracks_root / scen

        if not skip_extract:
            _run(cmd=[
                py, "-m", "downstream_et_lwa.preprocessing.extract_mpas_v250",
                "--scenario", scen,
                "--input-directory", str(mpas_combined_directory),
                "--output-directory", str(extract_base),
                "--year-start", str(year_start),
                "--year-end", str(year_end),
                "--workers", str(extract_workers),
            ] + (["--overwrite"] if force else []),
                dry=dry_run)

        if not skip_rwp:
            _run(cmd=[
                py, "-m", "downstream_et_lwa.rwp.envelope",
                "--input-directory", str(one_deg),
                "--output-directory", str(rwp_dir),
                "--year-start", str(year_start),
                "--year-end", str(year_end),
                "--workers", str(n_workers),
            ] + (["--overwrite"] if force else []),
                dry=dry_run)

        if not skip_clim:
            _run(cmd=[
                py, "-m", "downstream_et_lwa.rwp.climatology",
                "--input-directory", str(rwp_dir),
                "--output-directory", str(rwp_dir),
                "--year-start", str(year_start),
                "--year-end", str(year_end),
                "--seasons", "JJASON", "--seasons", "DJFMA", "--seasons", "ANN",
                "--output-filename", clim_path.name,
            ], dry=dry_run)

        if dry_run:
            print(f"(dry-run) would build tracks for {scen}")
            continue

        if skip_tracks:
            continue

        storms_df, full_tracks = mpas_composites.build_mpas_tracks(
            scenario=scen,
            et_track_dir=et_track_directory,
            all_tracks_root=all_tracks_root,
        )
        if len(storms_df) == 0:
            print(f"  [warn] no MPAS/{scen} tracks; skipping scenario")
            continue

        os.makedirs(tracks_base, exist_ok=True)
        tracks.save_track_database(
            storms_df=storms_df, full_tracks=full_tracks,
            output_directory=tracks_base)


if __name__ == "__main__":
    typer.run(main)
