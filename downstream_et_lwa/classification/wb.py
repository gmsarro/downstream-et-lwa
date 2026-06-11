"""Classify storms by downstream wave breaking (Q&J 2016 quintile analog):
mean AWB+CWB frequency in a fixed storm-relative downstream box over T+24..+120h,
single global quintile thresholds -> highwb / midwb / lowwb."""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.data_registry as data_registry
import downstream_et_lwa.grid_utils as grid_utils
import downstream_et_lwa.tracks as tracks

_LOG = logging.getLogger(__name__)


def compute_wb_metric(
        *,
        storm: pd.Series,
        full_tracks: dict[str, pd.DataFrame],
        reference: str,
        rel_lon_min: int = 10,
        rel_lon_max: int = 80,
        rel_lat_min: int = -5,
        rel_lat_max: int = 15,
        lag_start: int = 24,
        lag_end: int = 120,
        file_cache: dict | None = None,
) -> float:
    if file_cache is None:
        file_cache = {}

    awb_src = data_registry.get(key="era5_rwb_awb")
    cwb_src = data_registry.get(key="era5_rwb_cwb")
    if awb_src is None or cwb_src is None:
        return float(np.nan)

    sid = storm["storm_id"]
    if sid not in full_tracks:
        return float(np.nan)

    if reference == "recurvature":
        ref_time = pd.Timestamp(storm["recurv_time"])
        center_lat = float(storm["recurv_lat"])
        center_lon = float(storm["recurv_lon"])
    else:
        ref_time = pd.Timestamp(storm["et_time"])
        center_lat = float(storm["et_lat"])
        center_lon = float(storm["et_lon"])

    if pd.isna(ref_time):
        return float(np.nan)
    center_lon = center_lon % 360

    lag_mask = ((composite_config.LAG_HOURS >= lag_start)
                & (composite_config.LAG_HOURS <= lag_end))
    lag_indices = np.where(lag_mask)[0]

    rlat = np.arange(-composite_config.BOX_LAT_HALF,
                     composite_config.BOX_LAT_HALF + 1)
    rlon = np.arange(-composite_config.BOX_LON_WEST,
                     composite_config.BOX_LON_EAST + 1)
    lat_mask = (rlat >= rel_lat_min) & (rlat <= rel_lat_max)
    lon_mask = (rlon >= rel_lon_min) & (rlon <= rel_lon_max)

    wb_vals = []
    for li in lag_indices:
        lag_h = int(composite_config.LAG_HOURS[li])
        dt = ref_time + datetime.timedelta(hours=lag_h)
        raw_a = data_registry.load_snapshot(
            source=awb_src, target_dt=dt.to_pydatetime(), cache=file_cache)
        raw_c = data_registry.load_snapshot(
            source=cwb_src, target_dt=dt.to_pydatetime(), cache=file_cache)
        if raw_a is None or raw_c is None:
            continue
        fa = grid_utils.prepare_field(data=raw_a, source=awb_src)
        fc = grid_utils.prepare_field(data=raw_c, source=cwb_src)
        if fa is None or fc is None:
            continue
        field_2d = (fa > 0.5).astype(np.float32) + (fc > 0.5).astype(np.float32)
        patch = grid_utils.extract_storm_patch(
            field_2d=field_2d, center_lat=center_lat, center_lon=center_lon)
        if patch is None:
            continue
        sub = patch[np.ix_(lat_mask, lon_mask)]
        if np.any(np.isfinite(sub)):
            wb_vals.append(float(np.nanmean(sub)))

    return float(np.nanmean(wb_vals)) if wb_vals else float(np.nan)


def compute_all_metrics(
        *,
        storms_df: pd.DataFrame,
        full_tracks: dict[str, pd.DataFrame],
        basins: list[str],
        reference: str,
) -> pd.DataFrame:
    file_cache: dict = {}
    records = []
    n_done = 0
    try:
        for basin in basins:
            storm_list = storms_df[storms_df["basin"] == basin].copy()
            if reference == "et":
                storm_list = storm_list[storm_list["et_time"].notna()]

            for _, storm in storm_list.iterrows():
                metric = compute_wb_metric(
                    storm=storm, full_tracks=full_tracks,
                    reference=reference, file_cache=file_cache)
                records.append({
                    "storm_id": storm["storm_id"],
                    "name": storm["name"],
                    "basin": basin,
                    "season": storm["season"],
                    "reference": reference,
                    "wb_metric": metric,
                })
                n_done += 1
                if n_done % 200 == 0:
                    _LOG.info("  ... processed %d storm-rows", n_done)
    finally:
        data_registry.close_cache(cache=file_cache)

    return pd.DataFrame(records)


def classify_quintiles(*, df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["wb_metric"].notna()].copy()
    if len(valid) == 0:
        return valid

    q20 = valid["wb_metric"].quantile(0.20)
    q80 = valid["wb_metric"].quantile(0.80)

    valid["wb_group"] = "midwb"
    valid.loc[valid["wb_metric"] <= q20, "wb_group"] = "lowwb"
    valid.loc[valid["wb_metric"] >= q80, "wb_group"] = "highwb"

    for bsn in valid["basin"].unique():
        sub = valid[valid["basin"] == bsn]
        _LOG.info("  %s: total=%d, high=%d, mid=%d, low=%d",
                  bsn, len(sub),
                  len(sub[sub["wb_group"] == "highwb"]),
                  len(sub[sub["wb_group"] == "midwb"]),
                  len(sub[sub["wb_group"] == "lowwb"]))

    return valid


def main(
        data_config: Annotated[Path, typer.Option(
            help="JSON file mapping data-source keys to root directories")],
        tracks_directory: Annotated[Path, typer.Option(
            help="Directory with recurving_nh_tracks.csv and individual/")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the classification CSV")],
        basins: Annotated[Optional[list[str]], typer.Option()] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature, et, or both")] = "both",
        output_csv: Annotated[str, typer.Option()] = "storm_wb_classification.csv",
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())

    cfg = data_registry.load_data_config(path=data_config)
    data_registry.register_all(data_config=cfg)

    if basins is None:
        basins = ["WP", "NA"]

    os.makedirs(output_directory, exist_ok=True)

    print(f"Tracks: {tracks_directory}", flush=True)
    storms_df, full_tracks = tracks.load_track_database(
        tracks_directory=tracks_directory)
    print(f"Loaded {len(storms_df)} storms from index", flush=True)

    refs = (["recurvature", "et"] if reference == "both" else [reference])

    all_dfs = []
    for ref in refs:
        print(f"\n{'=' * 60}")
        print(f"  Computing WB metrics for {ref}, basins: {basins}")
        print(f"{'=' * 60}")
        df = compute_all_metrics(
            storms_df=storms_df, full_tracks=full_tracks,
            basins=list(basins), reference=ref)
        valid = df.dropna(subset=["wb_metric"])
        print(f"  Total storms with valid metric: {len(valid)}")
        if len(valid) == 0:
            continue
        print(f"  WB metric: mean={valid['wb_metric'].mean():.4f}, "
              f"median={valid['wb_metric'].median():.4f}, "
              f"Q20={valid['wb_metric'].quantile(0.20):.4f}, "
              f"Q80={valid['wb_metric'].quantile(0.80):.4f}")

        classified = classify_quintiles(df=valid)
        all_dfs.append(classified)

    if not all_dfs:
        raise SystemExit("No valid WB metrics; check tracks and ERA5 RWB paths.")

    result = pd.concat(all_dfs, ignore_index=True)
    outpath = os.path.join(output_directory, output_csv)
    result.to_csv(outpath, index=False)
    print(f"\nSaved: {outpath} ({len(result)} rows)")
    print(f"  highwb: {len(result[result['wb_group'] == 'highwb'])}")
    print(f"  midwb:  {len(result[result['wb_group'] == 'midwb'])}")
    print(f"  lowwb:  {len(result[result['wb_group'] == 'lowwb'])}")


if __name__ == "__main__":
    typer.run(main)
