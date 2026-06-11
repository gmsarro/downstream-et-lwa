"""CLI: 2-D Monte Carlo significance envelope for recurv-anchored composite maps."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import netCDF4 as nc
import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.composites.engine as engine
import downstream_et_lwa.composites.montecarlo as montecarlo
import downstream_et_lwa.data_registry as data_registry
import downstream_et_lwa.tracks as tracks

_LOG = logging.getLogger(__name__)

DEFAULT_VARS = [
    "era5_lwa",
    "era5_budget_termI", "era5_budget_termII", "era5_budget_termIII",
    "era5_lh_lwa",
    "era5_cc_Fc",
    "era5_rwb_awb", "era5_rwb_cwb",
    "imerg_precip",
    "merra2_heat_fortran_DTDTMST",
]


def _save_envelope_nc(
        *,
        path: str,
        p5_dict: dict[tuple[str, str], np.ndarray],
        p95_dict: dict[tuple[str, str], np.ndarray],
        lat: np.ndarray,
        lon: np.ndarray,
        t_window: tuple[float, float],
        baseline_window: tuple[float, float],
        var_quantity_resolved: dict[str, str],
        n_iter: int,
        day_offset: tuple[int, int],
        seed: int,
        n_storms_per_basin: dict[str, int],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("rel_lat", len(lat))
        ds.createDimension("rel_lon", len(lon))
        ds.createVariable("rel_lat", "f4", ("rel_lat",))[:] = lat
        ds.createVariable("rel_lon", "f4", ("rel_lon",))[:] = lon
        ds.t_window_h = list(t_window)
        ds.baseline_window_h = list(baseline_window)
        ds.n_iter = int(n_iter)
        ds.day_offset_low = int(day_offset[0])
        ds.day_offset_high = int(day_offset[1])
        ds.seed = int(seed)
        ds.note = (
            "QJ16-style Monte Carlo null for the 2-D recurv-anchored "
            "composite maps; spatial center FIXED at each storm's "
            "recurvature lat/lon, recurvature time replaced by "
            "recurv_time.replace(year=random) + uniform day_offset."
        )

        for (basin, vk), p5 in p5_dict.items():
            p95 = p95_dict[(basin, vk)]
            qname = f"{basin}__{vk}__p5"
            v = ds.createVariable(qname, "f4", ("rel_lat", "rel_lon"))
            v[:] = p5
            v.quantity = var_quantity_resolved.get(vk, "absolute")
            v.basin = basin
            v.var_key = vk
            qname = f"{basin}__{vk}__p95"
            v = ds.createVariable(qname, "f4", ("rel_lat", "rel_lon"))
            v[:] = p95
            v.quantity = var_quantity_resolved.get(vk, "absolute")
            v.basin = basin
            v.var_key = vk

        for b, n in n_storms_per_basin.items():
            ds.setncattr(f"n_storms_{b}", int(n))


def main(
        data_config: Annotated[Path, typer.Option(
            help="JSON file mapping data-source keys to root directories")],
        tracks_directory: Annotated[Path, typer.Option(
            help="Directory with recurving_nh_tracks.csv and individual/")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for envelope NetCDF output")],
        basins: Annotated[Optional[list[str]], typer.Option(
            help="Basins (use WPNA to pool WP+NA into one)")] = None,
        reference: Annotated[str, typer.Option(
            help="recurvature or et")] = "recurvature",
        variables: Annotated[Optional[list[str]], typer.Option(
            help="Variable keys (default: Fig. 6 panel set)")] = None,
        n_iter: Annotated[int, typer.Option()] = 400,
        workers: Annotated[int, typer.Option()] = 24,
        seed: Annotated[int, typer.Option()] = 42,
        day_offset: Annotated[Optional[list[int]], typer.Option(
            help="Random day offset range (QJ16 default -7 7)")] = None,
        t_window_h: Annotated[Optional[list[int]], typer.Option(
            help="Panel time-mean window in hours from reference")] = None,
        baseline_window_h: Annotated[Optional[list[int]], typer.Option(
            help="Pre-event baseline window for anomaly panels")] = None,
        filter_csv: Annotated[Optional[Path], typer.Option(
            help="Optional CSV to filter the storm sample")] = None,
        group: Annotated[Optional[str], typer.Option(
            help="Group column value to keep from --filter-csv")] = None,
        out_tag: Annotated[str, typer.Option(
            help="Optional suffix for the output NetCDF name")] = "",
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())

    cfg = data_registry.load_data_config(path=data_config)
    data_registry.register_all(data_config=cfg)

    if basins is None:
        basins = ["WP", "NA"]
    if day_offset is None:
        day_offset = [-7, 7]
    if t_window_h is None:
        t_window_h = [0, 144]
    if baseline_window_h is None:
        baseline_window_h = [-48, -12]

    if variables is None:
        var_keys = [v for v in DEFAULT_VARS
                    if data_registry.get(key=v) is not None]
    else:
        var_keys = [v for v in variables
                    if data_registry.get(key=v) is not None]
    if not var_keys:
        raise SystemExit("No registered variables in --variables")

    storms_df, full_tracks = tracks.load_track_database(
        tracks_directory=tracks_directory)
    print(f"Loaded {len(storms_df)} storms from {tracks_directory}")

    if filter_csv:
        filt = pd.read_csv(filter_csv, keep_default_na=False, na_values=[""])
        if group:
            filt = (filt[filt["wb_group"] == group]
                    if "wb_group" in filt.columns
                    else filt[filt.iloc[:, -1] == group])
        keep_ids = set(filt["storm_id"])
        storms_df = storms_df[storms_df["storm_id"].isin(keep_ids)]
        print(f"  filtered to {len(storms_df)} storms via {filter_csv}"
              f" group={group!r}")

    out = montecarlo.build_mc_envelope_2d(
        storms_df=storms_df, full_tracks=full_tracks, var_keys=var_keys,
        basins=list(basins),
        t_window=(float(t_window_h[0]), float(t_window_h[1])),
        baseline_window=(float(baseline_window_h[0]), float(baseline_window_h[1])),
        n_iter=n_iter, n_workers=workers, seed=seed,
        day_offset_range=(int(day_offset[0]), int(day_offset[1])),
        reference=reference,
    )

    p5_dict: dict[tuple[str, str], np.ndarray] = {}
    p95_dict: dict[tuple[str, str], np.ndarray] = {}
    for key, iter_maps in out.items():
        p5, p95 = montecarlo.percentile_envelope(
            iter_maps=iter_maps, p_low=5.0, p_high=95.0)
        p5_dict[key] = p5
        p95_dict[key] = p95

    quant_map = {vk: montecarlo.panel_quantity(var_key=vk)
                 for vk in set(k[1] for k in p5_dict.keys())}

    rel_lat = np.arange(-composite_config.BOX_LAT_HALF,
                        composite_config.BOX_LAT_HALF + 1, dtype=np.float32)
    rel_lon = np.arange(-composite_config.BOX_LON_WEST,
                        composite_config.BOX_LON_EAST + 1, dtype=np.float32)

    acc_basins, filt_basins, pool_wpna = engine.composite_accumulator_basins(
        basins=list(basins))
    n_per = {}
    for b in acc_basins:
        if pool_wpna:
            n_per[b] = int(storms_df["basin"].isin(filt_basins).sum())
        else:
            n_per[b] = int((storms_df["basin"] == b).sum())

    for b in acc_basins:
        sub_p5 = {k: v for k, v in p5_dict.items() if k[0] == b}
        sub_p95 = {k: v for k, v in p95_dict.items() if k[0] == b}
        tag = (f"_{out_tag}" if out_tag and not out_tag.startswith("_")
               else out_tag)
        out_nc = os.path.join(
            output_directory,
            f"mc_2d_envelope_{reference}_{b}{tag}.nc")
        _save_envelope_nc(
            path=out_nc, p5_dict=sub_p5, p95_dict=sub_p95,
            lat=rel_lat, lon=rel_lon,
            t_window=(float(t_window_h[0]), float(t_window_h[1])),
            baseline_window=(float(baseline_window_h[0]),
                             float(baseline_window_h[1])),
            var_quantity_resolved=quant_map,
            n_iter=n_iter,
            day_offset=(int(day_offset[0]), int(day_offset[1])),
            seed=seed,
            n_storms_per_basin={b: n_per[b]},
        )
        print(f"[ok] wrote {out_nc} ({len(sub_p5)} vars, basin {b}, "
              f"N={n_per[b]})")


if __name__ == "__main__":
    typer.run(main)
