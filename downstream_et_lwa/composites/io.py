"""NetCDF read/write for composite cubes and the NaN-aware composite-volume Gaussian."""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import netCDF4 as nc
import numpy as np
import scipy.ndimage

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.data_registry as data_registry

if TYPE_CHECKING:
    import downstream_et_lwa.composites.engine as engine

_LOG = logging.getLogger(__name__)


def smooth_composite_volume(
        *,
        field_3d: np.ndarray | None,
        sigma_3d: tuple[float, ...] | None = None,
) -> np.ndarray | None:
    if field_3d is None:
        return None
    raw_sig = (sigma_3d if sigma_3d is not None
               else composite_config.COMPOSITE_PREMEAN_VOLUME_SIGMA_3D)
    sig = tuple(float(x) for x in raw_sig)
    if all(s <= 0.0 for s in sig):
        return field_3d
    filled = np.nan_to_num(field_3d, nan=0.0)
    mask = np.isfinite(field_3d).astype(np.float64)
    sf = scipy.ndimage.gaussian_filter(filled, sigma=sig)
    sm = scipy.ndimage.gaussian_filter(mask, sigma=sig)
    return sf / np.where(sm > 0.3, sm, np.nan)


def save_composites(
        *,
        accum: engine.CompositeAccumulator,
        reference: str,
        output_directory: Path,
        group: str | None = None,
        bake_volume_smoothing: bool = False,
        volume_sigma_3d: tuple[float, ...] | None = None,
        premean_volume_sigma_3d: tuple[float, ...] | None = None,
) -> None:
    os.makedirs(output_directory, exist_ok=True)

    for basin in accum.basins:
        n_storms = len(accum.metadata[basin])
        if n_storms == 0:
            continue

        fname = f"composite_2d_{reference}_{basin}.nc"
        if group:
            fname = f"composite_2d_{reference}_{basin}_{group}.nc"
        path = os.path.join(output_directory, fname)

        with nc.Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("rel_lat", composite_config.BOX_NLAT)
            ds.createDimension("rel_lon", composite_config.BOX_NLON)
            ds.createDimension("lag_hours", composite_config.N_LAGS)
            ds.createDimension("storm", n_storms)
            ds.createDimension("char_len", 64)

            v = ds.createVariable("rel_lat", "f4", ("rel_lat",))
            v[:] = np.arange(-composite_config.BOX_LAT_HALF,
                             composite_config.BOX_LAT_HALF + 1, dtype=np.float32)
            v.units = "degrees_north"
            v.long_name = "latitude relative to storm center"

            v = ds.createVariable("rel_lon", "f4", ("rel_lon",))
            v[:] = np.arange(-composite_config.BOX_LON_WEST,
                             composite_config.BOX_LON_EAST + 1, dtype=np.float32)
            v.units = "degrees_east"
            v.long_name = "longitude relative to storm center"

            v = ds.createVariable("lag_hours", "f4", ("lag_hours",))
            v[:] = composite_config.LAG_HOURS.astype(np.float32)
            v.units = "hours"
            v.long_name = f"hours relative to {reference} time"

            meta = accum.metadata[basin]
            storm_ids = [m["storm_id"] for m in meta]
            names = [m["name"] for m in meta]
            seasons = [m["season"] for m in meta]

            for attr_name, values in [
                ("storm_id", storm_ids), ("storm_name", names)
            ]:
                v = ds.createVariable(attr_name, "S1", ("storm", "char_len"))
                for i, s in enumerate(values):
                    for j, c in enumerate(str(s)[:64]):
                        v[i, j] = c

            v = ds.createVariable("season", "i4", ("storm",))
            v[:] = np.array(seasons, dtype=np.int32)

            for attr in ["recurv_lat", "recurv_lon", "et_lat", "et_lon"]:
                v = ds.createVariable(attr, "f4", ("storm",))
                v[:] = np.array([m[attr] for m in meta], dtype=np.float32)

            v = ds.createVariable("track_lat", "f4", ("storm", "lag_hours"))
            v[:] = np.array(accum.track_lats[basin], dtype=np.float32)

            v = ds.createVariable("track_lon", "f4", ("storm", "lag_hours"))
            v[:] = np.array(accum.track_lons[basin], dtype=np.float32)

            v = ds.createVariable("track_wind", "f4", ("storm", "lag_hours"))
            v[:] = np.array(accum.track_winds[basin], dtype=np.float32)

            v = ds.createVariable("track_pres", "f4", ("storm", "lag_hours"))
            v[:] = np.array(accum.track_pres[basin], dtype=np.float32)

            v = ds.createVariable("track_rel_lat", "f4", ("storm", "lag_hours"))
            v[:] = np.array(accum.track_rel_lats[basin], dtype=np.float32)
            v.units = "degrees"
            v.long_name = "latitude relative to fixed composite center"

            v = ds.createVariable("track_rel_lon", "f4", ("storm", "lag_hours"))
            v[:] = np.array(accum.track_rel_lons[basin], dtype=np.float32)
            v.units = "degrees"
            v.long_name = "longitude relative to fixed composite center"

            for vk in accum.var_keys:
                ds_info = data_registry.get(key=vk)
                long_name = ds_info.long_name if ds_info else vk
                units = ds_info.units if ds_info else ""

                mean = accum.get_mean(basin=basin, var_key=vk)
                count = accum.get_count(basin=basin, var_key=vk)
                sumsq = accum.get_sumsq(basin=basin, var_key=vk)
                count_field = accum.get_count_field(basin=basin, var_key=vk)

                if bake_volume_smoothing:
                    smoothed = smooth_composite_volume(
                        field_3d=mean, sigma_3d=volume_sigma_3d)
                    if smoothed is not None:
                        mean = smoothed

                v = ds.createVariable(f"{vk}_mean", "f4",
                                      ("rel_lat", "rel_lon", "lag_hours"),
                                      zlib=True, complevel=4)
                v[:] = mean.astype(np.float32)
                v.long_name = f"composite mean {long_name}"
                v.units = units

                v = ds.createVariable(f"{vk}_sumsq", "f4",
                                      ("rel_lat", "rel_lon", "lag_hours"),
                                      zlib=True, complevel=4)
                v[:] = sumsq.astype(np.float32)
                v.long_name = f"composite sum of squares {long_name}"

                v = ds.createVariable(f"{vk}_count", "i4", ("lag_hours",))
                v[:] = count
                v.long_name = f"number of storms contributing to {vk}"

                v = ds.createVariable(f"{vk}_count_field", "i4",
                                      ("rel_lat", "rel_lon", "lag_hours"),
                                      zlib=True, complevel=4)
                v[:] = count_field
                v.long_name = f"per-pixel count for {vk}"

            ds.title = f"TC composite ({reference}-relative, {basin} basin)"
            ds.reference = "Quinting and Jones (2016, MWR)"
            ds.basin = basin
            ds.reference_time = reference
            ds.creation_date = datetime.datetime.now().isoformat()
            ds.n_storms = n_storms
            if bake_volume_smoothing:
                vs = (volume_sigma_3d
                      or composite_config.COMPOSITE_PREMEAN_VOLUME_SIGMA_3D)
                ds.volume_gaussian_sigma_rel_lat_rel_lon_lag = str(vs)
            elif premean_volume_sigma_3d is not None:
                ds.premean_gaussian_sigma_rel_lat_rel_lon_lag = ",".join(
                    str(float(x)) for x in premean_volume_sigma_3d)

        _LOG.info("Saved: %s (%d storms, %d variables)",
                  path, n_storms, len(accum.var_keys))


def _read_composite_vars_into(*, ds: nc.Dataset, data: dict[str, Any]) -> None:
    for vname in ds.variables:
        if vname.endswith("_mean"):
            key = vname[:-5]
            data[key] = np.array(ds[vname][:])
        elif vname.endswith("_sumsq"):
            key = vname[:-6]
            data[f"{key}__sumsq"] = np.array(ds[vname][:])
        elif vname.endswith("_count_field"):
            key = vname[:-12]
            data[f"{key}__count_field"] = np.array(ds[vname][:])


def load_composite(
        *,
        path: Path,
        reference: str = "recurvature",
        min_track_count: int = 15,
        supplement_path: Path | None = None,
) -> dict[str, Any] | None:
    if not os.path.exists(path):
        _LOG.warning("Composite file not found: %s", path)
        return None

    data: dict[str, Any] = {}
    with nc.Dataset(str(path), "r") as ds:
        _read_composite_vars_into(ds=ds, data=data)

        if supplement_path is not None and os.path.exists(supplement_path):
            with nc.Dataset(str(supplement_path), "r") as sds:
                _read_composite_vars_into(ds=sds, data=data)
            data["_has_merra2_source"] = True
        else:
            data["_has_merra2_source"] = False

        data["_lat"] = np.array(ds["rel_lat"][:])
        data["_lon"] = np.array(ds["rel_lon"][:])
        data["_lag_hours"] = np.array(ds["lag_hours"][:])
        data["_n_storms"] = ds.dimensions["storm"].size

        if "track_rel_lat" in ds.variables:
            trl = ds["track_rel_lat"][:]
            trn = ds["track_rel_lon"][:]
            valid_per_lag = np.sum(np.isfinite(trl), axis=0)
            data["_mean_rel_lat"] = np.where(valid_per_lag >= min_track_count,
                                             np.nanmean(trl, axis=0), np.nan)
            data["_mean_rel_lon"] = np.where(valid_per_lag >= min_track_count,
                                             np.nanmean(trn, axis=0), np.nan)
            data["_track_count"] = valid_per_lag

        ref_col = "recurv_lat" if reference == "recurvature" else "et_lat"
        lon_col = "recurv_lon" if reference == "recurvature" else "et_lon"
        if ref_col in ds.variables:
            data["_mean_abs_lat"] = float(np.nanmean(ds[ref_col][:]))
            data["_mean_abs_lon"] = float(np.nanmean(ds[lon_col][:]))

    return data
