"""Data-source registry: per-variable file finders and 2-D snapshot loaders, with all
root directories supplied through a user JSON data-config (see data_config.example.json)."""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Callable, Mapping

import netCDF4 as nc
import numpy as np

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.grid_utils as grid_utils

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class DataSource:
    key: str
    long_name: str
    source: str
    category: str
    file_finder: Callable[[int, int], str | None]
    nc_var: str
    time_encoding: str
    native_grid: grid_utils.GridSpec
    units: str
    needs_regrid: bool = False
    level_index: int | None = None
    is_climatology: bool = False
    npz_path: str | None = None
    npz_key: str | None = None
    is_derived: bool = False
    derived_loader: Callable[..., np.ndarray | None] | None = None


REGISTRY: dict[str, DataSource] = {}


def register(*, source: DataSource) -> None:
    REGISTRY[source.key] = source


def get(*, key: str) -> DataSource | None:
    return REGISTRY.get(key)


def list_keys(*, category: str | None = None, source: str | None = None) -> list[str]:
    out = []
    for k, ds in REGISTRY.items():
        if category and ds.category != category:
            continue
        if source and ds.source != source:
            continue
        out.append(k)
    return sorted(out)


def load_data_config(*, path: Path) -> dict[str, str]:
    with open(path) as fh:
        cfg = json.load(fh)
    return {str(k): str(v) for k, v in cfg.items()}


def _find_era5_baro(*, root: str | None, suffix: str) -> Callable[[int, int], str | None]:
    def finder(year: int, month: int) -> str | None:
        if root is None:
            return None
        path = f"{root}/{year}_{suffix}.nc"
        return path if os.path.exists(path) else None
    return finder


def _find_merra2_baro(*, root: str | None, suffix: str) -> Callable[[int, int], str | None]:
    def finder(year: int, month: int) -> str | None:
        if root is None:
            return None
        path = f"{root}/{year}_{month:02d}_{suffix}.nc"
        return path if os.path.exists(path) else None
    return finder


def _find_merra2_heat_fortran(*, root: str | None,
                              tdt_var: str) -> Callable[[int, int], str | None]:
    def finder(year: int, month: int) -> str | None:
        if root is None:
            return None
        path = f"{root}/{year}/{year}_{month:02d}_LWAb_{tdt_var}_N.nc"
        return path if os.path.exists(path) else None
    return finder


def _find_merra2_heat_sandro(*, root: str | None,
                             tdt_var: str) -> Callable[[int, int], str | None]:
    def finder(year: int, month: int) -> str | None:
        if root is None:
            return None
        path = f"{root}/{year}/{year}_{month:02d}_ncforce_baro_{tdt_var}.nc"
        return path if os.path.exists(path) else None
    return finder


def _find_era5_lh_lwa(*, root: str | None) -> Callable[[int, int], str | None]:
    def finder(year: int, month: int) -> str | None:
        if root is None:
            return None
        path = f"{root}/{year}_{month:02d}_LWAb_N.nc"
        return path if os.path.exists(path) else None
    return finder


def _find_era5_nonqg_lwa(*, root: str | None) -> Callable[[int, int], str | None]:
    def finder(year: int, month: int) -> str | None:
        if root is None:
            return None
        path = f"{root}/{year}_{month:02d}_AOUTbaro_N.nc"
        return path if os.path.exists(path) else None
    return finder


def _find_era5_raw(*, root: str | None) -> Callable[[int, int], str | None]:
    def finder(year: int, month: int) -> str | None:
        if root is None:
            return None
        path = f"{root}/{month:02d}_{year}.6hrly.nc"
        return path if os.path.exists(path) else None
    return finder


def _find_era5_qgpv(*, root: str | None) -> Callable[[int, int], str | None]:
    def finder(year: int, month: int) -> str | None:
        if root is None:
            return None
        path = f"{root}/{year}_{month:02d}_qgpv.nc"
        return path if os.path.exists(path) else None
    return finder


def _find_era5_rwb(*, root: str | None) -> Callable[[int, int], str | None]:
    def finder(year: int, month: int) -> str | None:
        if root is None:
            return None
        path = f"{root}/rwb_masks_{year}_{month:02d}.nc"
        return path if os.path.exists(path) else None
    return finder


def _find_imerg(*, root: str | None) -> Callable[[int, int], str | None]:
    def finder(year: int, month: int) -> str | None:
        if root is None:
            return None
        path = f"{root}/IMERG_{year}_{month:02d}.nc4"
        return path if os.path.exists(path) else None
    return finder


def _find_climatology(*, npz_path: str | None) -> Callable[[int, int], str | None]:
    def finder(year: int, month: int) -> str | None:
        if npz_path is None:
            return None
        return npz_path if os.path.exists(npz_path) else None
    return finder


def load_snapshot(
        *,
        source: DataSource,
        target_dt: datetime.datetime,
        cache: dict | None = None,
) -> np.ndarray | None:
    if source.is_derived and source.derived_loader is not None:
        return source.derived_loader(target_dt, cache)

    if source.is_climatology:
        return _load_climatology_snapshot(source=source, target_dt=target_dt)

    year = target_dt.year
    month = target_dt.month
    path = source.file_finder(year, month)
    if path is None:
        return None

    day = target_dt.day
    hour = target_dt.hour
    time_idx = (day - 1) * 4 + hour // 6

    try:
        if cache is not None and path in cache:
            ds_nc = cache[path]
        else:
            ds_nc = nc.Dataset(path, "r")
            if cache is not None:
                cache[path] = ds_nc

        var = ds_nc[source.nc_var]

        if source.time_encoding == "time_x_month":
            month_idx = month - 1
            if source.level_index is not None:
                data = np.array(var[time_idx, month_idx, source.level_index, :, :])
            elif var.ndim == 4:
                data = np.array(var[time_idx, month_idx, :, :])
            elif var.ndim == 3:
                data = np.array(var[time_idx, month_idx, :])
            else:
                data = np.array(var[time_idx, month_idx])

        elif source.time_encoding == "flat":
            if source.level_index is not None:
                data = np.array(var[time_idx, source.level_index, :, :])
            elif var.ndim == 3:
                data = np.array(var[time_idx, :, :])
            elif var.ndim == 4:
                data = np.array(var[time_idx, 0, :, :])
            else:
                data = np.array(var[time_idx, :])

        elif source.time_encoding == "flat_global_plev":
            plev_idx = source.level_index
            data = np.array(var[time_idx, plev_idx, :, :])

        elif source.time_encoding == "imerg":
            data = np.array(var[time_idx, :, :]).T

        else:
            return None

        return data.astype(np.float64)

    except Exception:
        _LOG.exception("Failed reading %s from %s at %s", source.key, path, target_dt)
        return None


def _load_climatology_snapshot(
        *,
        source: DataSource,
        target_dt: datetime.datetime,
) -> np.ndarray | None:
    if source.npz_path is None or source.npz_key is None:
        return None
    try:
        with np.load(source.npz_path) as npz:
            arr = npz[source.npz_key]
            month_idx = target_dt.month - 1
            return arr[month_idx].astype(np.float64)
    except Exception:
        _LOG.exception("Failed reading climatology %s", source.npz_path)
        return None


def close_cache(*, cache: dict) -> None:
    for path, ds_nc in cache.items():
        try:
            ds_nc.close()
        except Exception:
            _LOG.exception("Failed closing cached dataset %s", path)
    cache.clear()


def register_all(*, data_config: Mapping[str, str]) -> None:
    era5_baro = data_config.get("era5_baro")
    for var_key, (suffix, nc_var) in composite_config.LWA_BUDGET_SUFFIXES.items():
        register(source=DataSource(
            key=f"era5_{var_key}",
            long_name=f"ERA5 {var_key} (column-integrated barotropic)",
            source="era5", category="lwa_budget",
            file_finder=_find_era5_baro(root=era5_baro, suffix=suffix),
            nc_var=nc_var,
            time_encoding="time_x_month",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="m/s" if var_key in ("Ub",) else "m^2/s" if var_key == "lwa" else "m^2/s^2",
        ))

    merra2_baro = data_config.get("merra2_baro")
    for var_key, (suffix, nc_var) in composite_config.LWA_BUDGET_SUFFIXES_MERRA2.items():
        register(source=DataSource(
            key=f"merra2_{var_key}",
            long_name=f"MERRA2 {var_key} (column-integrated barotropic)",
            source="merra2", category="lwa_budget",
            file_finder=_find_merra2_baro(root=merra2_baro, suffix=suffix),
            nc_var=nc_var,
            time_encoding="time_x_month",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="m/s" if var_key in ("Ub",) else "m^2/s" if var_key == "lwa" else "m^2/s^2",
        ))

    merra2_heat_fortran = data_config.get("merra2_heat_fortran")
    for tdt_var in composite_config.TDT_VARS:
        register(source=DataSource(
            key=f"merra2_heat_fortran_{tdt_var}",
            long_name=f"MERRA2 LWA heating tendency ({tdt_var}, Fortran)",
            source="merra2", category="heating_fortran",
            file_finder=_find_merra2_heat_fortran(root=merra2_heat_fortran, tdt_var=tdt_var),
            nc_var="lwa",
            time_encoding="flat",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="m^2/s",
        ))

    merra2_heat_sandro = data_config.get("merra2_heat_sandro")
    for tdt_var in composite_config.TDT_VARS:
        register(source=DataSource(
            key=f"merra2_heat_sandro_{tdt_var}",
            long_name=f"MERRA2 LWA heating tendency ({tdt_var}, sandro/falwa)",
            source="merra2", category="heating_sandro",
            file_finder=_find_merra2_heat_sandro(root=merra2_heat_sandro, tdt_var=tdt_var),
            nc_var="lwa",
            time_encoding="flat",
            native_grid=grid_utils.GRID_MERRA2_NATIVE,
            units="m^2/s",
            needs_regrid=True,
        ))

    register(source=DataSource(
        key="era5_lh_lwa",
        long_name="ERA5 latent-heating LWA source (Fortran)",
        source="era5", category="lh_lwa",
        file_finder=_find_era5_lh_lwa(root=data_config.get("era5_lh_lwa")),
        nc_var="lwa",
        time_encoding="flat",
        native_grid=grid_utils.GRID_1DEG_NH,
        units="m^2/s",
    ))

    register(source=DataSource(
        key="era5_nonqg_lwa",
        long_name="ERA5 non-QG LWA source (ageostrophic forcing)",
        source="era5", category="nonqg_lwa",
        file_finder=_find_era5_nonqg_lwa(root=data_config.get("era5_nonqg")),
        nc_var="aout_baro",
        time_encoding="flat",
        native_grid=grid_utils.GRID_1DEG_NH,
        units="m/s^2",
    ))

    era5_raw = data_config.get("era5_raw")
    for raw_var in ["u", "v", "t", "q", "w"]:
        for plev in composite_config.COMPOSITE_PRESSURE_LEVELS:
            try:
                plev_idx = composite_config.ERA5_PRESSURE_LEVELS.index(plev)
            except ValueError:
                continue
            units_map = {"u": "m/s", "v": "m/s", "t": "K", "q": "kg/kg", "w": "Pa/s"}
            register(source=DataSource(
                key=f"era5_{raw_var}_{plev}hPa",
                long_name=f"ERA5 {raw_var} at {plev} hPa",
                source="era5", category="raw",
                file_finder=_find_era5_raw(root=era5_raw),
                nc_var=raw_var,
                time_encoding="flat_global_plev",
                native_grid=grid_utils.GRID_025DEG_GLOBAL,
                units=units_map.get(raw_var, ""),
                needs_regrid=True,
                level_index=plev_idx,
            ))

    register(source=DataSource(
        key="era5_qgpv_10km",
        long_name="ERA5 QGPV at ~10 km",
        source="era5", category="qgpv",
        file_finder=_find_era5_qgpv(root=data_config.get("era5_qgpv")),
        nc_var="qgpv",
        time_encoding="flat",
        native_grid=grid_utils.GRID_1DEG_GLOBAL,
        units="PVU",
        level_index=20,
    ))

    era5_rwb = data_config.get("era5_rwb")
    for rwb_type, nc_var in [("awb", "rwb_mask_awb"), ("cwb", "rwb_mask_cwb")]:
        register(source=DataSource(
            key=f"era5_rwb_{rwb_type}",
            long_name=f"ERA5 Rossby wave breaking ({rwb_type.upper()})",
            source="era5", category="rwb",
            file_finder=_find_era5_rwb(root=era5_rwb),
            nc_var=nc_var,
            time_encoding="flat",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="binary",
        ))

    era5_cc_npz = data_config.get("era5_cc_npz")
    for cc_var in composite_config.CC_VARS:
        register(source=DataSource(
            key=f"era5_cc_{cc_var}",
            long_name=f"ERA5 carrying capacity {cc_var} (monthly clim)",
            source="era5", category="carrying_capacity",
            file_finder=_find_climatology(npz_path=era5_cc_npz),
            nc_var="",
            time_encoding="climatology",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="m^2/s^2" if cc_var == "Fc" else "",
            is_climatology=True,
            npz_path=era5_cc_npz,
            npz_key=cc_var,
        ))

    merra2_cc_npz = data_config.get("merra2_cc_npz")
    for cc_var in composite_config.CC_VARS:
        register(source=DataSource(
            key=f"merra2_cc_{cc_var}",
            long_name=f"MERRA2 carrying capacity {cc_var} (monthly clim)",
            source="merra2", category="carrying_capacity",
            file_finder=_find_climatology(npz_path=merra2_cc_npz),
            nc_var="",
            time_encoding="climatology",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="m^2/s^2" if cc_var == "Fc" else "",
            is_climatology=True,
            npz_path=merra2_cc_npz,
            npz_key=cc_var,
        ))

    register(source=DataSource(
        key="imerg_precip",
        long_name="IMERG precipitation rate",
        source="imerg", category="precipitation",
        file_finder=_find_imerg(root=data_config.get("imerg")),
        nc_var="precipitation",
        time_encoding="imerg",
        native_grid=grid_utils.GRID_IMERG,
        units="mm/hr",
        needs_regrid=True,
    ))

    _register_derived_budget(prefix="era5")
    _register_derived_budget(prefix="merra2")


def _make_derived_budget_loader(*, prefix: str, term: str) -> Callable[..., np.ndarray | None]:
    def loader(target_dt: datetime.datetime, cache: dict | None = None) -> np.ndarray | None:
        keys = ["ua1", "ua2", "ep1", "ep2a", "ep3a", "ep4"]
        fields = {}
        for k in keys:
            ds_src = get(key=f"{prefix}_{k}")
            if ds_src is None:
                continue
            raw = load_snapshot(source=ds_src, target_dt=target_dt, cache=cache)
            if raw is not None:
                fields[k] = raw
        if "ua1" not in fields:
            return None
        if term == "termI":
            return grid_utils.compute_budget_termI_global(
                ua1=fields["ua1"], ua2=fields.get("ua2"), ep1=fields.get("ep1"))
        if term == "termII":
            return grid_utils.compute_budget_termII_global(
                ep2a=fields.get("ep2a"), ep3a=fields.get("ep3a"))
        if term == "termIII":
            return fields.get("ep4")
        return None
    return loader


def _register_derived_budget(*, prefix: str) -> None:
    for term, long_name in [
        ("termI", "zonal flux convergence -dF/dx"),
        ("termII", "meridional flux (ep2a-ep3a)/(2a cos phi dlam)"),
        ("termIII", "non-conservative (ep4)"),
    ]:
        register(source=DataSource(
            key=f"{prefix}_budget_{term}",
            long_name=f"{prefix.upper()} LWA budget {long_name}",
            source=prefix, category="lwa_budget_derived",
            file_finder=lambda y, m: "derived",
            nc_var="",
            time_encoding="derived",
            native_grid=grid_utils.GRID_1DEG_NH,
            units="m/s/s",
            is_derived=True,
            derived_loader=_make_derived_budget_loader(prefix=prefix, term=term),
        ))
