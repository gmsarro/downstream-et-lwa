"""Grid specs, regridding, storm-relative patch extraction, and LWA budget term derivations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.interpolate
import scipy.ndimage

import downstream_et_lwa.composite_config as composite_config

if TYPE_CHECKING:
    import downstream_et_lwa.data_registry as data_registry


@dataclasses.dataclass
class GridSpec:
    nlat: int
    nlon: int
    lat_start: float
    lat_end: float
    lon_start: float
    lon_end: float
    nh_only: bool = True

    @property
    def lat(self) -> np.ndarray:
        return np.linspace(self.lat_start, self.lat_end, self.nlat)

    @property
    def lon(self) -> np.ndarray:
        return np.linspace(self.lon_start, self.lon_end, self.nlon)


GRID_1DEG_NH = GridSpec(91, 360, 0, 90, 0, 359, nh_only=True)
GRID_1DEG_GLOBAL = GridSpec(181, 360, -90, 90, 0, 359, nh_only=False)
GRID_025DEG_GLOBAL = GridSpec(721, 1440, 90, -90, 0, 359.75, nh_only=False)
GRID_MERRA2_NATIVE = GridSpec(361, 576, -90, 90, 0, 359.375, nh_only=False)
GRID_IMERG = GridSpec(180, 360, -89.5, 89.5, -179.5, 179.5, nh_only=False)

MPAS_DESEAM_CATEGORIES = frozenset(
    {"lwa_budget", "lwa_budget_derived", "lh_lwa", "nonqg_lwa"}
)


def regrid_to_1deg_nh(*, data: np.ndarray | None, src_grid: GridSpec) -> np.ndarray | None:
    if data is None:
        return None

    src_lat = src_grid.lat
    src_lon = src_grid.lon

    tgt_lat = GRID_1DEG_NH.lat
    tgt_lon = GRID_1DEG_NH.lon

    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        data = data[::-1, :]

    if not src_grid.nh_only and src_lat[-1] >= 90:
        nh_mask = src_lat >= -1
        src_lat = src_lat[nh_mask]
        data = data[nh_mask, :]

    src_lon_ext = np.concatenate([src_lon, [src_lon[0] + 360]])
    data_ext = np.concatenate([data, data[:, :1]], axis=1)

    mask = ~np.isfinite(data_ext)
    data_clean = np.where(mask, 0.0, data_ext)

    interp = scipy.interpolate.RegularGridInterpolator(
        (src_lat, src_lon_ext), data_clean,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    tgt_lons_query = tgt_lon.copy()
    if src_lon[0] < 0:
        tgt_lons_query = np.where(tgt_lons_query > 180, tgt_lons_query - 360,
                                  tgt_lons_query)
    else:
        tgt_lons_query = tgt_lons_query % 360
    mesh_lat, mesh_lon = np.meshgrid(tgt_lat, tgt_lons_query, indexing="ij")
    pts = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])

    result = interp(pts).reshape(GRID_1DEG_NH.nlat, GRID_1DEG_NH.nlon)
    return result


def regrid_curvilinear_to_1deg_nh(
        *,
        data: np.ndarray | None,
        lat_1d: np.ndarray,
        lon_1d: np.ndarray,
) -> np.ndarray | None:
    if data is None or data.ndim != 2:
        return None
    src_lat = np.asarray(lat_1d, dtype=np.float64)
    src_lon = np.asarray(lon_1d, dtype=np.float64)
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        data = data[::-1, :].copy()
    tgt_lat = GRID_1DEG_NH.lat
    tgt_lon = GRID_1DEG_NH.lon
    src_lon_ext = np.concatenate([src_lon, [src_lon[0] + 360]])
    data_ext = np.concatenate([data, data[:, :1]], axis=1)
    mask = ~np.isfinite(data_ext)
    data_clean = np.where(mask, 0.0, data_ext)
    interp = scipy.interpolate.RegularGridInterpolator(
        (src_lat, src_lon_ext), data_clean,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    tgt_lons_query = tgt_lon.copy()
    if src_lon[0] < 0:
        tgt_lons_query = np.where(tgt_lons_query > 180, tgt_lons_query - 360,
                                  tgt_lons_query)
    else:
        tgt_lons_query = tgt_lons_query % 360
    mesh_lat, mesh_lon = np.meshgrid(tgt_lat, tgt_lons_query, indexing="ij")
    pts = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])
    return interp(pts).reshape(GRID_1DEG_NH.nlat, GRID_1DEG_NH.nlon)


def extract_nh_from_global(*, data: np.ndarray | None, src_grid: GridSpec) -> np.ndarray | None:
    if data is None:
        return None
    src_lat = src_grid.lat
    nh_mask = src_lat >= 0
    out = data[nh_mask, :]
    if out.shape[0] != 91:
        return regrid_to_1deg_nh(data=data, src_grid=src_grid)
    return out


def deseam_longitude(
        *,
        field: np.ndarray | None,
        half_width: int = 4,
        seam_lons: tuple[float, ...] = (0.0, 180.0),
) -> np.ndarray | None:
    if field is None:
        return field
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim == 1:
        f2 = arr[np.newaxis, :]
    elif arr.ndim == 2:
        f2 = arr
    else:
        return field

    nlon = f2.shape[1]
    out = f2.copy()
    w = int(half_width)
    span = 2 * w + 2

    for sl in seam_lons:
        c = int(round((float(sl) / 360.0) * nlon)) % nlon
        left = (c - w - 1) % nlon
        right = (c + w + 1) % nlon
        a = out[:, left]
        b = out[:, right]
        ok = np.isfinite(a) & np.isfinite(b)
        for k in range(-w, w + 1):
            col = (c + k) % nlon
            frac = (k + w + 1) / span
            interp = a + (b - a) * frac
            out[ok, col] = interp[ok]

    return out.reshape(arr.shape)


def prepare_field(
        *,
        data: np.ndarray | None,
        source: data_registry.DataSource,
) -> np.ndarray | None:
    if data is None:
        return None

    if source.is_derived and data.shape == (composite_config.NLAT, composite_config.NLON):
        out: np.ndarray | None = data.copy()
        if source.category == "carrying_capacity":
            out = postprocess_carrying_capacity_nh(field=out)
        return _maybe_deseam_mpas(out=out, source=source)

    grid = source.native_grid

    if source.needs_regrid:
        out = regrid_to_1deg_nh(data=data, src_grid=grid)
    elif grid == GRID_1DEG_GLOBAL or (grid.nlat == 181 and grid.nlon == 360):
        out = extract_nh_from_global(data=data, src_grid=grid)
    elif data.shape == (composite_config.NLAT, composite_config.NLON):
        out = data.copy()
    elif data.ndim == 1 and len(data) == composite_config.NLAT:
        out = np.broadcast_to(
            data[:, np.newaxis], (composite_config.NLAT, composite_config.NLON)).copy()
    else:
        out = data

    if out is None:
        return None

    if source.category == "carrying_capacity":
        out = postprocess_carrying_capacity_nh(field=out)

    return _maybe_deseam_mpas(out=out, source=source)


def _maybe_deseam_mpas(*, out: np.ndarray | None, source: Any) -> np.ndarray | None:
    if out is None:
        return None
    src = str(getattr(source, "source", "") or "")
    cat = getattr(source, "category", "")
    if (src.startswith("mpas") and cat in MPAS_DESEAM_CATEGORIES
            and isinstance(out, np.ndarray)
            and out.shape == (composite_config.NLAT, composite_config.NLON)):
        return deseam_longitude(field=out)
    return out


def postprocess_carrying_capacity_nh(*, field: np.ndarray | None) -> np.ndarray | None:
    if field is None:
        return None
    out = field.copy()
    out[:30, :] = np.nan
    out[out > 150] = np.nan
    filled = np.nan_to_num(out, nan=0.0)
    mask = np.isfinite(out).astype(float)
    sf = scipy.ndimage.gaussian_filter(filled, sigma=2)
    sm = scipy.ndimage.gaussian_filter(mask, sigma=2)
    sm[sm < 0.3] = np.nan
    return sf / sm


def extract_storm_patch(
        *,
        field_2d: np.ndarray | None,
        center_lat: float,
        center_lon: float,
        lat_half: int | None = None,
        lon_west: int | None = None,
        lon_east: int | None = None,
) -> np.ndarray | None:
    lat_half = lat_half or composite_config.BOX_LAT_HALF
    lon_west = lon_west or composite_config.BOX_LON_WEST
    lon_east = lon_east or composite_config.BOX_LON_EAST

    if field_2d is None:
        return None

    nlat, nlon = field_2d.shape

    center_lon_360 = center_lon % 360
    j_center = int(round(center_lat))
    j_start = j_center - lat_half
    j_end = j_center + lat_half + 1

    if j_start < 0 or j_end > nlat:
        out_nlat = lat_half * 2 + 1
        out_nlon = lon_west + lon_east + 1

        j_start_clamped = max(j_start, 0)
        j_end_clamped = min(j_end, nlat)

        pad_top = j_start_clamped - j_start
        pad_bot = j_end - j_end_clamped

        i_center = int(round(center_lon_360)) % nlon
        i_start = i_center - lon_west
        i_end = i_center + lon_east + 1

        lon_indices = np.arange(i_start, i_end) % nlon
        lat_slice = field_2d[j_start_clamped:j_end_clamped, :][:, lon_indices]

        result = np.full((out_nlat, out_nlon), np.nan)
        result[pad_top:out_nlat - pad_bot, :] = lat_slice
        return result

    i_center = int(round(center_lon_360)) % nlon
    i_start = i_center - lon_west
    i_end = i_center + lon_east + 1

    lon_indices = np.arange(i_start, i_end) % nlon
    patch = field_2d[j_start:j_end, :][:, lon_indices]
    return patch


def cosine_weights_2d() -> np.ndarray:
    return np.broadcast_to(
        composite_config.COSPHI[:, np.newaxis],
        (composite_config.NLAT, composite_config.NLON),
    ).copy()


def compute_zonal_gradient(*, field_2d: np.ndarray) -> np.ndarray:
    dx = (composite_config.A_EARTH * composite_config.COSPHI[:, np.newaxis]
          * composite_config.DLAMBDA)
    dx = np.broadcast_to(dx, field_2d.shape).copy()
    dx[dx < 1e-6] = np.nan

    grad = np.empty_like(field_2d)
    grad[:, 1:-1] = (field_2d[:, 2:] - field_2d[:, :-2]) / (2 * dx[:, 1:-1])
    grad[:, 0] = (field_2d[:, 1] - field_2d[:, -1]) / (2 * dx[:, 0])
    grad[:, -1] = (field_2d[:, 0] - field_2d[:, -2]) / (2 * dx[:, -1])
    return grad


def compute_budget_termI_global(
        *,
        ua1: np.ndarray,
        ua2: np.ndarray | None,
        ep1: np.ndarray | None,
) -> np.ndarray:
    F1 = ua1.copy()
    if ua2 is not None:
        F1 = F1 + ua2
    if ep1 is not None:
        F1 = F1 + ep1
    return -compute_zonal_gradient(field_2d=F1)


def compute_budget_termII_global(
        *,
        ep2a: np.ndarray | None,
        ep3a: np.ndarray | None,
) -> np.ndarray | None:
    if ep2a is None or ep3a is None:
        return None
    denom = (2.0 * composite_config.A_EARTH * composite_config.COSPHI
             * composite_config.DLAMBDA)
    denom[np.abs(denom) < 1e-6] = np.nan
    return (ep2a - ep3a) / denom[:, np.newaxis]


def compute_lwa_budget(*, patches: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    prefix = "era5_"
    if "merra2_lwa" in patches and "era5_lwa" not in patches:
        prefix = "merra2_"

    lwa = patches.get(f"{prefix}lwa")
    ua1 = patches.get(f"{prefix}ua1")
    ua2 = patches.get(f"{prefix}ua2")
    ep1 = patches.get(f"{prefix}ep1")
    ep2a = patches.get(f"{prefix}ep2a")
    ep3a = patches.get(f"{prefix}ep3a")
    ep4 = patches.get(f"{prefix}ep4")

    if lwa is None or ua1 is None:
        return None

    nlat, nlon, nlags = lwa.shape
    dt = composite_config.DT_SEC

    rel_lat = np.arange(-composite_config.BOX_LAT_HALF,
                        composite_config.BOX_LAT_HALF + 1, dtype=np.float64)

    ref_idx = int(np.argmin(np.abs(composite_config.LAG_HOURS)))
    lwa_ref = lwa[:, :, ref_idx:ref_idx + 1]
    tendency_integrated = lwa - lwa_ref

    F1 = ua1 + ua2 + ep1
    dx_patch = _patch_dx(rel_lat=rel_lat)[:, np.newaxis]
    flux_conv_inst = np.full_like(F1, np.nan)
    for t in range(nlags):
        f = F1[:, :, t]
        if np.any(np.isfinite(f)):
            grad = np.empty_like(f)
            grad[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dx_patch)
            dx_1d = dx_patch.ravel()[:, np.newaxis]
            grad[:, 0:1] = (f[:, 1:2] - f[:, -1:]) / (2 * dx_1d)
            grad[:, -1:] = (f[:, 0:1] - f[:, -2:-1]) / (2 * dx_1d)
            flux_conv_inst[:, :, t] = -grad
    flux_conv = _cumulative_integrate(field_3d=flux_conv_inst, dt=dt, ref_idx=ref_idx)

    if ep2a is not None and ep3a is not None:
        abs_lat = 35.0 + rel_lat
        merid_factor = (2 * composite_config.A_EARTH * np.cos(np.deg2rad(abs_lat))
                        * composite_config.DLAMBDA)
        merid_factor[np.abs(merid_factor) < 1e-6] = np.nan
        merid_3d = merid_factor[:, np.newaxis, np.newaxis]
        merid_inst = (ep2a - ep3a) / merid_3d
        meridional = _cumulative_integrate(field_3d=merid_inst, dt=dt, ref_idx=ref_idx)
    else:
        meridional = np.full_like(lwa, np.nan)

    if ep4 is not None:
        ep4_integrated = _cumulative_integrate(field_3d=ep4, dt=dt, ref_idx=ref_idx)
    else:
        ep4_integrated = np.full_like(lwa, np.nan)

    residual = tendency_integrated - flux_conv - meridional - ep4_integrated

    return {
        "lwa": lwa,
        "tendency": tendency_integrated,
        "flux_conv": flux_conv,
        "meridional": meridional,
        "ep4": ep4_integrated,
        "residual": residual,
    }


def _patch_dx(*, rel_lat: np.ndarray) -> np.ndarray:
    abs_lat = 35.0 + rel_lat
    cosphi = np.cos(np.deg2rad(abs_lat))
    cosphi[cosphi < 0.01] = np.nan
    return composite_config.A_EARTH * cosphi * composite_config.DLAMBDA


def _cumulative_integrate(*, field_3d: np.ndarray, dt: float, ref_idx: int) -> np.ndarray:
    nlat, nlon, nlags = field_3d.shape
    result = np.full_like(field_3d, np.nan)
    result[:, :, ref_idx] = 0.0

    for t in range(ref_idx + 1, nlags):
        prev = result[:, :, t - 1]
        curr_rate = field_3d[:, :, t]
        prev_rate = field_3d[:, :, t - 1]
        both_valid = np.isfinite(prev) & np.isfinite(curr_rate) & np.isfinite(prev_rate)
        result[:, :, t] = np.where(
            both_valid,
            prev + 0.5 * (prev_rate + curr_rate) * dt,
            np.nan,
        )

    for t in range(ref_idx - 1, -1, -1):
        nxt = result[:, :, t + 1]
        curr_rate = field_3d[:, :, t]
        nxt_rate = field_3d[:, :, t + 1]
        both_valid = np.isfinite(nxt) & np.isfinite(curr_rate) & np.isfinite(nxt_rate)
        result[:, :, t] = np.where(
            both_valid,
            nxt - 0.5 * (curr_rate + nxt_rate) * dt,
            np.nan,
        )

    return result
