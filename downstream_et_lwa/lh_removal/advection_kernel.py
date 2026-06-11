"""2-D barotropic LWA advection kernel (Lubis & Nakamura 2024) with time-dependent forcing."""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable

import numpy as np

import downstream_et_lwa.constants as constants

_LOG = logging.getLogger(__name__)

NLAT = 91
NLON = 360
A_EARTH = constants.EARTH_RADIUS_M
DLAMBDA = np.deg2rad(1.0)
DPHI = np.deg2rad(1.0)
LATS = np.linspace(0.0, 90.0, NLAT)
LONS = np.linspace(0.0, 359.0, NLON)
COSPHI = np.cos(np.deg2rad(LATS))
COSPHI_SAFE = np.where(np.abs(COSPHI) < 1e-6, 1e-6, COSPHI)

ForcingFunc = Callable[[float], dict[str, np.ndarray]]


@dataclasses.dataclass
class IntegratorConfig:
    dt: float = 1800.0
    K: float = 2.3e4
    asselin_alpha: float = 0.05
    boundary_north: int = 5
    boundary_south: int = 6
    include_LH: bool = True
    include_S_other: bool = True
    factor_residual: float = 1.0
    scheme: str = "rk3"


def zonal_advect(*, c_x: np.ndarray, A: np.ndarray) -> np.ndarray:
    flux = c_x * A
    dflux = np.empty_like(flux)
    dflux[:, 1:-1] = flux[:, 2:] - flux[:, :-2]
    dflux[:, 0] = flux[:, 1] - flux[:, -1]
    dflux[:, -1] = flux[:, 0] - flux[:, -2]
    denom = (2.0 * A_EARTH * COSPHI_SAFE * DLAMBDA)[:, None]
    return -dflux / denom


def merid_advect(*, cyp: np.ndarray, cym: np.ndarray, A: np.ndarray) -> np.ndarray:
    out = np.zeros_like(A)
    cosp = np.cos(np.deg2rad(LATS + 1.0))
    cosm = np.cos(np.deg2rad(LATS - 1.0))
    A_jp = np.empty_like(A)
    A_jm = np.empty_like(A)
    A_jp[:-1, :] = A[1:, :]
    A_jp[-1, :] = A[-1, :]
    A_jm[1:, :] = A[:-1, :]
    A_jm[0, :] = A[0, :]
    flux_n = cyp * A_jp * cosp[:, None]
    flux_s = cym * A_jm * cosm[:, None]
    denom = (2.0 * A_EARTH * COSPHI_SAFE * DPHI)[:, None]
    out[1:-1, :] = -(flux_n[1:-1, :] - flux_s[1:-1, :]) / denom[1:-1, :]
    return out


def laplacian(*, A: np.ndarray) -> np.ndarray:
    lap = np.empty_like(A)
    A_jp = np.empty_like(A)
    A_jm = np.empty_like(A)
    A_jp[:-1, :] = A[1:, :]
    A_jp[-1, :] = A[-1, :]
    A_jm[1:, :] = A[:-1, :]
    A_jm[0, :] = A[0, :]
    A_ip = np.empty_like(A)
    A_im = np.empty_like(A)
    A_ip[:, :-1] = A[:, 1:]
    A_ip[:, -1] = A[:, 0]
    A_im[:, 1:] = A[:, :-1]
    A_im[:, 0] = A[:, -1]

    cosp = np.cos(np.deg2rad(LATS + 1.0))[:, None]
    cosm = np.cos(np.deg2rad(LATS - 1.0))[:, None]
    cos0 = COSPHI_SAFE[:, None]

    diff_x = (A_ip + A_im - 2.0 * A) / (A_EARTH * cos0 * DLAMBDA) ** 2
    diff_y = 0.5 * (
        (A_jp - A) * (cos0 + cosp)
        - (A - A_jm) * (cos0 + cosm)
    ) / (cos0 * (A_EARTH * DPHI) ** 2)
    lap[:, :] = diff_x + diff_y
    lap[0, :] = 0.0
    lap[-1, :] = 0.0
    return lap


def tendency(*, A: np.ndarray, force: dict[str, np.ndarray],
             cfg: IntegratorConfig) -> np.ndarray:
    rhs = (
        zonal_advect(c_x=force["c_x"], A=A)
        + merid_advect(cyp=force["cyp"], cym=force["cym"], A=A)
        + cfg.factor_residual * force["gamma"] * A
        + cfg.factor_residual * force["eps4"]
        + cfg.K * laplacian(A=A)
    )
    if cfg.include_S_other:
        rhs = rhs + cfg.factor_residual * force["S_other"]
    if cfg.include_LH:
        rhs = rhs + force["LH"]
    return rhs


def apply_boundary_clamp(*, A: np.ndarray, A_obs: np.ndarray,
                         cfg: IntegratorConfig) -> np.ndarray:
    if cfg.boundary_south > 0:
        A[:cfg.boundary_south, :] = A_obs[:cfg.boundary_south, :]
    if cfg.boundary_north > 0:
        A[NLAT - cfg.boundary_north:, :] = A_obs[NLAT - cfg.boundary_north:, :]
    return A


def _step_rk3(*, A: np.ndarray, t: float, dt: float,
              forcing: ForcingFunc, cfg: IntegratorConfig) -> np.ndarray:
    f0 = forcing(t)
    k1 = tendency(A=A, force=f0, cfg=cfg)
    A1 = A + dt * k1
    A1 = apply_boundary_clamp(A=A1, A_obs=f0["A_obs"], cfg=cfg)

    f1 = forcing(t + dt)
    k2 = tendency(A=A1, force=f1, cfg=cfg)
    A2 = 0.75 * A + 0.25 * (A1 + dt * k2)
    f_half = forcing(t + 0.5 * dt)
    A2 = apply_boundary_clamp(A=A2, A_obs=f_half["A_obs"], cfg=cfg)

    k3 = tendency(A=A2, force=f_half, cfg=cfg)
    A_new = (1.0 / 3.0) * A + (2.0 / 3.0) * (A2 + dt * k3)
    A_new = apply_boundary_clamp(A=A_new, A_obs=forcing(t + dt)["A_obs"], cfg=cfg)
    return A_new


def _step_rk4(*, A: np.ndarray, t: float, dt: float,
              forcing: ForcingFunc, cfg: IntegratorConfig) -> np.ndarray:
    f_t = forcing(t)
    f_h = forcing(t + 0.5 * dt)
    f_e = forcing(t + dt)
    k1 = tendency(A=A, force=f_t, cfg=cfg)
    k2 = tendency(A=A + 0.5 * dt * k1, force=f_h, cfg=cfg)
    k3 = tendency(A=A + 0.5 * dt * k2, force=f_h, cfg=cfg)
    k4 = tendency(A=A + dt * k3, force=f_e, cfg=cfg)
    A_new = A + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    A_new = apply_boundary_clamp(A=A_new, A_obs=f_e["A_obs"], cfg=cfg)
    return A_new


def _step_euler(*, A: np.ndarray, t: float, dt: float,
                forcing: ForcingFunc, cfg: IntegratorConfig) -> np.ndarray:
    f_t = forcing(t)
    A_new = A + dt * tendency(A=A, force=f_t, cfg=cfg)
    A_new = apply_boundary_clamp(A=A_new, A_obs=forcing(t + dt)["A_obs"], cfg=cfg)
    return A_new


def integrate(
    *,
    A0: np.ndarray,
    forcing: ForcingFunc,
    t_start: float,
    t_end: float,
    cfg: IntegratorConfig,
    snapshot_dt: float | None = None,
    progress_every: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    dt = float(cfg.dt)
    n_steps = max(1, int(np.round((t_end - t_start) / dt)))
    snapshot_dt = float(snapshot_dt) if snapshot_dt is not None else dt
    save_every = max(1, int(np.round(snapshot_dt / dt)))

    times = [t_start]
    A_out = [A0.copy()]

    if cfg.scheme == "leapfrog":
        f0 = forcing(t_start)
        A_prev = A0.copy()
        rhs0 = tendency(A=A_prev, force=f0, cfg=cfg)
        A_curr = A_prev + dt * rhs0
        A_curr = apply_boundary_clamp(A=A_curr, A_obs=f0["A_obs"], cfg=cfg)

        for n in range(1, n_steps + 1):
            t_now = t_start + n * dt
            f_now = forcing(t_now)
            rhs = tendency(A=A_curr, force=f_now, cfg=cfg)
            A_next = A_prev + 2.0 * dt * rhs
            A_curr_filt = A_curr + cfg.asselin_alpha * (
                A_prev - 2.0 * A_curr + A_next
            )
            A_next = apply_boundary_clamp(A=A_next, A_obs=f_now["A_obs"], cfg=cfg)
            A_prev = A_curr_filt
            A_curr = A_next
            if n % save_every == 0:
                times.append(t_now)
                A_out.append(A_curr.copy())
            if progress_every and n % progress_every == 0:
                _log_step(n=n, n_steps=n_steps, t_now=t_now, A_curr=A_curr)
    else:
        if cfg.scheme == "rk3":
            step = _step_rk3
        elif cfg.scheme == "rk4":
            step = _step_rk4
        elif cfg.scheme == "euler":
            step = _step_euler
        else:
            raise ValueError(f"unknown scheme: {cfg.scheme}")

        A_curr = A0.copy()
        for n in range(1, n_steps + 1):
            t_now_prev = t_start + (n - 1) * dt
            A_curr = step(A=A_curr, t=t_now_prev, dt=dt, forcing=forcing, cfg=cfg)
            t_now = t_start + n * dt
            if n % save_every == 0:
                times.append(t_now)
                A_out.append(A_curr.copy())
            if progress_every and n % progress_every == 0:
                _log_step(n=n, n_steps=n_steps, t_now=t_now, A_curr=A_curr)

    return np.asarray(times), np.asarray(A_out)


def _log_step(*, n: int, n_steps: int, t_now: float, A_curr: np.ndarray) -> None:
    if not np.all(np.isfinite(A_curr)):
        bad = np.argwhere(~np.isfinite(A_curr))
        _LOG.info(
            "[step %d/%d t=%.2f h] NaN/Inf at %d cells; first = lat_idx=%d lon_idx=%d",
            n, n_steps, t_now / 3600, len(bad), bad[0, 0], bad[0, 1],
        )
        return
    mn, mx, mean = float(A_curr.min()), float(A_curr.max()), float(A_curr.mean())
    jmx, imx = np.unravel_index(np.argmax(np.abs(A_curr)), A_curr.shape)
    _LOG.info(
        "[step %d/%d t=%.2f h] A min/mean/max = %.3g / %.3g / %.3g |A|max @ (%.0fN, %dE)",
        n, n_steps, t_now / 3600, mn, mean, mx, LATS[jmx], imx,
    )


def build_forcing_table(
    *,
    times: np.ndarray,
    A_obs: np.ndarray,
    c_x: np.ndarray,
    cyp: np.ndarray,
    cym: np.ndarray,
    gamma: np.ndarray,
    eps4: np.ndarray,
    S_other: np.ndarray,
    LH: np.ndarray,
) -> ForcingFunc:
    times = np.asarray(times, dtype=np.float64)
    fields = {
        "A_obs": A_obs, "c_x": c_x, "cyp": cyp, "cym": cym,
        "gamma": gamma, "eps4": eps4, "S_other": S_other, "LH": LH,
    }
    t0, t1 = times[0], times[-1]

    def _interp(t: float) -> dict[str, np.ndarray]:
        t = float(min(max(t, t0), t1))
        if t <= t0:
            j = 0
            w = 0.0
        elif t >= t1:
            j = len(times) - 2
            w = 1.0
        else:
            j = int(np.searchsorted(times, t) - 1)
            j = max(0, min(j, len(times) - 2))
            w = (t - times[j]) / max(times[j + 1] - times[j], 1e-9)
        out = {}
        for k, arr in fields.items():
            out[k] = (1.0 - w) * arr[j] + w * arr[j + 1]
        return out

    return _interp
