"""Load MPAS WPNA composite or merge WP + NA with storm-count weights;
alias MPAS composite keys to the ERA5 names compute_budget expects."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

import downstream_et_lwa.composites.mpas_composites as mpas_composites

_LOG = logging.getLogger(__name__)

MPAS_COMPOSITE_FILENAME = "composite_2d_{reference}_mpas_{scenario}_{basin}.nc"


def load_mpas_wpna_pooled(
        *,
        composite_dir: Path,
        reference: str,
        scenario: str,
) -> dict[str, Any]:
    composite_dir = Path(composite_dir)
    prefix = f"mpas_{scenario}"
    p_pool = composite_dir / MPAS_COMPOSITE_FILENAME.format(
        reference=reference, scenario=scenario, basin="WPNA")
    if p_pool.is_file():
        return mpas_composites.load_mpas_composite(path=p_pool, prefix=prefix)
    p_wp = composite_dir / MPAS_COMPOSITE_FILENAME.format(
        reference=reference, scenario=scenario, basin="WP")
    p_na = composite_dir / MPAS_COMPOSITE_FILENAME.format(
        reference=reference, scenario=scenario, basin="NA")
    if not p_wp.is_file() or not p_na.is_file():
        raise FileNotFoundError(
            f"Need either {p_pool.name} or both {p_wp.name} and {p_na.name} "
            f"under {composite_dir}")
    return _merge_wp_na(
        a=mpas_composites.load_mpas_composite(path=p_wp, prefix=prefix),
        b=mpas_composites.load_mpas_composite(path=p_na, prefix=prefix))


def alias_mpas_to_era5_keys(*, data_mp: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: v for k, v in data_mp.items()
                           if str(k).startswith("_")}
    pairs = [
        ("lwa", "era5_lwa"),
        ("budget_termI", "era5_budget_termI"),
        ("budget_termII", "era5_budget_termII"),
        ("budget_termIII", "era5_budget_termIII"),
        ("mpas_lh_lwa", "era5_lh_lwa"),
        ("mpas_nonqg_lwa", "era5_nonqg_lwa"),
        ("mpas_dadt", "era5_dadt"),
        ("mpas_residual", "era5_residual"),
        ("dadt", "era5_dadt"),
        ("residual", "era5_residual"),
    ]
    for src, dst in pairs:
        if src in data_mp:
            out[dst] = data_mp[src]
        s1, s2 = f"{src}__sumsq", f"{dst}__sumsq"
        if s1 in data_mp:
            out[s2] = data_mp[s1]
        c1, c2 = f"{src}__count_field", f"{dst}__count_field"
        if c1 in data_mp:
            out[c2] = data_mp[c1]
    for k in (
        "mpas_cc_Fc",
        "mpas_rwb_awb",
        "mpas_rwb_cwb",
        "mpas_precip",
        "mpas_qgpv_10km",
    ):
        if k in data_mp:
            out[k] = data_mp[k]
        for suf in ("__sumsq", "__count_field"):
            ks = f"{k}{suf}"
            if ks in data_mp:
                out[ks] = data_mp[ks]
    for src, dst in (("ua1", "era5_ua1"), ("ua2", "era5_ua2"),
                     ("ep1", "era5_ep1"), ("ep2a", "era5_ep2a"),
                     ("ep3a", "era5_ep3a")):
        if src in data_mp:
            out[dst] = data_mp[src]
    return out


def _merge_wp_na(*, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    na = int(a.get("_n_storms", 0))
    nb = int(b.get("_n_storms", 0))
    ntot = na + nb
    if ntot <= 0:
        raise ValueError("WP+NA merge: zero storms")
    out: dict[str, Any] = {}
    for k, va in a.items():
        if k == "_n_storms":
            continue
        vb = b.get(k)
        if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
            if va.shape != vb.shape:
                out[k] = va.copy()
                continue
            if k.endswith("__sumsq"):
                out[k] = (va.astype(np.float64)
                          + vb.astype(np.float64)).astype(np.float32)
            elif k.endswith("__count_field"):
                out[k] = (va.astype(np.float64)
                          + vb.astype(np.float64)).astype(np.float32)
            elif k.endswith("__count"):
                out[k] = (va.astype(np.float64)
                          + vb.astype(np.float64)).astype(np.float32)
            else:
                out[k] = (
                    (na * va.astype(np.float64) + nb * vb.astype(np.float64))
                    / float(ntot)
                ).astype(np.float32)
        elif isinstance(va, np.ndarray):
            out[k] = va.copy()
        elif (isinstance(va, (float, np.floating))
              and isinstance(vb, (float, np.floating))):
            out[k] = (na * float(va) + nb * float(vb)) / float(ntot)
        else:
            out[k] = va
    out["_n_storms"] = ntot
    return out
