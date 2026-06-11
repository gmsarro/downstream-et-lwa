"""Classify recurving WP+NA TCs into RW / no-RW quintiles (Quinting & Jones 2016):
storm-relative RWP-presence masks from Hanning-filtered v250 envelopes, pooled
frequency vs a Monte Carlo null, and quintile split on the overlap score."""

from __future__ import annotations

import calendar
import concurrent.futures
import datetime
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Optional

import netCDF4 as nc
import numpy as np
import pandas as pd
import scipy.ndimage
import typer
from typing_extensions import Annotated

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.data_registry as data_registry
import downstream_et_lwa.tracks as tracks

_LOG = logging.getLogger(__name__)

RWP_THRESH = 15.0
HANN_WINDOW = 20

REL_LON_HALF = 180
REL_LON = np.arange(-REL_LON_HALF, REL_LON_HALF + 1)
NREL = REL_LON.size


def _merid_avg(
        *,
        field_2d: np.ndarray,
        lat_min: float = composite_config.HOV_LAT_MIN,
        lat_max: float = composite_config.HOV_LAT_MAX,
) -> np.ndarray:
    j0 = max(int(round(lat_min)), 0)
    j1 = min(int(round(lat_max)) + 1, field_2d.shape[0])
    strip = field_2d[j0:j1, :]
    w = composite_config.COSPHI[j0:j1, np.newaxis]
    valid = np.isfinite(strip)
    wv = np.where(valid, w, 0.0)
    ws = wv.sum(axis=0)
    ws[ws == 0] = np.nan
    return np.nansum(strip * wv, axis=0) / ws


def _preload_year(args: tuple) -> dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]]:
    import downstream_et_lwa.hovmoller as hovmoller

    yr, months, extracted_dir = args

    all_keys = []
    all_v250 = []
    for mo in months:
        path = os.path.join(extracted_dir, f"{mo:02d}_{yr}.nc")
        if not os.path.exists(path):
            continue
        with nc.Dataset(path, "r") as ds:
            v = np.array(ds["v250"][:], dtype=np.float32)
        ndays = calendar.monthrange(yr, mo)[1]
        for t_idx in range(v.shape[0]):
            day = t_idx // 4 + 1
            hr = (t_idx % 4) * 6
            if day > ndays:
                break
            all_keys.append((yr, mo, day, hr))
            all_v250.append(v[t_idx])

    if not all_keys:
        return {}

    arr = np.stack(all_v250, axis=0)

    hann = np.hanning(HANN_WINDOW)
    hann /= hann.sum()
    arr_filt = scipy.ndimage.convolve1d(arr, hann, axis=0, mode="nearest")

    results = {}
    for i, k in enumerate(all_keys):
        env = hovmoller.compute_rwp_envelope(v_field=arr_filt[i].astype(np.float64))
        env_strip = _merid_avg(field_2d=env).astype(np.float32)
        freq_strip = _merid_avg(
            field_2d=(env > RWP_THRESH).astype(float)).astype(np.float32)
        results[k] = (env_strip, freq_strip)

    return results


def preload_all_envelopes(
        *,
        extracted_directory: Path,
        years: range = range(2000, 2025),
        n_workers: int = 16,
) -> dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]]:
    _LOG.info(
        "Pre-loading v250 RWP envelopes with %.0f-day Hanning filter "
        "(%d years, %d workers)...",
        HANN_WINDOW * 6 / 24, len(list(years)), n_workers)

    months = list(range(1, 13))
    all_data: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("fork")) as pool:
        futs = {pool.submit(_preload_year,
                            (yr, months, str(extracted_directory))): yr
                for yr in years}
        for fut in futs:
            try:
                d = fut.result()
                all_data.update(d)
            except Exception:
                _LOG.exception("Year %s failed", futs[fut])

    _LOG.info("Total: %d time steps in memory (~%.0f MB)",
              len(all_data), len(all_data) * 360 * 4 * 2 / 1e6)
    return all_data


def lookup_strip(
        *,
        data_cache: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]],
        dt: datetime.datetime,
) -> tuple[np.ndarray, np.ndarray] | None:
    key = (dt.year, dt.month, dt.day, dt.hour)
    return data_cache.get(key)


def _per_storm_mask(
        *,
        data_cache: dict,
        storm: dict[str, Any],
        reference: str,
) -> np.ndarray | None:
    rt = pd.Timestamp(
        storm["recurv_time"] if reference == "recurvature" else storm["et_time"])
    rl = (storm["recurv_lon"] if reference == "recurvature"
          else storm["et_lon"])
    if pd.isna(rt) or np.isnan(rl):
        return None
    rl_round = int(round(float(rl) % 360.0))
    M = np.zeros((NREL, composite_config.N_LAGS), dtype=np.int8)
    have = np.zeros(composite_config.N_LAGS, dtype=bool)
    abs_lons = np.arange(composite_config.NLON)
    rel_idx = ((abs_lons - rl_round + REL_LON_HALF)
               % composite_config.NLON) - REL_LON_HALF
    valid = (rel_idx >= -REL_LON_HALF) & (rel_idx <= REL_LON_HALF)
    abs_to_rel = rel_idx + REL_LON_HALF
    for li, lh in enumerate(composite_config.LAG_HOURS):
        dt = (rt + datetime.timedelta(hours=int(lh))).to_pydatetime()
        res = lookup_strip(data_cache=data_cache, dt=dt)
        if res is None:
            continue
        _, fs = res
        present = (np.asarray(fs) > 0).astype(np.int8)
        M[abs_to_rel[valid], li] = present[valid]
        have[li] = True
    if not have.any():
        return None
    return M


def _composite_M(
        *,
        data_cache: dict,
        storm_recs: list[dict[str, Any]],
        reference: str,
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    F = np.zeros((NREL, composite_config.N_LAGS), dtype=np.float64)
    n = 0
    masks: list[np.ndarray | None] = []
    for st in storm_recs:
        M = _per_storm_mask(data_cache=data_cache, storm=st, reference=reference)
        if M is None:
            masks.append(None)
            continue
        F += M.astype(np.float64)
        n += 1
        masks.append(M)
    F /= max(n, 1)
    return F, masks


def _mc_null_relative(
        *,
        data_cache: dict,
        storm_recs: list[dict[str, Any]],
        reference: str,
        n_iter: int,
        seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n_iter, NREL, composite_config.N_LAGS), dtype=np.float32)
    for it in range(n_iter):
        F = np.zeros((NREL, composite_config.N_LAGS), dtype=np.float64)
        n = 0
        for st in storm_recs:
            shift = datetime.timedelta(days=int(rng.integers(-7, 8)))
            shifted = dict(st)
            for col in ("recurv_time", "et_time"):
                if col in shifted and pd.notna(shifted[col]):
                    shifted[col] = pd.Timestamp(shifted[col]) + shift
            M = _per_storm_mask(
                data_cache=data_cache, storm=shifted, reference=reference)
            if M is None:
                continue
            F += M.astype(np.float64)
            n += 1
        out[it] = (F / max(n, 1)).astype(np.float32)
    return out


def _mc_worker(args: tuple) -> np.ndarray:
    cache, recs, ref, n_it, seed = args
    return _mc_null_relative(
        data_cache=cache, storm_recs=recs, reference=ref,
        n_iter=n_it, seed=seed)


def main(
        data_config: Annotated[Path, typer.Option(
            help="JSON file mapping data-source keys to root directories "
                 "(needs era5_1deg_extracted)")],
        tracks_directory: Annotated[Path, typer.Option(
            help="Directory with recurving_nh_tracks.csv and individual/")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the classification CSV")],
        reference: Annotated[str, typer.Option(
            help="recurvature, et, or both")] = "recurvature",
        basins: Annotated[Optional[list[str]], typer.Option()] = None,
        years: Annotated[Optional[list[int]], typer.Option(
            help="Envelope preload year range (two values)")] = None,
        workers: Annotated[int, typer.Option()] = 16,
        mc_iter: Annotated[int, typer.Option()] = 300,
        output_csv: Annotated[str, typer.Option()] = "storm_rw_classification.csv",
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())

    cfg = data_registry.load_data_config(path=data_config)
    extracted_dir = cfg.get("era5_1deg_extracted")
    if not extracted_dir:
        raise SystemExit("data-config missing key 'era5_1deg_extracted'")

    if basins is None:
        basins = ["WP", "NA"]
    if years is None:
        years = [2000, 2022]

    os.makedirs(output_directory, exist_ok=True)
    storms_df, _ = tracks.load_track_database(tracks_directory=tracks_directory)
    print(f"Loaded {len(storms_df)} storms", flush=True)

    refs = (["recurvature", "et"] if reference == "both" else [reference])

    print(f"Pre-loading RWP envelopes {years[0]}..{years[1]} "
          f"({workers} workers)...", flush=True)
    data_cache = preload_all_envelopes(
        extracted_directory=Path(extracted_dir),
        years=range(years[0], years[1] + 1),
        n_workers=workers)

    rows = []
    for ref in refs:
        sl = storms_df[storms_df["basin"].isin(basins)]
        if ref == "et":
            sl = sl.dropna(subset=["et_time"])
        recs = sl.to_dict("records")
        print(f"\n=== reference={ref}, pooled {basins}: "
              f"N={len(recs)} ===", flush=True)

        print("  Building per-storm M_c (storm-relative)...", flush=True)
        F_obs, M_list = _composite_M(
            data_cache=data_cache, storm_recs=recs, reference=ref)

        print(f"  MC null ({mc_iter} iter, recurvature-time shifts)...",
              flush=True)
        nw = max(1, workers)
        per_w = max(1, mc_iter // nw)
        rem = mc_iter - per_w * nw
        argv = [(data_cache, recs, ref,
                 per_w + (1 if w < rem else 0), 4242 + w)
                for w in range(nw)
                if per_w + (1 if w < rem else 0) > 0]
        all_F = []
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(nw, len(argv)),
                mp_context=mp.get_context("fork")) as pool:
            for fa in pool.map(_mc_worker, argv):
                all_F.append(fa)
        all_F_arr = np.concatenate(all_F, axis=0)
        F_p95 = np.nanpercentile(all_F_arr, 95, axis=0)

        sig_mask = F_obs > F_p95
        lag_window = ((composite_config.LAG_HOURS >= 0)
                      & (composite_config.LAG_HOURS <= 120))[None, :]
        rel_window = ((REL_LON >= -10) & (REL_LON <= 80))[:, None]
        S = sig_mask & lag_window & rel_window
        S_total = int(S.sum())
        print(f"  S total points (95% sig & rel-lon -10..80 & T+0..+5d) = "
              f"{S_total}", flush=True)
        if S_total == 0:
            print("  WARNING: empty significance mask, falling back to box-only.",
                  flush=True)
            S = lag_window & rel_window

        scores = []
        for st, M in zip(recs, M_list):
            sid = st["storm_id"]
            if M is None:
                scores.append((sid, np.nan))
                continue
            num = int(((M.astype(bool)) & S).sum())
            den = int(S.sum())
            scores.append((sid, num / den if den > 0 else np.nan))

        score_map = dict(scores)
        valid = [(sid, s) for sid, s in scores if np.isfinite(s)]
        valid.sort(key=lambda x: x[1])
        n = len(valid)
        nq = max(1, n // 5)
        no_ids = {valid[i][0] for i in range(nq)}
        rw_ids = {valid[i][0] for i in range(n - nq, n)}
        print(f"  N={n} valid; quintile size={nq}")
        print(f"    no-RW score range: {valid[0][1]:.3f}..{valid[nq - 1][1]:.3f}")
        print(f"    RW    score range: {valid[n - nq][1]:.3f}..{valid[-1][1]:.3f}")

        for st in recs:
            sid = st["storm_id"]
            grp = ("rwcase" if sid in rw_ids
                   else "norwcase" if sid in no_ids
                   else "midrw")
            rows.append({
                "storm_id": sid,
                "name": st["name"],
                "basin": st["basin"],
                "season": st["season"],
                "reference": ref,
                "rw_score": score_map.get(sid, np.nan),
                "wb_group": grp,
            })

    out = pd.DataFrame(rows)
    out_path = os.path.join(output_directory, output_csv)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out)} rows)")
    for grp in ("rwcase", "norwcase", "midrw"):
        for ref in refs:
            sub = out[(out.wb_group == grp) & (out.reference == ref)]
            print(f"  {ref:12s} {grp:9s} : N={len(sub)} "
                  f"(WP={len(sub[sub.basin == 'WP'])}, "
                  f"NA={len(sub[sub.basin == 'NA'])})")


if __name__ == "__main__":
    typer.run(main)
