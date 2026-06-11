"""Classify MPAS storms into RW/noRW with the ERA5 QJ16 score: per-scenario
RWP-envelope strips (current/future) pooled into one score distribution."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Optional

import netCDF4 as nc
import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.tracks as tracks

_LOG = logging.getLogger(__name__)

NLAT = 91
NLON = 360
LATS = np.linspace(0, 90, NLAT)
COSPHI_STRIP = np.cos(np.deg2rad(LATS))

LAT_MIN = 20.0
LAT_MAX = 80.0

REL_LON_HALF = 180
REL_LON = np.arange(-REL_LON_HALF, REL_LON_HALF + 1)
NREL = REL_LON.size


def _merid_avg_2d(
        *,
        arr_tlatlon: np.ndarray,
        lat_min: float = LAT_MIN,
        lat_max: float = LAT_MAX,
) -> np.ndarray:
    sel = (LATS >= lat_min) & (LATS <= lat_max)
    w = COSPHI_STRIP[sel]
    sub = arr_tlatlon[..., sel, :]
    return (sub * w[:, None]).sum(axis=-2) / w.sum()


def load_strips(
        *,
        rwp_directory: Path,
        year_start: int,
        year_end: int,
        lat_min: float = LAT_MIN,
        lat_max: float = LAT_MAX,
) -> dict[str, Any]:
    times_list, m_list = [], []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            path = rwp_directory / f"rwp_envelope_{year}_{month:02d}.nc"
            if not path.exists():
                continue
            with nc.Dataset(str(path), "r") as d:
                ts = nc.num2date(
                    d["time"][:], d["time"].units,
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True,
                )
                env_thr = d["envelope_thr"][:]
            e1d = _merid_avg_2d(
                arr_tlatlon=env_thr, lat_min=lat_min,
                lat_max=lat_max).astype(np.float32)
            m1d = (e1d > 0).astype(np.float32)
            if not isinstance(ts, np.ndarray):
                ts = np.array([ts])
            times_list.append(np.array([np.datetime64(t, "s") for t in ts]))
            m_list.append(m1d)

    if not times_list:
        raise FileNotFoundError(
            f"No rwp_envelope_*.nc under {rwp_directory} for years "
            f"{year_start}-{year_end}")

    strip_times = np.concatenate(times_list)
    strip_m = np.concatenate(m_list, axis=0)

    order = np.argsort(strip_times)
    strip_times = strip_times[order]
    strip_m = strip_m[order]
    strip_index = {t: i for i, t in enumerate(strip_times)}

    _LOG.info("Loaded %d 1-D strips covering %s .. %s",
              len(strip_times), strip_times[0], strip_times[-1])

    return {"M": strip_m, "index": strip_index}


def _round_to_6h(*, ts: pd.Timestamp) -> pd.Timestamp:
    dt = ts.to_pydatetime()
    hr = int(round(dt.hour / 6.0)) * 6
    if hr == 24:
        dt = dt + datetime.timedelta(days=1)
        hr = 0
    return pd.Timestamp(dt.replace(hour=hr, minute=0, second=0, microsecond=0))


def _strip_at(*, cache: dict[str, Any], target: pd.Timestamp) -> np.ndarray | None:
    t64 = np.datetime64(target.to_pydatetime().replace(tzinfo=None), "s")
    j = cache["index"].get(t64)
    if j is None:
        return None
    return cache["M"][j]


def _per_storm_mask(
        *,
        caches: dict[str, dict[str, Any]],
        storm: dict[str, Any],
        reference: str,
) -> np.ndarray | None:
    rt = pd.Timestamp(
        storm["recurv_time"] if reference == "recurvature" else storm["et_time"])
    rl = storm["recurv_lon"] if reference == "recurvature" else storm["et_lon"]
    if pd.isna(rt) or pd.isna(rl):
        return None

    cache = caches[storm["scenario"]]
    rt = _round_to_6h(ts=rt)
    rl_round = int(round(float(rl) % 360.0)) % composite_config.NLON
    out = np.zeros((NREL, composite_config.N_LAGS), dtype=np.int8)
    have = np.zeros(composite_config.N_LAGS, dtype=bool)

    abs_lons = np.arange(composite_config.NLON)
    rel_idx = ((abs_lons - rl_round + REL_LON_HALF)
               % composite_config.NLON) - REL_LON_HALF
    valid = (rel_idx >= -REL_LON_HALF) & (rel_idx <= REL_LON_HALF)
    abs_to_rel = rel_idx + REL_LON_HALF

    for li, lag_h in enumerate(composite_config.LAG_HOURS):
        target = rt + pd.Timedelta(hours=int(lag_h))
        strip = _strip_at(cache=cache, target=target)
        if strip is None:
            continue
        present = (np.asarray(strip) > 0).astype(np.int8)
        out[abs_to_rel[valid], li] = present[valid]
        have[li] = True

    return out if have.any() else None


def _composite_and_masks(
        *,
        caches: dict[str, dict[str, Any]],
        records: list[dict[str, Any]],
        reference: str,
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    F = np.zeros((NREL, composite_config.N_LAGS), dtype=np.float64)
    masks: list[np.ndarray | None] = []
    n = 0
    for storm in records:
        M = _per_storm_mask(caches=caches, storm=storm, reference=reference)
        masks.append(M)
        if M is None:
            continue
        F += M.astype(np.float64)
        n += 1
    return F / max(n, 1), masks


def _mc_null(
        *,
        caches: dict[str, dict[str, Any]],
        records: list[dict[str, Any]],
        reference: str,
        n_iter: int,
        workers: int,
) -> np.ndarray:
    chunks = []
    per_worker = max(1, n_iter // max(1, workers))
    remainder = n_iter - per_worker * max(1, workers)
    for w in range(max(1, workers)):
        n = per_worker + (1 if w < remainder else 0)
        if n > 0:
            chunks.append((n, 4242 + w))

    out = []
    for n, seed in chunks:
        rng = np.random.default_rng(seed)
        for _ in range(n):
            F = np.zeros((NREL, composite_config.N_LAGS), dtype=np.float64)
            count = 0
            for storm in records:
                shifted = dict(storm)
                shift = pd.Timedelta(days=int(rng.integers(-7, 8)))
                for col in ("recurv_time", "et_time"):
                    if col in shifted and pd.notna(shifted[col]):
                        shifted[col] = pd.Timestamp(shifted[col]) + shift
                M = _per_storm_mask(
                    caches=caches, storm=shifted, reference=reference)
                if M is None:
                    continue
                F += M.astype(np.float64)
                count += 1
            out.append((F / max(count, 1)).astype(np.float32))
    return np.stack(out, axis=0)


def _load_records(
        *,
        tracks_root: Path,
        basins: tuple[str, ...],
) -> list[dict[str, Any]]:
    records = []
    for scenario in ("current", "future"):
        storms, _ = tracks.load_track_database(
            tracks_directory=tracks_root / scenario)
        storms = storms[storms["basin"].isin(basins)].copy()
        storms["scenario"] = scenario
        records.extend(storms.to_dict("records"))
    return records


def classify(
        *,
        records: list[dict[str, Any]],
        caches: dict[str, dict[str, Any]],
        reference: str,
        mc_iter: int,
        workers: int,
) -> tuple[pd.DataFrame, int, int, int]:
    F_obs, masks = _composite_and_masks(
        caches=caches, records=records, reference=reference)
    F_p95 = np.nanpercentile(
        _mc_null(caches=caches, records=records, reference=reference,
                 n_iter=mc_iter, workers=workers), 95, axis=0)

    sig_mask = F_obs > F_p95
    lag_window = ((composite_config.LAG_HOURS >= 0)
                  & (composite_config.LAG_HOURS <= 120))[None, :]
    rel_window = ((REL_LON >= -10) & (REL_LON <= 80))[:, None]
    S = sig_mask & lag_window & rel_window
    if int(S.sum()) == 0:
        S = lag_window & rel_window

    scores: list[tuple[str, float]] = []
    for storm, M in zip(records, masks):
        sid = storm["storm_id"]
        if M is None:
            scores.append((sid, float(np.nan)))
            continue
        scores.append((sid, int((M.astype(bool) & S).sum()) / int(S.sum())))

    valid = [(sid, score) for sid, score in scores if np.isfinite(score)]
    valid.sort(key=lambda item: item[1])
    nq = max(1, len(valid) // 5)
    no_ids = {sid for sid, _ in valid[:nq]}
    rw_ids = {sid for sid, _ in valid[-nq:]}
    score_map = dict(scores)

    rows = []
    for storm in records:
        sid = storm["storm_id"]
        group = ("rwcase" if sid in rw_ids
                 else "norwcase" if sid in no_ids else "midrw")
        rows.append({
            "storm_id": sid,
            "name": storm["name"],
            "basin": storm["basin"],
            "scenario": storm["scenario"],
            "season": storm["season"],
            "reference": reference,
            "rw_score": score_map.get(sid, np.nan),
            "wb_group": group,
        })

    return pd.DataFrame(rows), len(valid), nq, int(S.sum())


def main(
        tracks_root: Annotated[Path, typer.Option(
            help="Directory containing current/ and future/ track DBs")],
        rwp_root: Annotated[Path, typer.Option(
            help="Directory containing current/ and future/ rwp_envelope files")],
        output_directory: Annotated[Path, typer.Option(
            help="Directory for the classification CSV")],
        output_csv: Annotated[str, typer.Option()] = "mpas_rw_classification.csv",
        year_start: Annotated[int, typer.Option()] = 1988,
        year_end: Annotated[int, typer.Option()] = 2016,
        mc_iter: Annotated[int, typer.Option()] = 300,
        workers: Annotated[int, typer.Option()] = 16,
        log_level: Annotated[Optional[str], typer.Option()] = "INFO",
) -> None:
    logging.basicConfig(level=str(log_level).upper())

    caches = {
        scenario: load_strips(
            rwp_directory=rwp_root / scenario,
            year_start=year_start, year_end=year_end)
        for scenario in ("current", "future")
    }
    records = _load_records(tracks_root=tracks_root, basins=("WP", "NA"))
    out, n_valid, nq, s_total = classify(
        records=records, caches=caches, reference="recurvature",
        mc_iter=mc_iter, workers=workers)
    out_path = output_directory / output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({len(out)} rows)")
    print(f"finite-score N={n_valid}; quintile N={nq}; S points={s_total}")
    print(out.groupby(["scenario", "basin", "wb_group"]).size().unstack(fill_value=0))
    print(out.groupby(["scenario", "wb_group"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    typer.run(main)
