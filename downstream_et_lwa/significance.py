"""Monte Carlo significance testing for Hovmoller composites (Quinting & Jones 2016, Sec. 2c)."""

from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd

import downstream_et_lwa.composite_config as composite_config
import downstream_et_lwa.data_registry as data_registry
import downstream_et_lwa.grid_utils as grid_utils
import downstream_et_lwa.hovmoller as hovmoller

_LOG = logging.getLogger(__name__)


def monte_carlo_hovmoller(
        *,
        storms_df: pd.DataFrame,
        var_key: str = "era5_v_250hPa",
        basin: str = "WP",
        reference: str = "recurvature",
        n_random: int = 1000,
        seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    storm_list = storms_df[storms_df["basin"] == basin]

    if reference == "recurvature":
        ref_col = "recurv_time"
    else:
        ref_col = "et_time"

    ref_times = pd.to_datetime(storm_list[ref_col])
    years_available = sorted(storm_list["season"].unique())

    ds = data_registry.get(key=var_key)
    if ds is None:
        raise ValueError(f"Variable {var_key} not in registry")

    random_composites = np.zeros(
        (n_random, composite_config.NLON, composite_config.N_LAGS))

    for iteration in range(n_random):
        if (iteration + 1) % 100 == 0:
            _LOG.info("Monte Carlo iteration %d/%d", iteration + 1, n_random)

        hov_sum = np.zeros((composite_config.NLON, composite_config.N_LAGS))
        hov_count = np.zeros(composite_config.N_LAGS, dtype=int)
        file_cache: dict = {}

        for i, ref_t in enumerate(ref_times):
            if pd.isna(ref_t):
                continue

            random_year = rng.choice(years_available)
            day_offset = rng.integers(-7, 8)
            random_ref = (ref_t.replace(year=random_year)
                          + datetime.timedelta(days=int(day_offset)))

            for lag_idx, lag_h in enumerate(composite_config.LAG_HOURS):
                target_dt = random_ref + datetime.timedelta(hours=int(lag_h))
                try:
                    target_dt = target_dt.to_pydatetime()
                except Exception:
                    _LOG.exception("Failed converting %s to datetime", target_dt)

                raw = data_registry.load_snapshot(
                    source=ds, target_dt=target_dt, cache=file_cache)
                if raw is None:
                    continue

                field = grid_utils.prepare_field(data=raw, source=ds)
                if field is None:
                    continue

                strip = hovmoller.extract_hovmoller_strip(field_2d=field)
                if np.any(np.isfinite(strip)):
                    hov_sum[:, lag_idx] += np.nan_to_num(strip)
                    hov_count[lag_idx] += 1

        data_registry.close_cache(cache=file_cache)

        n = hov_count.copy()
        n[n == 0] = 1
        random_composites[iteration] = hov_sum / n[np.newaxis, :]

    percentile_5 = np.percentile(random_composites, 5, axis=0)
    percentile_95 = np.percentile(random_composites, 95, axis=0)

    return percentile_5, percentile_95


def significance_mask(
        *,
        composite: np.ndarray,
        p5: np.ndarray,
        p95: np.ndarray,
) -> np.ndarray:
    return (composite > p95) | (composite < p5)
