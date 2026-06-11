# Downstream ET LWA

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Typed](https://img.shields.io/badge/typed-mypy-blue.svg)

Local finite-amplitude wave activity (LWA) diagnostics for the downstream
Rossby wave response to recurving tropical cyclones undergoing extratropical
transition (ET). Associated with the methods described in Sarro et al.
(submitted): *Latent Heating and Jet Carrying Capacity Govern the Downstream
Rossby Wave Response to Extratropical Transition in Present and Future
Climates*.

## Overview

The pipeline takes 6-hourly reanalysis or model output (ERA5, MERRA-2,
MPAS-A, or any dataset regridded to an ERA5-style layout) and produces:

- the column-mean **LWA budget** (zonal/meridional flux convergence,
  lower-boundary injection, non-advective residual) following
  Huang & Nakamura (2016) and Nakamura & Huang (2018), computed with the
  published [falwa](https://github.com/csyhuang/hn2016_falwa) package;
- **diabatic LWA sources**: the latent-heating tendency inferred from the
  material derivative of specific humidity (ERA5-style `Dq/Dt`), archived
  process tendencies (MERRA-2 `DTDT*`), and the ageostrophic (non-QG)
  outflow source;
- the **jet carrying capacity** `F_c` (Barpanda & Nakamura 2025) as a
  monthly climatology interpolated to daily cadence, plus a time-resolved
  variant;
- **Rossby wave packet (RWP) envelopes** via the Hilbert transform of the
  250-hPa meridional wind (Zimin et al. 2003; Quinting & Jones 2016);
- a **counterfactual TC-local latent-heating removal experiment**: forward
  integration of the 2-D LWA budget with the storm's latent-heating source
  masked (extension of Neal et al. 2022 and Lubis & Nakamura 2024);
- recurvature-relative **storm composites**, RWP/wave-breaking
  stratifications, Monte Carlo significance, and all paper figures.

Rossby wave breaking masks are produced by the companion repository
[wave_breaking_using_qgpv](https://github.com/gmsarro/wave_breaking_using_qgpv),
which detects and classifies overturning of the QGPV contour at 10 km
pseudo-height — the same field used by the LWA framework.

## Repository layout

```
downstream-et-lwa/
├── downstream_et_lwa/            # Core Python package
│   ├── cli.py                    # Typer CLI entry point
│   ├── constants.py              # Physical constants
│   ├── tracks.py                 # IBTrACS recurving-TC database
│   ├── data_registry.py          # Variable loaders (JSON data config)
│   ├── grid_utils.py             # Budget term derivations, regridding
│   ├── hovmoller.py              # Hovmöller strip composites
│   ├── significance.py           # Monte Carlo significance
│   ├── preprocessing/            # ERA5/MPAS regridding and extraction
│   ├── budget/                   # LWA budget terms, Dq/Dt, heating sources
│   ├── capacity/                 # Jet carrying capacity F_c
│   ├── rwp/                      # RWP envelopes and climatologies
│   ├── lh_removal/               # Counterfactual LH-removal experiment
│   ├── classification/           # RWP / wave-breaking stratification
│   ├── composites/               # Composite engine and drivers
│   └── plotting/                 # Shared figure helpers
├── figures/                      # Paper figure scripts (standalone CLIs)
├── fortran/
│   └── ageo/                     # Ageostrophic (non-QG) LWA source
├── data_config.example.json      # Template for dataset locations
└── pyproject.toml
```

## Installation

```bash
pip install -e .
```

For development (mypy, pytest, ruff):

```bash
pip install -e ".[dev]"
```

The LWA computations require [falwa](https://github.com/csyhuang/hn2016_falwa)
(installed automatically; needs a Fortran compiler for its extensions).
The ageostrophic source in `fortran/ageo/` is the only component not
reproducible with falwa and is compiled separately:

```bash
cd fortran/ageo && ./build.sh
```

## Pipeline

All steps are exposed through the `downstream-et-lwa` CLI. Run
`downstream-et-lwa --help` for the full list, or
`downstream-et-lwa <command> --help` for per-step options. Every step takes
its input locations explicitly; the composite steps read a JSON mapping of
dataset roots (copy and edit `data_config.example.json`).

| Phase | Command | Purpose |
|-------|---------|---------|
| 0. Preprocessing | `extract-era5-1deg` | ERA5 0.25° u, v, t at 250/500/850 hPa → 1° NH |
| | `mpas-to-era5-grid` | MPAS mesh → ERA5-style 1° × 37 pressure levels |
| | `mpas-topoclean` | Topography masking + fill for MPAS columns |
| | `extract-mpas-v250` | MPAS 250-hPa meridional wind → 1° NH |
| 1. LWA budget | `compute-baro-terms` | falwa LWA + budget flux terms (BARO archive) |
| | `build-budget-strips` | 20–80°N budget strips (Terms I–III, residual) |
| | `build-budget-strips-climatology` | Seasonal strip climatology |
| 2. Heating sources | `compute-dqdt` | Material derivative Dq/Dt from u, v, w, q |
| | `dqdt-to-dtheta-dt` | Dq/Dt → latent-heating θ̇ (ERA5-style) |
| | `dtdt-to-dtheta-dt` | Archived dT/dt (MERRA-2 DTDT*) → θ̇ |
| | `heating-lwa-source` | θ̇ → barotropic LWA tendency (falwa ncforce) |
| | `export-qg-binaries` | QG fields → Fortran binaries for the ageo source |
| | `fortran/ageo` | Ageostrophic (non-QG) LWA source |
| 3. RWP | `build-rwp-envelope` | Hilbert envelope of v250 (k = 5–15) |
| | `build-rwp-climatology` | Seasonal RWP frequency/amplitude climatology |
| | `build-lwa-envelope` | LWA-based RWP masks (Ghinassi-style) |
| | `build-lwa-climatology` | LWA RWP climatology |
| 4. Jet capacity | `compute-stationary-a0` | Stationary LWA A₀ from monthly-mean fields |
| | `compute-carrying-capacity` | Monthly-climatology F_c, A_c, α, c_dopp |
| | `compute-carrying-capacity-time-resolved` | Rolling-window daily F_c |
| | `interpolate-capacity-daily` | Monthly F_c → daily cadence (cubic) |
| 5. Tracks & composites | `build-tracks` | IBTrACS → recurving NH TC database |
| | `run-composites` | Recurvature-relative 2-D composites |
| | `classify-rw` / `classify-wb` / `classify-mpas-rw` | RWP / wave-breaking stratification |
| | `run-rw-stratified` / `run-wb-stratified` | Stratified composites |
| | `merra2-source-composites` | MERRA-2 heating decomposition composites |
| | `run-mc-envelope` | Monte Carlo significance envelopes |
| | `mpas-rwp-pipeline` / `mpas-qgpv-export` / `mpas-rwb-masks` | MPAS orchestration |
| | `build-table-cases` | Track-following case statistics |
| 6. LH removal | `lh-event-catalog` | Per-storm forcing catalog (budget closure) |
| | `lh-forward-integrate` | CTRL / NoLH / NoR⁺ integrations |
| | `lh-run-population` | Batch over all storms → strips |
| | `lh-composite-figure` / `lh-fc-quantiles` | Removal composites |
| 7. Figures | `figures/fig*.py` | Paper figures (standalone typer CLIs) |

### Example

```bash
downstream-et-lwa compute-baro-terms \
    --input-directory /path/to/era5_style \
    --output-directory /path/to/baro_n \
    --year 2014 --month 11

downstream-et-lwa build-rwp-envelope \
    --input-directory /path/to/1deg_extracted \
    --output-directory /path/to/rwp \
    --year-start 2000 --year-end 2022
```

## Reproducing the paper figures

Each script in `figures/` is a standalone typer CLI; run with `--help` for
its options. All take `--output-directory` and the locations of the
composites, strips, tracks, and climatologies produced by the pipeline.

| Figure | Script |
|--------|--------|
| 1 | `fig01_track_map.py` |
| 2 | `fig02_rwp_lwa_wp_na.py` |
| 3 | `fig03_lwa_budget_wp_na.py` |
| 4 | `fig04_lwa_merra2_wp_na.py` |
| 5 (+diff, 5b) | `fig05_budget_maps_wp_na.py`, `fig05_budget_maps_diff.py`, `fig05b_merra2_diabatic_wp_na.py` |
| 5c | `lh-composite-figure`, `lh-fc-quantiles` (CLI commands) |
| 6 (+diff) | `fig06_budget_maps_rw_strat.py`, `fig06_budget_maps_rw_strat_diff.py`, `fig06_budget_maps_wb_strat.py` |
| 7 | `fig07_rwp_lwa_wb_strat.py` |
| 8 | `fig08_lwa_budget_rw_strat.py`, `fig08_lwa_budget_wb_strat.py` |
| 9–11 | `fig09_mpas_rwp_lwa.py`, `fig10_mpas_zonal_budget.py`, `fig11_mpas_budget_maps.py` (+`_diff`) |
| 12–14 | `fig12_mpas_rwp_lwa_wp_only.py`, `fig13_mpas_zonal_budget_wp_only.py`, `fig14_mpas_budget_maps_wp_only.py` (+`_diff`) |
| 15–17 | `fig15_mpas_rwp_lwa_na_only.py`, `fig16_mpas_zonal_budget_na_only.py`, `fig17_mpas_budget_maps_na_only.py` |
| 18 | `fig18_rwp_lwa_rw_strat.py` |

## Input data

The pipeline is dataset-agnostic: any source that can be brought to
ERA5-style monthly NetCDF files of `u, v, t` (plus `w, q` for the heating
source and `z` for the ageo export) on pressure levels works. The expected
filename patterns are CLI options on every step.

| Dataset | Notes |
|---------|-------|
| ERA5 | 6-hourly pressure-level fields; Dq/Dt latent heating from u, v, w, q |
| MERRA-2 | Archived process tendencies (`DTDTMST`, `DTDTRAD`, `DTDTTRB`, `DTDTANA`) via `dtdt-to-dtheta-dt` |
| MPAS-A | Regrid with `mpas-to-era5-grid`, then identical pipeline |
| IBTrACS | v04r00 NetCDF for the TC track database |
| IMERG | Optional precipitation composites |

Composite steps locate gridded archives through a JSON config (see
`data_config.example.json`) so that no paths are baked into the code.

## References

- Nakamura, N., and C. S. Y. Huang, 2018: Atmospheric blocking as a traffic
  jam in the jet stream. *Science*, **361**, 42–47.
- Huang, C. S. Y., and N. Nakamura, 2016: Local finite-amplitude wave
  activity as a diagnostic of anomalous weather events. *J. Atmos. Sci.*,
  **73**, 211–229.
- Barpanda, P., and N. Nakamura, 2025: The seasonal carrying capacity of
  midlatitude jet streams. *J. Climate*, **38**, 4653–4672.
- Lubis, S. W., and N. Nakamura, 2024: A two-dimensional advective model
  for jet-stream waviness. *J. Climate*.
- Neal, E., C. S. Y. Huang, and N. Nakamura, 2022: The 2021 Pacific
  Northwest heat wave and associated blocking. *Geophys. Res. Lett.*
- Quinting, J. F., and S. C. Jones, 2016: On the impact of tropical cyclones
  on Rossby wave packets. *Mon. Wea. Rev.*, **144**, 2021–2048.
- Zimin, A. V., I. Szunyogh, D. J. Patil, B. R. Hunt, and E. Ott, 2003:
  Extracting envelopes of Rossby wave packets. *Mon. Wea. Rev.*, **131**,
  1011–1017.
- Huang, C. S. Y., C. Polster, and N. Nakamura, 2025: falwa: Python package
  for finite-amplitude local wave activity diagnostics. *Geoscience Data
  Journal*, **12**, e70006.
  [csyhuang/hn2016_falwa](https://github.com/csyhuang/hn2016_falwa)
- Kaderli, S., 2023: WaveBreaking — detection, classification and tracking
  of Rossby wave breaking. [skaderli/WaveBreaking](https://github.com/skaderli/WaveBreaking)
- [gmsarro/wave_breaking_using_qgpv](https://github.com/gmsarro/wave_breaking_using_qgpv):
  RWB detection on QGPV at pseudo-height levels (companion repository).

## License

MIT
