"""Composite geometry, lag window, target grid, and variable-name constants."""

from __future__ import annotations

import numpy as np

import downstream_et_lwa.constants as constants

HOURS_BEFORE = 48
HOURS_AFTER = 168
DT_HOURS = 6
N_LAGS = (HOURS_BEFORE + HOURS_AFTER) // DT_HOURS + 1
LAG_HOURS = np.arange(-HOURS_BEFORE, HOURS_AFTER + DT_HOURS, DT_HOURS)

COMPOSITE_PREMEAN_VOLUME_SIGMA_3D: tuple[float, float, float] = (1.0, 2.5, 1.0)
COMPOSITE_READ_VOLUME_SIGMA_3D: tuple[float, float, float] = (0.0, 0.0, 0.0)

BOX_LAT_HALF = 25
BOX_LON_WEST = 25
BOX_LON_EAST = 125
BOX_NLAT = 2 * BOX_LAT_HALF + 1
BOX_NLON = BOX_LON_WEST + BOX_LON_EAST + 1

HOV_LAT_MIN = 20
HOV_LAT_MAX = 80

RWP_KMIN = 5
RWP_KMAX = 15

NLAT = 91
NLON = 360
TARGET_LAT = np.linspace(0, 90, NLAT)
TARGET_LON = np.linspace(0, 359, NLON)

A_EARTH = constants.EARTH_RADIUS_M
DT_SEC = DT_HOURS * 3600
DLAMBDA = np.deg2rad(1.0)
COSPHI = np.cos(np.deg2rad(TARGET_LAT))

MIN_RECURV_LAT = 15.0
YEAR_START = 2000
YEAR_END = 2022
NH_BASINS = ["WP", "EP", "NA", "NI"]

ERA5_PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175,
    200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650,
    700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
]

COMPOSITE_PRESSURE_LEVELS = [250, 500, 850]

LWA_BUDGET_SUFFIXES: dict[str, tuple[str, str]] = {
    "lwa": ("LWAb_N", "lwa"),
    "ua1": ("ua1_N", "ua1"),
    "ua2": ("ua2_N", "ua2"),
    "ep1": ("ep1_N", "ep1"),
    "ep2a": ("ep2a_N", "ep2a"),
    "ep3a": ("ep3a_N", "ep3a"),
    "ep4": ("ep4_N", "ep4"),
    "Ub": ("Ub_N", "u"),
    "Urefb": ("Urefb_N", "uref"),
}

LWA_BUDGET_SUFFIXES_MERRA2: dict[str, tuple[str, str]] = {
    **LWA_BUDGET_SUFFIXES,
    "ep2a": ("ep2_N", "ep2"),
    "ep3a": ("ep3_N", "ep3"),
}

TDT_VARS = [
    "DTDTMST", "DTDTRAD", "DTDTDYN", "DTDTFRI",
    "DTDTGWD", "DTDTTRB", "DTDTANA", "DTDTTOT",
]

CC_VARS = ["Fc", "Ac", "C", "A0", "alpha_zm"]
