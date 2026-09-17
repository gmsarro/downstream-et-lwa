#!/usr/bin/env python3
"""Flip the sign of legacy ageostrophic (non-QG) LWA-source fields in place.

Background
----------
Before 2026-09-17 ``fortran/ageo/ageo_lwa_source.f90`` accumulated the
LWA-projected ageostrophic forcing ``AOUT`` with the opposite sign to the
LWA operator (``+FORCE`` on the poleward q_e<=0 branch, ``-FORCE`` on the
equatorward q_e>0 branch).  The stored field was therefore
``-d(LWA cos phi)/dt`` rather than the LWA tendency ``S_q`` defined in the
paper (Appendix C).  The Fortran has been fixed and now writes the global
attribute ``aout_sign_convention = "lwa_tendency"``.

This tool brings *existing* products to the new convention without
recomputation.  A file is only touched if it lacks the marker attribute, so
the script is idempotent and safe to re-run.

Supported products (variables negated / marker attribute written):

* monthly ``*_AOUTbaro_N.nc``           ``aout_baro``            ``aout_sign_convention``
* Hovmoller strips ``budget_strips_*``  ``nonqg_lwa``            ``nonqg_sign_convention``
* strip climatology (per-season groups) ``nonqg_lwa``            ``nonqg_sign_convention``
* 2-D composites ``composite_2d_*.nc``  ``*nonqg_lwa_mean``      ``nonqg_sign_convention``

Sum-of-squares and count fields are sign-invariant and left untouched.

Usage
-----
    python tools/fix_legacy_aout_sign.py FILE [FILE ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
from datetime import datetime, timezone

import netCDF4 as nc

MARKER_VALUE = "lwa_tendency"

# (glob on variable name, global-attribute marker)
RULES = (
    ("aout_baro", "aout_sign_convention"),
    ("nonqg_lwa", "nonqg_sign_convention"),
    ("*nonqg_lwa_mean", "nonqg_sign_convention"),
)


def _matching_vars(grp: nc.Dataset | nc.Group, pattern: str):
    return [v for v in grp.variables if fnmatch.fnmatch(v, pattern)]


def fix_file(path: str, dry_run: bool = False) -> str:
    with nc.Dataset(path, "r" if dry_run else "r+") as ds:
        groups = [ds] + list(ds.groups.values())
        touched = []
        for pattern, marker in RULES:
            hits = [(g, v) for g in groups for v in _matching_vars(g, pattern)]
            if not hits:
                continue
            if getattr(ds, marker, None) == MARKER_VALUE:
                return f"skip (already {marker}={MARKER_VALUE}): {path}"
            for g, v in hits:
                touched.append(v)
                if not dry_run:
                    var = g.variables[v]
                    var[:] = -var[:]
            if not dry_run:
                ds.setncattr(marker, MARKER_VALUE)
                stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
                note = (f"{stamp}: sign of {', '.join(sorted(set(touched)))} flipped to the "
                        f"LWA-tendency convention (tools/fix_legacy_aout_sign.py)")
                old = getattr(ds, "history", "")
                ds.setncattr("history", (old + "\n" if old else "") + note)
        if not touched:
            return f"no matching variables: {path}"
        verb = "would flip" if dry_run else "flipped"
        return f"{verb} {sorted(set(touched))}: {path}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    for f in a.files:
        try:
            print(fix_file(f, a.dry_run), flush=True)
        except Exception as exc:  # keep going, report at the end
            print(f"ERROR {f}: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
