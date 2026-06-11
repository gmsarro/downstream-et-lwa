#!/bin/bash
# Build ageo_lwa_source.out with gfortran (default) or ifort/ifx (set FC).
# NetCDF flags are taken from nf-config when available; override with
# NETCDF_INC / NETCDF_LIBS if needed.
set -euo pipefail
cd "$(dirname "$0")"

FC="${FC:-gfortran}"
FCFLAGS="${FCFLAGS:--O2}"

if command -v nf-config >/dev/null 2>&1; then
    NETCDF_INC="${NETCDF_INC:-$(nf-config --includedir)}"
    NETCDF_LIBS="${NETCDF_LIBS:-$(nf-config --flibs)}"
else
    NETCDF_INC="${NETCDF_INC:-/usr/include}"
    NETCDF_LIBS="${NETCDF_LIBS:--lnetcdff -lnetcdf}"
fi

"$FC" $FCFLAGS -I"$NETCDF_INC" ageo_lwa_source.f90 $NETCDF_LIBS \
    -o ageo_lwa_source.out

rm -f -- *.o *.mod
echo "built ./ageo_lwa_source.out"
