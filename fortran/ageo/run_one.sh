#!/bin/bash
# Run ageo_lwa_source.out for one (year, month).
#  Usage: run_one.sh <year> <month>
#  Required env: AGEO_INDIR  (directory with {year}_{month}_QG* binaries)
#                AGEO_OUTDIR (output directory)
#  Optional env: AGEO_LOGDIR (defaults to $AGEO_OUTDIR/log)
set -euo pipefail
YYYY="$1"; MM="$2"
: "${AGEO_INDIR:?set AGEO_INDIR to the directory containing the QG binaries}"
: "${AGEO_OUTDIR:?set AGEO_OUTDIR to the output directory}"
LOGDIR="${AGEO_LOGDIR:-$AGEO_OUTDIR/log}"
MM2=$(printf '%02d' "$MM")
OUT="${AGEO_OUTDIR}/${YYYY}_${MM2}_AOUTbaro_N.nc"
LOG="${LOGDIR}/${YYYY}_${MM2}.log"

mkdir -p "$LOGDIR" "$AGEO_OUTDIR"

# Skip if already done
if [ -s "$OUT" ]; then
  echo "$YYYY-$MM2  SKIP (exists)" >> "$LOG"
  exit 0
fi

cd "$(dirname "$0")"

# don't crash if QGREF / QGPV missing
for f in QGPV QGU QGT QGV QGZ QVORT QGREF_N; do
    if [ ! -s "${AGEO_INDIR}/${YYYY}_${MM2}_${f}" ]; then
        echo "$YYYY-$MM2  SKIP (missing input ${f})" >> "$LOG"
        exit 0
    fi
done

echo "$YYYY-$MM2  START $(date)" >> "$LOG"
./ageo_lwa_source.out "$YYYY" "$MM" "$AGEO_INDIR" "$AGEO_OUTDIR" >> "$LOG" 2>&1
echo "$YYYY-$MM2  DONE  $(date)" >> "$LOG"
