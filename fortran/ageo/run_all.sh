#!/bin/bash
# Driver: build the (year, month) job list and run them in parallel via xargs.
#  Usage: run_all.sh [Nparallel] [year_start] [year_end]   (defaults 32 / 2000 / 2022)
#  Required env: AGEO_INDIR  (directory with {year}_{month}_QG* binaries)
#                AGEO_OUTDIR (output directory)
#  Optional env: AGEO_LOGDIR (defaults to $AGEO_OUTDIR/log)
#  Skips months whose output file already exists.
set -euo pipefail
NPAR="${1:-32}"
YEAR_START="${2:-2000}"
YEAR_END="${3:-2022}"
: "${AGEO_INDIR:?set AGEO_INDIR to the directory containing the QG binaries}"
: "${AGEO_OUTDIR:?set AGEO_OUTDIR to the output directory}"
CODEDIR="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="${AGEO_LOGDIR:-$AGEO_OUTDIR/log}"
JOBLIST="${LOGDIR}/joblist.txt"
SUMMARY="${LOGDIR}/summary.log"

mkdir -p "$LOGDIR" "$AGEO_OUTDIR"

# Build job list (months that have all required QG inputs)
> "$JOBLIST"
for YYYY in $(seq "$YEAR_START" "$YEAR_END"); do
  for MM in $(seq 1 12); do
    MM2=$(printf '%02d' "$MM")
    have_all=1
    for f in QGPV QGU QGT QGV QGZ QVORT QGREF_N; do
      [ -s "${AGEO_INDIR}/${YYYY}_${MM2}_${f}" ] || { have_all=0; break; }
    done
    [ "$have_all" -eq 1 ] || continue
    OUT="${AGEO_OUTDIR}/${YYYY}_${MM2}_AOUTbaro_N.nc"
    [ -s "$OUT" ] && continue   # skip if already done
    echo "${YYYY} ${MM}" >> "$JOBLIST"
  done
done

NJOBS=$(wc -l < "$JOBLIST")
echo "$(date)  starting $NJOBS jobs with $NPAR parallel workers" | tee -a "$SUMMARY"

# Launch with xargs -P
xargs -L1 -P"$NPAR" -a "$JOBLIST" "$CODEDIR/run_one.sh"

echo "$(date)  all jobs finished" | tee -a "$SUMMARY"
