#!/bin/sh
#
#  run_accelerate.sh [RESULTS]: gemm_accelerate.jl once per setting of
#  VECLIB_MAXIMUM_THREADS (1 to 4), each in its own Julia process (the
#  variable is read when Accelerate is loaded).  The results of all runs are
#  appended to RESULTS.tsv (default gemm_accelerate.tsv, replaced at the
#  start).  Start it on a quiet machine.
#
cd "$(dirname "$0")" || exit 1
results=${1:-gemm_accelerate}
rm -f "$results.tsv"
for t in 1 2 3 4; do
    echo "######## VECLIB_MAXIMUM_THREADS=$t"
    VECLIB_MAXIMUM_THREADS=$t RESULTS=$results \
        julia --project=. gemm_accelerate.jl || exit 1
done
