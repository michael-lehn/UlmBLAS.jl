#!/bin/bash
#
#  sbatch -p PARTITION job-trsm.sh R JULIA [CONFIGS] (after prepare.sh R JULIA):
#  Netlib LAPACK + ulmBLAS with OpenBLAS's DTRSM (netlib_ulmblas_obtrsm)
#  against Netlib LAPACK + ulmBLAS and Netlib LAPACK + OpenBLAS's BLAS:
#  test.jl and coldom.jl (correctness), then bench.jl (DGETRF) with random
#  and column-dominant matrices taking turns.  Results:
#  R/results/PARTITION-JOBID.*
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --mem=32000mb
#SBATCH --output=%x-%j.out
set -eu
R=$1; JULIA=$2
P=$SLURM_JOB_PARTITION
L=$R/liblapack-$P
O=$R/results/$P-$SLURM_JOB_ID
mkdir -p $R/results
module load compiler/gnu/14.2
export JULIA_DEPOT_PATH=$R/depot JULIA_PKG_OFFLINE=true
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LAPACK_LIBRARY=$L/lib/liblapack.so
CORE=2
run() { taskset -c $CORE "$JULIA" --startup-file=no --project=$R/src/lapack "$@"; }

{
    echo "job $SLURM_JOB_ID on $(hostname), partition $P, $(date -Is)"
    lscpu | grep -E 'Model name|^CPU\(s\)|Socket|Thread|MHz|L1d|L2|L3'
    echo "pinned to core $CORE (taskset)"
    cat /proc/loadavg
} > $O.txt

if [ ! -f $L/lib/liblapack.so ]; then
    mkdir -p $L/src && cp $R/lapack-3.12.1.tar.gz $L/src/
    J=32 sh $R/src/lapack/build.sh $L "$JULIA" > $O.build.txt 2>&1
fi
cat $L/build.txt >> $O.txt

cd $R/src
export CONFIGS=${3:-netlib_ulmblas,netlib_ulmblas_obtrsm,netlib_openblas}
run lapack/test.jl > $O.test.txt 2>&1
run lapack/coldom.jl > $O.coldom.txt 2>&1
MATRIX=dense,coldom run lapack/bench.jl $O.tsv > $O.bench.txt 2>&1
echo "done $(date -Is)" >> $O.txt
