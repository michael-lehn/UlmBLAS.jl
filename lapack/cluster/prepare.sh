#!/bin/sh
#
#  prepare.sh R JULIA (login node, with internet): the depot R/depot with
#  the environment lapack/ of the sources in R/src (and the artifact of
#  UlmBLAS), and the tarball of Netlib LAPACK in R.
#
set -eu
R=$1; JULIA=$2
export JULIA_DEPOT_PATH=$R/depot
"$JULIA" --project="$R/src/lapack" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
T=$R/lapack-3.12.1.tar.gz
[ -f "$T" ] || curl -fsSL -o "$T" \
    https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz
echo "2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422  $T" | sha256sum -c -
