#!/bin/sh
#
#  check_blis_kernel.sh LIBDIR: which BLIS kernels run in dgemm_ for
#  n = 1000 and 1500 with 1 and 4 threads.  Builds blis_dgemm.c against
#  LIBDIR/libblis.dylib and runs it under lldb with auto-continuing
#  breakpoints on all BLIS dgemm micro-kernels (conventional and sup) and
#  dpackm kernels; prints the functions that were hit with their hit counts.
#
cd "$(dirname "$0")" || exit 1
lib=${1:?usage: check_blis_kernel.sh LIBDIR}
tmp=$(mktemp -d) || exit 1
clang -O2 blis_dgemm.c -o "$tmp/dg" -L"$lib" -lblis -Wl,-rpath,"$lib" || exit 1
for n in 1000 1500; do
    for t in 1 4; do
        echo "n = $n, BLIS_NUM_THREADS = $t"
        lldb -b -o "settings set target.env-vars BLIS_NUM_THREADS=$t" \
             -o "breakpoint set -r '^bli_dgemm.*_ukr|^bli_dgemm_arm|^bli_dgemmsup_|^bli_dpackm_|^bli_dgemm_.*ref' -s libblis.4.dylib" \
             -o "breakpoint modify --auto-continue true 1" \
             -o "run $n" -o "breakpoint list" -o quit "$tmp/dg" 2>&1 |
            grep -E "where = .*hit count = [1-9]" |
            sed -E 's/.*`([^,]*),.*hit count = ([0-9]+).*/    \1: \2 calls/'
    done
done
rm -rf "$tmp"
