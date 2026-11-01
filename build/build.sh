#!/bin/sh
#
#  build.sh SRC OUT [VARIANT ...]: the shared libraries of ulmBLAS for
#  Julia (make lbt) from the ulmBLAS source tree SRC, one per variant, in
#
#      OUT/lib/VARIANT/libulmblas_lbt.{so,dylib}
#
#  together with the license of ulmBLAS (OUT/share/licenses/ulmBLAS/LICENSE)
#  and a record of the build (OUT/share/ulmBLAS/build.txt: commit, compiler,
#  flags).  Without VARIANT, all variants of this platform are built:
#
#      macOS arm64:   neon, sme
#      Linux x86-64:  x86_64, haswell, skylakex, zen4
#
#  The flags of each variant select the micro kernels for S, D, C and
#  ZGEMM (the fastest of the benchmark campaigns of ulmBLAS for this kind
#  of CPU).  CXX and CC can be set as usual; J = number of parallel jobs;
#  the build logs go to LOGDIR (default OUT.logs, not into the artifact);
#  on Linux the debug information is stripped;
#  ULMBLAS_COMMIT names the commit of SRC if SRC is no git checkout.
#
set -eu
SRC=$(cd "$1" && pwd); OUT=$2; shift 2
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd)
J=${J:-8}
LOGDIR=${LOGDIR:-$OUT.logs}; mkdir -p "$LOGDIR"
#  macOS: the oldest version the libraries run on (default: the version of
#  the build machine)
[ "$(uname -s)" = Darwin ] && export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-12.0}

flags() {
    case $1 in
    neon)     echo "-DULMBLAS_NO_SME";;
    sme)      echo "-mcpu=apple-m4";;
    x86_64)   echo "-msse3";;
    haswell)  echo "-mavx -mavx2 -mfma" \
                   "-DULMBLAS_DGEMM_UKERNEL_openblas_haswell_4x8" \
                   "-DULMBLAS_SGEMM_UKERNEL_openblas_haswell_8x4" \
                   "-DULMBLAS_CGEMM_UKERNEL_openblas_haswell_8x2" \
                   "-DULMBLAS_ZGEMM_UKERNEL_openblas_haswell_4x2";;
    skylakex) echo "-mavx -mavx2 -mfma -mavx512f -mavx512dq" \
                   "-DULMBLAS_DGEMM_UKERNEL_openblas_skylakex_16x12" \
                   "-DULMBLAS_SGEMM_UKERNEL_openblas_skylakex_16x4" \
                   "-DULMBLAS_CGEMM_UKERNEL_openblas_skylakex_8x2" \
                   "-DULMBLAS_ZGEMM_UKERNEL_blis_zen4_12x4";;
    zen4)     echo "-mavx -mavx2 -mfma -mavx512f -mavx512dq" \
                   "-DULMBLAS_DGEMM_UKERNEL_blis_zen4_32x6" \
                   "-DULMBLAS_SGEMM_UKERNEL_openblas_skylakex_16x4" \
                   "-DULMBLAS_CGEMM_UKERNEL_openblas_skylakex_8x2" \
                   "-DULMBLAS_ZGEMM_UKERNEL_blis_zen4_12x4";;
    *)        echo "unknown variant $1" >&2; exit 1;;
    esac
}

if [ $# -eq 0 ]; then
    case $(uname -s)-$(uname -m) in
    Darwin-arm64)  set -- neon sme;;
    Linux-x86_64)  set -- x86_64 haswell skylakex zen4;;
    *)             echo "no variants for $(uname -s)-$(uname -m)" >&2; exit 1;;
    esac
fi

mkdir -p "$OUT/share/licenses/ulmBLAS" "$OUT/share/ulmBLAS"
cp "$SRC/LICENSE" "$OUT/share/licenses/ulmBLAS/LICENSE"
R=$OUT/share/ulmBLAS/build.txt
{
    c=${ULMBLAS_COMMIT:-$(cd "$SRC" && git rev-parse HEAD 2>/dev/null || echo unknown)}
    [ -n "$(cd "$SRC" && git status --porcelain --untracked-files=no 2>/dev/null)" ] && c="$c with local changes"
    echo "ulmBLAS $c"
    echo "built $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(uname -s) $(uname -m)"
    echo "CXX: $(${CXX:-c++} --version | head -1)"
    [ -n "${MACOSX_DEPLOYMENT_TARGET:-}" ] && echo "MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"
} > "$R"

for v in "$@"; do
    f=$(flags $v)
    b=$OUT/build/$v
    make -C "$SRC" -j$J lbt ARCHFLAGS="$f" OUTDIR="$OUT/lib/$v" \
         BUILD_DIR="$b" > "$LOGDIR/build-$v.log" 2>&1 \
        || { echo "build of $v failed, see $LOGDIR/build-$v.log" >&2; exit 1; }
    #  Linux: no debug information (mostly of the static libstdc++)
    [ "$(uname -s)" = Linux ] && strip --strip-debug "$OUT/lib/$v/libulmblas_lbt.so"
    echo "$v: ARCHFLAGS=$f" >> "$R"
done
rm -rf "$OUT/build"
cat "$R"
