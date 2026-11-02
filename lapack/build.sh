#!/bin/sh
#
#  build.sh OUT [JULIA]: Netlib LAPACK 3.12.1 as a shared library
#
#      OUT/lib/liblapack.{so,dylib}
#
#  linked against the libblastrampoline of the Julia installation JULIA
#  (default: julia in PATH), with a record of the build in OUT/build.txt.
#
#  - ILP64 with the suffix 64_ (dgetrf_64_, ...), the interface of the BLAS
#    and LAPACK of Julia: the extended _64 API of LAPACK's CMake build
#    (BUILD_INDEX64_EXT_API, -fdefault-integer-8, every routine renamed by
#    SRC/lapack_64.h).  The library also contains the LP64 routines without
#    suffix (part of that build); they are not used.
#  - Every BLAS routine LAPACK calls (dgemm_64_, dtrsm_64_, ...) is
#    imported from libblastrampoline, so it runs in the library
#    libblastrampoline forwards it to: ulmBLAS after UlmBLAS.forward().
#  - The only change of the sources: STOP removed from SRC/xerbla.f (as in
#    Yggdrasil's LAPACK_jll), an illegal argument must not end Julia.
#
#  Needs gfortran, a C compiler and CMake >= 3.18.  J = parallel jobs.
#
set -eu
OUT=$1; JULIA=${2:-julia}
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd)
J=${J:-8}
VERSION=3.12.1
SHA256=2ca6407a001a474d4d4d35f3a61550156050c48016d949f0da0529c0aa052422
URL=https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v$VERSION.tar.gz

LBT=$("$JULIA" --startup-file=no -e \
      'using libblastrampoline_jll; print(libblastrampoline_jll.libblastrampoline_path)')
[ -f "$LBT" ] || { echo "no libblastrampoline: $LBT" >&2; exit 1; }

TARBALL=$OUT/src/lapack-$VERSION.tar.gz
mkdir -p "$OUT/src"
[ -f "$TARBALL" ] || curl -fsSL -o "$TARBALL" "$URL"
if command -v sha256sum >/dev/null; then
    echo "$SHA256  $TARBALL" | sha256sum -c -
else
    echo "$SHA256  $TARBALL" | shasum -a 256 -c -
fi
SRC=$OUT/src/lapack-$VERSION
[ -d "$SRC" ] || tar -xzf "$TARBALL" -C "$OUT/src"
sed '/^[ \t]*STOP/d' "$SRC/SRC/xerbla.f" > "$SRC/SRC/xerbla.f.new"
mv "$SRC/SRC/xerbla.f.new" "$SRC/SRC/xerbla.f"

cmake -S "$SRC" -B "$OUT/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$OUT" \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_INSTALL_RPATH="$(dirname "$LBT")" \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_INDEX64_EXT_API=ON \
      -DBUILD_TESTING=OFF \
      -DTEST_FORTRAN_COMPILER=OFF \
      -DCBLAS=OFF -DLAPACKE=OFF \
      -DUSE_OPTIMIZED_BLAS=ON \
      -DBLAS_LIBRARIES="$LBT" > "$OUT/cmake.log"
cmake --build "$OUT/build" -j "$J" > "$OUT/build.log"
cmake --install "$OUT/build" > "$OUT/install.log"
[ -f "$OUT/lib/libblas.so" ] || [ -f "$OUT/lib/libblas.dylib" ] &&
    { echo "a reference BLAS was built: not linked against libblastrampoline" >&2; exit 1; }

case $(uname -s) in
Darwin) LIB=$OUT/lib/liblapack.dylib; DEPS=$(otool -L "$LIB");;
*)      LIB=$OUT/lib/liblapack.so;    DEPS=$(ldd "$LIB");;
esac
{
    echo "Netlib LAPACK $VERSION ($URL, sha256 $SHA256)"
    echo "patch:     STOP removed from SRC/xerbla.f"
    echo "julia:     $("$JULIA" --startup-file=no -e 'print(VERSION)') ($JULIA)"
    echo "BLAS:      $LBT"
    echo "gfortran:  $(gfortran --version | head -1)"
    echo "cmake:     $(cmake --version | head -1)"
    echo "flags:     $(grep -E '^CMAKE_Fortran_FLAGS(_RELEASE)?:' "$OUT/build/CMakeCache.txt" | tr '\n' ' ')"
    echo "host:      $(uname -sm)"
    echo "library:   $LIB"
    echo "$DEPS"
} > "$OUT/build.txt"
cat "$OUT/build.txt"
