#!/bin/sh
#
#  linux-container.sh SRC OUT: the Linux x86-64 libraries (build.sh, all
#  four variants) for the oldest supported glibc, built in the container
#
#      quay.io/pypa/manylinux_2_28_x86_64 (AlmaLinux 8, glibc 2.28)
#
#  with GCC 14 (gcc-toolset-14), then checked: the libraries need no
#  symbol of glibc newer than 2.28, no symbol of libstdc++ (linked
#  statically) and only libc, libm and the dynamic loader.  Run inside the
#  container (.github/workflows/build-linux.yml does it), e.g.
#
#      docker run --rm -v $PWD:/w -w /w IMAGE@DIGEST \
#          sh UlmBLAS.jl/build/linux-container.sh ulmBLAS out-linux
#
#  IMAGE (environment) names the image for build.txt.
#
set -eu
SRC=$1; OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
GLIBC_MAX=2.28

#  part of the image (manylinux_2_28 2026.10.31-1: GCC 14.2.1)
[ -x /opt/rh/gcc-toolset-14/root/usr/bin/g++ ] ||
    dnf install -y -q gcc-toolset-14-gcc-c++ gcc-toolset-14-binutils \
                      gcc-toolset-14-libstdc++-devel > /dev/null
. /opt/rh/gcc-toolset-14/enable

ULMBLAS_BUILD_ENV="container ${IMAGE:-manylinux_2_28_x86_64}, $(cat /etc/redhat-release), glibc $(ldd --version | sed -n '1s/.* //p'), $(rpm -q gcc-toolset-14-gcc-c++)" \
CXX=g++ CC=gcc sh "$HERE/build.sh" "$SRC" "$OUT"

status=0
for L in "$OUT"/lib/*/libulmblas_lbt.so; do
    glibc=$(objdump -T "$L" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV)
    newest=$(echo "$glibc" | tail -1)
    needed=$(objdump -p "$L" | awk '/NEEDED/{print $2}' | sort | tr '\n' ' ')
    cxx=$(objdump -T "$L" | grep -cE 'GLIBCXX|CXXABI' || true)
    echo "$L: GLIBC $(echo $glibc), NEEDED $needed, GLIBCXX/CXXABI symbols $cxx"
    if [ "$(printf '%s\n%s\n' "$newest" "$GLIBC_MAX" | sort -V | tail -1)" != "$GLIBC_MAX" ]; then
        echo "error: needs GLIBC_$newest (newer than $GLIBC_MAX)"; status=1
    fi
    [ "$cxx" = 0 ] || { echo "error: needs libstdc++"; status=1; }
    for n in $needed; do
        case $n in
        libc.so.6|libm.so.6|ld-linux-x86-64.so.2) ;;
        *) echo "error: needs $n"; status=1;;
        esac
    done
done
exit $status
