# Note that this script can accept some limited command-line arguments, run
# `julia build_tarballs.jl --help` to see a usage message.
using BinaryBuilder, Pkg

const YGGDRASIL_DIR = "../.."
include(joinpath(YGGDRASIL_DIR, "fancy_toys.jl"))

name = "ulmBLAS"
version = v"0.3.0"

sources = [
    # draft: a commit of the branch windows-lapack (test build); the release
    # will use the tag v0.3.0
    GitSource("https://github.com/michael-lehn/ulmBLAS.git",
              "e01ff187fcba74c469831d9c16a47b87f78a8d07"),
]

# One library per instruction set, `libulmblas_VARIANT`, with the BLAS
# interface for libblastrampoline (ILP64, suffix `64_`), GEMM with
# arbitrary strides (`ulmblas_dgemm_strided`, ...) and the LAPACK functions
# ?getrf, ?potrf, solves and inverses with OpenMP tasks (OPENMP=1
# LAPACK=1; their number of threads: `ulmblas_set_num_threads_64_`).
# UlmBLAS.jl selects the variant for the CPU at run time.
script = raw"""
cd ${WORKSPACE}/srcdir/ulmBLAS

flags() {
    case $1 in
    neon)     echo "-DULMBLAS_NO_SME";;
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
    esac
}

for v in ${variants}; do
    build=${WORKSPACE}/build/${v}
    # The makefile picks these with `uname`, i.e. for the build host.
    if [[ ${target} == *-apple-* ]]; then
        link="-dynamiclib -Wl,-force_load,${build}/liblbtblas.a"
        link+=" -Wl,-force_load,${build}/liblbtlapack.a"
        # libomp of LLVMOpenMP_jll
        omp="OMPFLAGS=-fopenmp -I${includedir}"
        omplibs="OMPLIBS=-L${libdir} -lomp"
    elif [[ ${target} == *-mingw* ]]; then
        # MinGW-w64: all global symbols are exported (none is dllexport)
        link="-shared -Wl,--whole-archive ${build}/liblbtblas.a ${build}/liblbtlapack.a -Wl,--no-whole-archive"
        link+=" -static-libstdc++ -static-libgcc -Wl,--no-undefined"
        # libgomp of CompilerSupportLibraries_jll
        omp="OMPFLAGS=-fopenmp"
        omplibs="OMPLIBS=-fopenmp"
    else
        link="-shared -Wl,--whole-archive ${build}/liblbtblas.a ${build}/liblbtlapack.a -Wl,--no-whole-archive"
        # libgomp of CompilerSupportLibraries_jll
        omp="OMPFLAGS=-fopenmp"
        omplibs="OMPLIBS=-fopenmp"
        link+=" -static-libstdc++ -static-libgcc -Wl,--no-undefined -Wl,--strip-debug"
        # don't export the static libstdc++
        link+=" -Wl,--exclude-libs,libstdc++.a:libgcc.a:libgcc_eh.a"
    fi
    make -j${nproc} lbt CXX="${CXX}" CC="${CC}" ARCHFLAGS="$(flags ${v})" \
         OPENMP=1 LAPACK=1 "${omp}" "${omplibs}" \
         BUILD_DIR=${build} LBT_LIB=${build}/libulmblas_${v}.${dlext} LBT_LINK="${link}"
    install -Dvm 755 ${build}/libulmblas_${v}.${dlext} ${libdir}/libulmblas_${v}.${dlext}
done

# includes the licenses of the micro kernels from BLIS, AOCL-BLAS and OpenBLAS
install_license LICENSE
"""

platforms = supported_platforms()
# UlmBLAS.jl selects variants only on x86-64 and AArch64; other architectures
# (portable reference kernels only) are not built or tested yet.
filter!(p -> arch(p) in ("x86_64", "aarch64"), platforms)
# Not built or tested yet: the NEON variant on Linux.
filter!(p -> !(Sys.islinux(p) && arch(p) == "aarch64"), platforms)
# Not built or tested yet: the x86-64 variants on macOS.
filter!(p -> !(Sys.isapple(p) && arch(p) == "x86_64"), platforms)
# Windows: x86-64 only (MinGW-w64)
filter!(p -> !Sys.iswindows(p) || arch(p) == "x86_64", platforms)
# Not built or tested yet: FreeBSD and musl.
filter!(p -> !Sys.isfreebsd(p) && libc(p) != "musl", platforms)

# macOS: NEON for all Apple silicon.  The SME2 kernels of ulmBLAS (M4 and
# later) need the SME support routines of compiler-rt (LLVMCompilerRT_jll
# 20, not yet available) and will be added in a later version.
variants(p) = Sys.isapple(p) ? ["neon"] : ["x86_64", "haswell", "skylakex", "zen4"]

# Not loaded with the JLL: only some variants run on a given CPU.
products(p) = [LibraryProduct("libulmblas_$(v)", Symbol("libulmblas_$(v)"); dont_dlopen=true)
               for v in variants(p)]

#  OpenMP runtime: libgomp (GCC) on Linux and Windows, libomp (LLVM) on macOS
dependencies = [
    Dependency(PackageSpec(name="CompilerSupportLibraries_jll", uuid="e66e0078-7015-5450-92f7-15fbd957f2ae");
               platforms=filter(!Sys.isapple, platforms)),
    Dependency(PackageSpec(name="LLVMOpenMP_jll", uuid="1d63c593-3942-5779-bab2-d838dc0a180e");
               platforms=filter(Sys.isapple, platforms)),
]

for platform in platforms
    should_build_platform(triplet(platform)) || continue
    # GCC 11 is the oldest GCC without broad slowdowns (GCC 10: dgemv up to 27%,
    # GCC 6: zgemm up to 2.8x in the SSE3 variant); GCC 5 fails.  With GCC 11
    # only zgemv without transposition is slower than with GCC 14 (about 3x).
    # libstdc++ is linked statically.
    # The OpenMP tasks need OpenMP 5.0 (iterators in depend clauses): GCC 9
    # or newer, also with MinGW-w64.
    compilers = Sys.isapple(platform) ? (;) : (; preferred_gcc_version=v"11")
    build_tarballs(ARGS, name, version, sources,
                   "variants=\"$(join(variants(platform), ' '))\"\n" * script,
                   [platform], products(platform), dependencies;
                   julia_compat="1.9",
                   # The variants choose their instruction sets.
                   lock_microarchitecture=false,
                   compilers...)
end
