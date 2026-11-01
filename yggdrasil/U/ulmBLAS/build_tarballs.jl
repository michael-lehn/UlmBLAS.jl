# Note that this script can accept some limited command-line arguments, run
# `julia build_tarballs.jl --help` to see a usage message.
using BinaryBuilder, Pkg

name = "ulmBLAS"
version = v"0.1.0"

sources = [
    GitSource("https://github.com/michael-lehn/ulmBLAS.git",
              "6c554eec707e22f118bae5dcaea10ad9945aac54"),
]

# One library per instruction set, `libulmblas_VARIANT`, with the BLAS
# interface for libblastrampoline (ILP64, suffix `64_`) and GEMM with
# arbitrary strides (`ulmblas_dgemm_strided`, ...).  UlmBLAS.jl selects the
# variant for the CPU at run time.
script = raw"""
cd ${WORKSPACE}/srcdir/ulmBLAS

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
    esac
}

for v in ${variants}; do
    build=${WORKSPACE}/build/${v}
    # The makefile picks these with `uname`, i.e. for the build host.
    if [[ ${target} == *-apple-* ]]; then
        link="-dynamiclib -Wl,-force_load,${build}/liblbtblas.a"
    else
        link="-shared -Wl,--whole-archive ${build}/liblbtblas.a -Wl,--no-whole-archive"
        link+=" -static-libstdc++ -static-libgcc -Wl,--no-undefined -Wl,--strip-debug"
        # don't export the static libstdc++
        link+=" -Wl,--exclude-libs,libstdc++.a:libgcc.a:libgcc_eh.a"
    fi
    make -j${nproc} lbt CXX="${CXX}" CC="${CC}" ARCHFLAGS="$(flags ${v})" \
         BUILD_DIR=${build} LBT_LIB=${build}/libulmblas_${v}.${dlext} LBT_LINK="${link}"
    install -Dvm 755 ${build}/libulmblas_${v}.${dlext} ${libdir}/libulmblas_${v}.${dlext}
done

# includes the licenses of the micro kernels from BLIS, AOCL-BLAS and OpenBLAS
install_license LICENSE
"""

# The platforms of UlmBLAS.jl: the micro kernels are for x86-64 and Apple
# silicon (SME2 and NEON), the makefiles for Linux and macOS.
platforms = filter(supported_platforms()) do p
    (arch(p) == "x86_64" && Sys.islinux(p) && libc(p) == "glibc") ||
        (arch(p) == "aarch64" && Sys.isapple(p))
end

variants(p) = Sys.isapple(p) ? ["neon", "sme"] : ["x86_64", "haswell", "skylakex", "zen4"]

# Not loaded with the JLL: only some variants run on a given CPU.
products(p) = [LibraryProduct("libulmblas_$(v)", Symbol("libulmblas_$(v)"); dont_dlopen=true)
               for v in variants(p)]

dependencies = Dependency[]

for platform in platforms
    should_build_platform(triplet(platform)) || continue
    # GCC 14 as for the libraries of UlmBLAS.jl; libstdc++ is linked statically.
    compilers = Sys.islinux(platform) ? (; preferred_gcc_version=v"14") : (;)
    build_tarballs(ARGS, name, version, sources,
                   "variants=\"$(join(variants(platform), ' '))\"\n" * script,
                   [platform], products(platform), dependencies;
                   julia_compat="1.9",
                   # The variants choose their instruction sets.
                   lock_microarchitecture=false,
                   compilers...)
end
