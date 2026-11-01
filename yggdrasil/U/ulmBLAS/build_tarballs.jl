# Note that this script can accept some limited command-line arguments, run
# `julia build_tarballs.jl --help` to see a usage message.
using BinaryBuilder, BinaryBuilderBase, Pkg

name = "ulmBLAS"
version = v"0.1.0"

# ulmBLAS has no releases; this is the commit the libraries of UlmBLAS.jl
# (release libulmblas-6c554eec of https://github.com/michael-lehn/UlmBLAS.jl)
# are built from.
sources = [
    GitSource("https://github.com/michael-lehn/ulmBLAS.git",
              "6c554eec707e22f118bae5dcaea10ad9945aac54"),
    # SME support routines (`__arm_tpidr2_save`, ...) for macOS: Apple's
    # clang links them from its libclang_rt.osx.a, LLVMCompilerRT_jll
    # (17.0.6) doesn't have them yet.
    ArchiveSource("https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.8/compiler-rt-20.1.8.src.tar.xz",
                  "15277402f6fd63397c0917a5c7171cda82d16d226094b828c1ed0f58f73b9c69"),
]

# One library per kind of CPU, `libulmblas_VARIANT`; the variant differs in
# the instruction set and the micro kernels of S, D, C and ZGEMM.
# UlmBLAS.jl picks the one for the CPU it runs on and forwards
# libblastrampoline to it.  Each library has the BLAS interface with 64-bit
# integers and the suffix `64_` (`dgemm_64_`, as OpenBLAS for Julia) and
# GEMM with arbitrary strides (`ulmblas_dgemm_strided`, ...).
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

case ${target} in
    x86_64-linux-gnu*)    variants="x86_64 haswell skylakex zen4";;
    aarch64-apple-darwin*) variants="neon sme";;
esac

for v in ${variants}; do
    lib=libulmblas_${v}.${dlext}
    out=${WORKSPACE}/out/${v}
    build=${WORKSPACE}/build/${v}
    if [[ ${target} == *apple* ]]; then
        smeabi=
        if [[ ${v} == sme ]]; then
            # hidden, as in libclang_rt.osx.a
            mkdir -p ${WORKSPACE}/build
            smeabi=${WORKSPACE}/build/arm_apple_sme_abi.o
            cc -c -x assembler-with-cpp -DVISIBILITY_HIDDEN \
               ${WORKSPACE}/srcdir/compiler-rt-*/lib/builtins/aarch64/arm_apple_sme_abi.s \
               -o ${smeabi}
        fi
        # The makefile decides between .so and .dylib with `uname`, which
        # is the build machine here.
        make -j${nproc} lbt CXX=c++ CC=cc ARCHFLAGS="$(flags ${v})" \
             OUTDIR=${out} BUILD_DIR=${build} \
             LBT_LIB=${out}/${lib} \
             LBT_LINK="-dynamiclib -install_name @rpath/${lib} -Wl,-force_load,${build}/liblbtblas.a ${smeabi}"
    else
        # libstdc++ is linked statically; don't export its symbols.
        make -j${nproc} lbt CXX=c++ CC=cc ARCHFLAGS="$(flags ${v})" \
             OUTDIR=${out} BUILD_DIR=${build} \
             LDFLAGS="-Wl,--exclude-libs,libstdc++.a:libgcc.a:libgcc_eh.a"
        mv ${out}/libulmblas_lbt.${dlext} ${out}/${lib}
        patchelf --set-soname ${lib} ${out}/${lib}
        # mostly the debug information of the static libstdc++
        strip --strip-debug ${out}/${lib}
    fi
    install -Dvm 755 ${out}/${lib} ${libdir}/${lib}
done

# ulmBLAS and the micro kernels from BLIS, AOCL-BLAS and OpenBLAS
install_license LICENSE
if [[ ${target} == *apple* ]]; then
    install_license ${WORKSPACE}/srcdir/compiler-rt-*/LICENSE.TXT
fi
"""

# The libraries are not loaded when the JLL is loaded: a library for a newer
# instruction set may not even load on an older CPU.
variant_products(variants) = [
    LibraryProduct("libulmblas_$(v)", Symbol("libulmblas_$(v)"); dont_dlopen=true)
    for v in variants
]

# Only the C++ runtime of the system (macOS) or none at all: on Linux the
# makefile links libstdc++ statically, so the libraries only depend on libc
# and libm.
dependencies = Dependency[]

platform_args = filter(arg -> !startswith(arg, "--"), ARGS)
option_args = filter(arg -> startswith(arg, "--"), ARGS)
non_register_option_args = filter(arg -> arg != "--register", option_args)

builds = [
    (Platform("x86_64", "linux"; libc="glibc"),
     variant_products(["x86_64", "haswell", "skylakex", "zen4"])),
    (Platform("aarch64", "macos"),
     variant_products(["neon", "sme"])),
]
if !isempty(platform_args)
    requested = BinaryBuilderBase.parse_platform.(split(only(platform_args), ","))
    filter!(b -> any(p -> platforms_match(b[1], p), requested), builds)
end

for (n, (platform, products)) in enumerate(builds)
    # Register only after the last build.
    args = n == length(builds) ? option_args : non_register_option_args
    # Linux: GCC 14 (the compiler of the benchmark campaigns of ulmBLAS);
    # macOS: clang 19 for SME2 (ACLE keyword attributes, -mcpu=apple-m4).
    compilers = Sys.islinux(platform) ? (; preferred_gcc_version=v"14") :
                                         (; preferred_llvm_version=v"19")
    build_tarballs(args, name, version, sources, script, [platform], products, dependencies;
                   julia_compat="1.9",
                   # -mavx512f etc. select the micro kernels of each variant
                   lock_microarchitecture=false,
                   compilers...)
end
