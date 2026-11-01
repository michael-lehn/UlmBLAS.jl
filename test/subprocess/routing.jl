#  Run in a fresh process by runtests.jl: the routing of every symbol of
#  libblastrampoline (BLAS, CBLAS, LAPACK, both interfaces) before and
#  after `using UlmBLAS` (no change) and after `UlmBLAS.forward()`.
using LinearAlgebra, Libdl, Test

const NAMES = BLAS.get_config().exported_symbols
const DEFAULT = UInt(BLAS.lbt_get_default_func())   # no library provides it
forward(n, i) = UInt(BLAS.lbt_get_forward(n, i))
snapshot() = Dict((n, i) => forward(n, i) for n in NAMES, i in (:ilp64, :lp64))
backing(n) = (b = BLAS.lbt_find_backing_library(n, :ilp64); b === nothing ? "" : b.libname)

before = snapshot()
backing_before = Dict(n => backing(n) for n in NAMES)
nlibs = length(BLAS.get_config().loaded_libs)

using UlmBLAS

@testset "using UlmBLAS changes nothing" begin
    @test snapshot() == before
    @test length(BLAS.get_config().loaded_libs) == nlibs
end

UlmBLAS.forward()
after = snapshot()

#  Windows: libblastrampoline returns the double complex results of zdotc_
#  and zdotu_ through an adapter (complex return style
#  float_normal_double_argument); after forward() the adapter still calls
#  the library loaded before (OpenBLAS), see docs/details.md
const ADAPTED = BLAS.get_config().loaded_libs[end].complex_retstyle ===
                :float_normal_double_argument ? ["zdotc_", "zdotu_"] : String[]
const ULM = dlopen(UlmBLAS.library_path())
#  address of the symbol in the library of ulmBLAS (dgemm_64_, cblas_ddot64_)
ulm(n) = (p = dlsym_e(ULM, n * "64_"); p == C_NULL ? UInt(0) : UInt(p))

@testset "routing of $(length(NAMES)) symbols" begin
    #  nothing that was provided before is missing now
    lost = [k for (k, v) in before if v != DEFAULT && after[k] == DEFAULT]
    @test isempty(lost)
    #  ILP64: what ulmBLAS provides goes to ulmBLAS, everything else is
    #  unchanged; LP64 is unchanged
    wrong = [n for n in NAMES if !(n in ADAPTED) && after[(n, :ilp64)] !=
             (ulm(n) != 0 ? ulm(n) : before[(n, :ilp64)])]
    @test isempty(wrong)
    @test all(after[(n, :ilp64)] == before[(n, :ilp64)] for n in ADAPTED)
    @test all(after[(n, :lp64)] == before[(n, :lp64)] for n in NAMES)
    @test count(n -> ulm(n) != 0, NAMES) >= 149
    #  BLAS routines of LinearAlgebra that ulmBLAS does not provide keep
    #  their library
    for p in "sdcz", n in ("gemmt_", "axpby_")
        @test ulm(p * n) == 0
        @test backing(p * n) == backing_before[p * n]
        #  ?gemmt: in OpenBLAS (and BLAS.gemmt!) with Julia 1.11 and later
        (n == "axpby_" || isdefined(BLAS, :gemmt!)) && @test backing(p * n) != ""
    end
    for n in ("crot_", "zrot_", "csymv_", "zsymv_", "csyr_", "zsyr_",
              "cblas_ddot", "cblas_sdot", "cblas_zdotc_sub", "cblas_cdotc_sub",
              "cblas_zdotu_sub", "cblas_cdotu_sub", "dgetrf_", "zheevd_")
        @test backing(n) == backing_before[n] != ""
    end
    for n in ("dgemm_", "zgemm_", "strsm_", "ctrmm_", "dgemv_", "zherk_",
              "daxpy_", "idamax_", "dnrm2_", "zdotc_")
        @test occursin("ulmblas", backing(n))
    end
end

@testset "loading again" begin
    t = snapshot()
    UlmBLAS.forward()
    @eval using UlmBLAS
    @test snapshot() == t
    @test length(BLAS.get_config().loaded_libs) == nlibs + 1
end
