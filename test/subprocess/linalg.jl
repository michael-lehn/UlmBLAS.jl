#  Run in a fresh process by runtests.jl: test files of LinearAlgebra
#  (LINALG_FILES) with UlmBLAS loaded.  Every call of a routine that no
#  library provides is counted (as in the tests of MKL.jl and
#  AppleAccelerate.jl) and makes the test fail.
using LinearAlgebra, Test, Random

const MISSING = Ref(0)
missing_function() = (MISSING[] += 1; nothing)
BLAS.lbt_set_default_func(@cfunction(missing_function, Cvoid, ()))

using UlmBLAS
UlmBLAS.forward()

const DIR = joinpath(dirname(pathof(LinearAlgebra)), "..", "test")
@testset "LinearAlgebra with UlmBLAS" begin
    @test any(l -> occursin("ulmblas", l.libname), BLAS.get_config().loaded_libs)
    for f in split(get(ENV, "LINALG_FILES", "blas.jl matmul.jl"))
        @testset "$f" begin
            cd(() -> include(joinpath(DIR, f)), DIR)
        end
    end
    @test MISSING[] == 0
end
