#  Run in a fresh process by runtests.jl with ULMBLAS_LIBRARY set to a file
#  that does not exist: `using UlmBLAS` warns and leaves the BLAS of Julia
#  unchanged.
using LinearAlgebra, Test
config = BLAS.get_config()
before = [l.libname for l in config.loaded_libs]
using UlmBLAS
@testset "no library" begin
    @test [l.libname for l in BLAS.get_config().loaded_libs] == before
    @test UlmBLAS.library_path() == ""
    A = rand(50, 50); B = rand(50, 50)
    @test A * B ≈ [sum(A[i, l] * B[l, j] for l in 1:50) for i in 1:50, j in 1:50]
end
