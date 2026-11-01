#  Run in a fresh process by runtests.jl with ULMBLAS_LIBRARY set to a file
#  that does not exist or is no library: `using UlmBLAS` warns, the BLAS of Julia is
#  unchanged, UlmBLAS.mul! and UlmBLAS.forward() raise errors.
using LinearAlgebra, Test
before = [l.libname for l in BLAS.get_config().loaded_libs]
using UlmBLAS
@testset "no library" begin
    @test UlmBLAS.library_path() == ""
    @test_throws ErrorException UlmBLAS.mul!(zeros(2, 2), rand(2, 2), rand(2, 2))
    @test_throws ErrorException UlmBLAS.forward()
    @test [l.libname for l in BLAS.get_config().loaded_libs] == before
    A = rand(50, 50); B = rand(50, 50)
    @test A * B ≈ [sum(A[i, l] * B[l, j] for l in 1:50) for i in 1:50, j in 1:50]
end
