#  Run in a fresh process by runtests.jl for every variant this CPU can run
#  (ULMBLAS_VARIANT): the library of the variant from the artifact is
#  loaded and computes correct results.
using LinearAlgebra, Test
using UlmBLAS

const V = ENV["ULMBLAS_VARIANT"]
@testset "variant $V" begin
    @test UlmBLAS.chosen[] == V
    @test occursin(joinpath("lib", V, UlmBLAS.LIBNAME), UlmBLAS.library_path())
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (5, 64, 301)
        P = rand(T, 2n, n); A = view(P, 1:2:2n, :); B = rand(T, n, n)
        C = zeros(T, n, n)
        @test UlmBLAS.mul!(C, A', B) ≈ Matrix(A)' * B
    end
    UlmBLAS.forward()
    @test any(l -> l.libname == UlmBLAS.library_path(), BLAS.get_config().loaded_libs)
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (5, 64, 301)
        A = rand(T, n, n); B = rand(T, n, n)
        @test A * B ≈ [sum(A[i, l] * B[l, j] for l in 1:n) for i in 1:n, j in 1:n]
        U = UpperTriangular(A + n * I)
        @test U * (U \ B) ≈ B
    end
end
