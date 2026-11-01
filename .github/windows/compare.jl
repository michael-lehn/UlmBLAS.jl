#  ulmBLAS against OpenBLAS: the same products with the BLAS of Julia
#  (OpenBLAS) and, after UlmBLAS.forward(), with ulmBLAS; UlmBLAS.mul! with
#  arbitrary (also negative) strides against OpenBLAS on copies.
using LinearAlgebra, Random, Test, UlmBLAS
Random.seed!(1)
println("library ", UlmBLAS.library_path(), ", variants ", UlmBLAS.variants())

rel(X, Y) = norm(X - Y) / max(norm(Y), floatmin(real(eltype(Y))))
ops(A, B, C, U, x) = (A * B, A' * C, UpperTriangular(U) \ B, Hermitian(A' * A) * B,
                      A * x, B' * A', 3A * B + 2C)

types = (Float32, Float64, ComplexF32, ComplexF64)
data = Dict(T => (rand(T, 301, 263), rand(T, 263, 157), rand(T, 301, 157),
                  rand(T, 263, 263) + 263I, rand(T, 263)) for T in types)
ref = Dict(T => ops(data[T]...) for T in types)
strided = Dict(T => (P = rand(T, 700, 500);
                     (view(P, 2:2:600, 3:3:450), view(P, 699:-2:401, 1:150),
                      zeros(T, 300, 150))) for T in types)

@testset "ulmBLAS against OpenBLAS" begin
    @test UlmBLAS.forward() > 140
    cfg = BLAS.get_config()
    @test cfg.loaded_libs[end].libname == UlmBLAS.library_path()
    @test cfg.loaded_libs[end].interface == :ilp64
    for T in types
        tol = 100 * eps(real(T)) * 263
        for (i, (X, Y)) in enumerate(zip(ops(data[T]...), ref[T]))
            r = rel(X, Y)
            r <= tol || println("$T op $i: relative difference $r")
            @test r <= tol
        end
        A, B, C = strided[T]                    # B: negative row stride
        UlmBLAS.mul!(C, A, transpose(B))
        @test rel(C, Matrix(A) * transpose(Matrix(B))) <= tol
        C2 = rand(T, 300, 150); C0 = copy(C2)
        UlmBLAS.mul!(C2, A, B', T(2), T(-1))
        @test rel(C2, 2 * Matrix(A) * Matrix(B)' - C0) <= tol
    end
end
