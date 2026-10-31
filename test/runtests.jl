using LinearAlgebra, Libdl, Test
using ulmBLAS

@testset "ulmBLAS" begin

@testset "forwarding" begin
    lib = ulmBLAS.library_path()
    @test isfile(lib)
    @test any(l -> l.libname == lib, BLAS.get_config().loaded_libs)
    h = dlopen(lib)
    for f in ("dgemm", "sgemm", "zgemm", "cgemm", "ddot", "zdotc", "idamax",
              "dtrsm", "dgemv", "dsyrk", "zherk", "drot")
        @test UInt(BLAS.lbt_get_forward(f * "_", :ilp64)) == UInt(dlsym(h, f * "_64_"))
    end
end

#  results against Julia's generic implementations (no BLAS)
gemm_ref(A, B) = [sum(A[i, l] * B[l, j] for l in axes(A, 2)) for i in axes(A, 1), j in axes(B, 2)]

@testset "$T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    tol = 50 * eps(real(T))
    for (m, n, k) in ((1, 1, 1), (7, 5, 3), (64, 64, 64), (301, 157, 233))
        A = rand(T, m, k); B = rand(T, k, n); C = rand(T, m, n)
        @test norm(A * B - gemm_ref(A, B)) <= tol * k * norm(A) * norm(B)
        D = copy(C); BLAS.gemm!('T', 'N', T(2), copy(transpose(A)), B, T(-1), D)
        @test norm(D - (2 * gemm_ref(A, B) - C)) <= tol * k * (norm(A) * norm(B) + norm(C))
    end
    x = rand(T, 1000); y = rand(T, 1000)
    xy = T <: Complex ? BLAS.dotc(x, y) : BLAS.dot(x, y)
    @test abs(xy - sum(conj.(x) .* y)) <= tol * 1000 * norm(x) * norm(y)
    @test abs(BLAS.nrm2(x) - sqrt(sum(abs2, x))) <= tol * 1000 * norm(x)
    @test BLAS.iamax(x) == argmax(abs.(real.(x)) .+ abs.(imag.(x)))
    @test BLAS.asum(x) ≈ sum(abs.(real.(x)) .+ abs.(imag.(x))) rtol = tol * 1000
    A = rand(T, 200, 200) + 200I; b = rand(T, 200, 7)
    X = UpperTriangular(A) \ b
    @test norm(UpperTriangular(A) * X - b) <= tol * 200 * norm(A) * norm(X)
end

include("strided.jl")

#  the tests of LinearAlgebra for BLAS and matrix products (several minutes)
if get(ENV, "ULMBLAS_TEST_LINALG", "") == "1"
    dir = joinpath(Sys.STDLIB, "LinearAlgebra", "test")
    @testset "LinearAlgebra $f" for f in ("blas.jl", "matmul.jl")
        cd(dir) do
            include(joinpath(dir, f))
        end
    end
end

end
