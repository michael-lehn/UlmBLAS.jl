using LinearAlgebra, Libdl, Test
const LIBS_BEFORE = [l.libname for l in BLAS.get_config().loaded_libs]
using UlmBLAS

#  a test script in a fresh Julia process (for tests that need the state
#  before `using UlmBLAS`, or that could end the process); returns the
#  process, stdout and stderr
function run_script(file, env...; threads=1)
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$threads
           --project=$(Base.active_project()) $(joinpath(@__DIR__, "subprocess", file))`
    out = IOBuffer(); err = IOBuffer()
    p = run(pipeline(ignorestatus(addenv(cmd, env...)); stdout=out, stderr=err))
    return p, String(take!(out)), String(take!(err))
end
function script_passes(file, env...; threads=1)
    p, out, err = run_script(file, env...; threads)
    success(p) || println(out, err)
    return success(p)
end

@testset "UlmBLAS" begin

@testset "loading" begin
    @test isfile(UlmBLAS.library_path())
    #  `using UlmBLAS` does not change the BLAS of Julia
    @test [l.libname for l in BLAS.get_config().loaded_libs] == LIBS_BEFORE
    @test UlmBLAS.variant() in UlmBLAS.variants()
    @info "UlmBLAS: library $(UlmBLAS.library_path()) ($(UlmBLAS.chosen[])), " *
          "variants of this CPU: $(join(UlmBLAS.variants(), ", "))"
end

#  UlmBLAS.mul! before forward()
include("strided.jl")

@testset "forwarding" begin
    lib = UlmBLAS.library_path()
    @test UlmBLAS.forward() > 140
    @test [l.libname for l in BLAS.get_config().loaded_libs] == [LIBS_BEFORE; lib]
    @test UlmBLAS.forward() > 140              # again: no change
    @test [l.libname for l in BLAS.get_config().loaded_libs] == [LIBS_BEFORE; lib]
    #  forward() calls are serialized (lbt_forward is not thread-safe): a
    #  call waits while another one holds the lock
    lock(UlmBLAS.FORWARD_LOCK)
    t = Threads.@spawn UlmBLAS.forward()
    sleep(0.5)
    @test !istaskdone(t)
    unlock(UlmBLAS.FORWARD_LOCK)
    @test fetch(t) > 140
    #  concurrent calls: one entry in the list of libraries
    @sync for _ in 1:8
        Threads.@spawn UlmBLAS.forward()
    end
    @test [l.libname for l in BLAS.get_config().loaded_libs] == [LIBS_BEFORE; lib]
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

#  routines of LinearAlgebra.BLAS that ulmBLAS does not provide: they stay
#  with the library loaded before (OpenBLAS)
@testset "fallback $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A = rand(T, 9, 6); B = rand(T, 6, 9); C0 = rand(T, 9, 9)
    for uplo in ('U', 'L')
        isdefined(BLAS, :gemmt!) || break          # Julia 1.11 and later
        C = copy(C0)
        BLAS.gemmt!(uplo, 'N', 'N', T(2), A, B, T(3), C)
        R = 2 * gemm_ref(A, B) + 3 * C0
        in_tri = uplo == 'U' ? triu : tril
        out_tri = uplo == 'U' ? (M -> tril(M, -1)) : (M -> triu(M, 1))
        @test in_tri(C) ≈ in_tri(R)
        @test out_tri(C) == out_tri(C0)
    end
    x = rand(T, 11); y = rand(T, 11); y0 = copy(y)
    BLAS.axpby!(T(2), x, T(-1), y)
    @test y ≈ 2x - y0
    if T <: Complex
        @test BLAS.dotc(11, x, 1, y0, 1) ≈ sum(conj.(x) .* y0)
        @test BLAS.dotu(x, y0) ≈ sum(x .* y0)
        S = A[1:6, :] + transpose(A[1:6, :]); v = rand(T, 6)
        @test BLAS.symv('U', S, v) ≈ S * v
        M = zeros(T, 6, 6); BLAS.syr!('U', one(T), v, M)
        @test triu(M) ≈ triu(v * transpose(v))
        u = rand(T, 5); w = rand(T, 5); u0, w0 = copy(u), copy(w)
        c = real(T)(0.6); s = T(0.48 + 0.64im)
        BLAS.rot!(5, u, 1, w, 1, c, s)
        @test u ≈ c * u0 + s * w0
        @test w ≈ -conj(s) * u0 + c * w0
    else
        @test BLAS.dot(11, x, 1, y0, 1) ≈ sum(x .* y0)
    end
end

#  UlmBLAS.mul! after forward()
@testset "strided after forward()" begin
    P = rand(40, 30); A = view(P, 1:2:40, :); B = rand(30, 25); C = zeros(20, 25)
    @test UlmBLAS.mul!(C, A, B) ≈ Matrix(A) * B
end

#  in fresh processes
@testset "routing of all symbols, loading again" begin
    @test script_passes("routing.jl")
end
@testset "invalid argument" begin
    p, out, err = run_script("errors.jl")
    @test success(p)
    @test occursin("DGEMM", err)
end
@testset "threads" begin
    @test script_passes("threads.jl"; threads=4)
end
#  every build of the artifact this CPU can run (not with ULMBLAS_LIBRARY)
@testset "variant $v" for v in (isempty(get(ENV, "ULMBLAS_LIBRARY", "")) ?
                                UlmBLAS.variants() : String[])
    @test script_passes("variant.jl", "ULMBLAS_VARIANT" => v)
end
@testset "ULMBLAS_LIBRARY does not exist" begin
    p, out, err = run_script("init.jl", "ULMBLAS_LIBRARY" => "/nonexistent/libulmblas_lbt")
    @test success(p)
    @test occursin("does not exist", err)
end
@testset "ULMBLAS_LIBRARY cannot be loaded" begin
    p, out, err = run_script("init.jl", "ULMBLAS_LIBRARY" => joinpath(@__DIR__, "runtests.jl"))
    @test success(p)
    @test occursin("cannot be loaded", err)
end
#  test files of LinearAlgebra; ULMBLAS_TEST_LINALG=1 adds more (several
#  minutes)
@testset "LinearAlgebra" begin
    files = get(ENV, "ULMBLAS_TEST_LINALG", "") == "1" ?
            "blas.jl matmul.jl lapack.jl triangular.jl dense.jl symmetric.jl" :
            "blas.jl matmul.jl"
    if isdir(joinpath(dirname(pathof(LinearAlgebra)), "..", "test"))
        @test script_passes("linalg.jl", "LINALG_FILES" => files)
    else
        @info "no tests of LinearAlgebra in this Julia installation"
    end
end

end
