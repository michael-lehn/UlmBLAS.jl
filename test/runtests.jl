using LinearAlgebra, Libdl, Test, BFloat16s
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
    #  only the library of the selected variant is loaded
    @test basename.(filter(contains("libulmblas_"), Libdl.dllist())) ==
          [basename(UlmBLAS.library_path())]
    @info "UlmBLAS: library $(UlmBLAS.library_path()) ($(UlmBLAS.chosen[])), " *
          "variants of this CPU: $(join(UlmBLAS.variants(), ", "))"
end

@testset "AArch64 variants" begin
    sysctls(kv...) = name -> get(Dict(kv...), name, nothing)
    m4 = ("hw.optional.arm.FEAT_SME2" => 1, "hw.optional.arm.FEAT_SME_F64F64" => 1,
          "hw.optional.arm.sme_max_svl_b" => 64)
    @test UlmBLAS.arm_variants(sysctls(m4...)) == ["sme", "neon"]
    #  M1 to M3 (without the SME sysctls on older macOS)
    @test UlmBLAS.arm_variants(sysctls()) == ["neon"]
    @test UlmBLAS.arm_variants(sysctls("hw.optional.arm.FEAT_SME2" => 0,
                                       "hw.optional.arm.FEAT_SME_F64F64" => 0,
                                       "hw.optional.arm.sme_max_svl_b" => 0)) == ["neon"]
    #  every requirement is needed
    for i in 1:3
        @test UlmBLAS.arm_variants(sysctls(m4[1:i-1]..., m4[i+1:end]...)) == ["neon"]
    end
    @test UlmBLAS.arm_variants(sysctls(m4[1:2]..., m4[3][1] => 32)) == ["neon"]
    @test UlmBLAS.arm_variants(sysctls(m4[1:2]..., m4[3][1] => 128)) == ["neon"]
    if Sys.isapple()
        #  sysctlbyname agrees with sysctl(8)
        for (name, _) in m4
            v = UlmBLAS.sysctl_int(name)
            out = readchomp(pipeline(ignorestatus(`sysctl -n $name`); stderr=devnull))
            @test (v === nothing ? "" : string(v)) == out
        end
        @test UlmBLAS.sysctl_int("hw.optional.arm.FEAT_DOES_NOT_EXIST") === nothing
        @test UlmBLAS.sysctl_int("hw.memsize") == parse(Int, readchomp(`sysctl -n hw.memsize`))
    else
        @test UlmBLAS.sysctl_int("hw.optional.arm.FEAT_SME2") === nothing
    end
    if Sys.isapple() && Sys.ARCH === :aarch64
        @test UlmBLAS.cpu_variants() == UlmBLAS.arm_variants(UlmBLAS.sysctl_int)
        @info "UlmBLAS: CPU $(Sys.cpu_info()[1].model), " *
              join(("$n = $(UlmBLAS.sysctl_int(n))" for (n, _) in m4), ", ")
    end
end

@testset "variants in ulmBLAS_jll" begin
    #  the variants the CPU runs and ulmBLAS_jll has, in the order of
    #  cpu_variants(); on Apple silicon at least neon
    @test UlmBLAS.variants() == filter(UlmBLAS.has_library, UlmBLAS.cpu_variants())
    @test all(v -> isfile(UlmBLAS.artifact_library(v)), UlmBLAS.variants())
    @test !UlmBLAS.has_library("no_such_variant")
    if Sys.isapple() && Sys.ARCH === :aarch64
        @test "neon" in UlmBLAS.variants()
        @test UlmBLAS.has_library("sme")
        #  sme whenever the CPU runs it, otherwise never
        expected = "sme" in UlmBLAS.cpu_variants() ? "sme" : "neon"
        @test UlmBLAS.variant() == expected
        isempty(get(ENV, "ULMBLAS_LIBRARY", "")) && @test UlmBLAS.chosen[] == expected
    end
end

@testset "x86-64 variants" begin
    X = Base.BinaryPlatforms.CPUID
    feats(fs...) = f -> f in fs
    sse3, avx2 = (X.JL_X86_sse3,), (X.JL_X86_sse3, X.JL_X86_avx2, X.JL_X86_fma)
    f_dq = (avx2..., X.JL_X86_avx512f, X.JL_X86_avx512dq)
    @test UlmBLAS.x86_variants(feats(), false) == String[]
    @test UlmBLAS.x86_variants(feats(sse3...), false) == ["x86_64"]
    @test UlmBLAS.x86_variants(feats(X.JL_X86_sse3, X.JL_X86_avx2), false) == ["x86_64"]
    @test UlmBLAS.x86_variants(feats(avx2...), true) == ["haswell", "x86_64"]
    #  AVX-512 needs F, DQ and VL
    @test UlmBLAS.x86_variants(feats(f_dq...), false) == ["haswell", "x86_64"]
    @test UlmBLAS.x86_variants(feats(avx2..., X.JL_X86_avx512f, X.JL_X86_avx512vl), false) ==
          ["haswell", "x86_64"]
    @test UlmBLAS.x86_variants(feats(f_dq..., X.JL_X86_avx512vl), false) ==
          ["skylakex", "zen4", "haswell", "x86_64"]
    @test UlmBLAS.x86_variants(feats(f_dq..., X.JL_X86_avx512vl), true) ==
          ["zen4", "skylakex", "haswell", "x86_64"]
    if Sys.ARCH === :x86_64
        #  this CPU: an AVX-512 variant only with F, DQ and VL
        avx512 = any(in(UlmBLAS.cpu_variants()), ("zen4", "skylakex"))
        @test avx512 == all(X.test_cpu_feature,
                            (X.JL_X86_avx512f, X.JL_X86_avx512dq, X.JL_X86_avx512vl))
    end
end

#  UlmBLAS.mul! before forward()
include("strided.jl")
#  the GEMM for Int8, Float16, BFloat16: only with a library that has it
if all(p -> p != C_NULL, values(UlmBLAS.gemm_mixed_ptr))
    include("mixed.jl")
else
    @info "UlmBLAS: the library has no ulmblas_i8i32gemm_strided, " *
          "ulmblas_sbgemm_strided, ulmblas_shgemm_strided: tests skipped"
    @test_skip false
end

@testset "forwarding" begin
    lib = UlmBLAS.library_path()
    zdotc_before = UInt(BLAS.lbt_get_forward("zdotc_", :ilp64))
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
        #  Windows: zdotc through the adapter of libblastrampoline, which
        #  still calls OpenBLAS (see subprocess/routing.jl)
        expected = f == "zdotc" && BLAS.get_config().loaded_libs[end].complex_retstyle ===
                   :float_normal_double_argument ? zdotc_before : UInt(dlsym(h, f * "_64_"))
        @test UInt(BLAS.lbt_get_forward(f * "_", :ilp64)) == expected
    end
end

#  the LAPACK functions of ulmBLAS (libraries built with LAPACK): threads
#  as BLAS.set_num_threads, results of lu, cholesky, inv and \
@testset "LAPACK" begin
    h = dlopen(UlmBLAS.library_path())
    getter = dlsym(h, :ulmblas_get_num_threads_64_; throw_error=false)
    if getter === nothing
        #  a library without LAPACK
        @test dlsym(h, :dgetrf_64_; throw_error=false) === nothing
    else
        @test UInt(BLAS.lbt_get_forward("dgetrf_", :ilp64)) == UInt(dlsym(h, :dgetrf_64_))
        @test UInt(BLAS.lbt_get_forward("zpotrf_", :ilp64)) == UInt(dlsym(h, :zpotrf_64_))
        n0 = BLAS.get_num_threads()
        BLAS.set_num_threads(2)
        @test ccall(getter, Cint, ()) == 2
        BLAS.set_num_threads(n0)
        @test ccall(getter, Cint, ()) == n0
        for T in (Float32, Float64, ComplexF32, ComplexF64), n in (5, 120, 400)
            tol = 100 * n * eps(real(T))
            A = rand(T, n, n) + n * I
            F = lu(A)
            @test norm(F.L * F.U - A[F.p, :]) <= tol * norm(A)
            S = A' * A
            C = cholesky(Hermitian(S))
            @test norm(C.U' * C.U - S) <= tol * norm(S)
            b = rand(T, n)
            @test norm(A * (A \ b) - b) <= tol * norm(A) * norm(A \ b)
            @test norm(A * inv(A) - I) <= tol * cond(A)
        end
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
#  a CPU without SME2: the sme library is not loaded even if requested
if Sys.isapple() && Sys.ARCH === :aarch64 && !("sme" in UlmBLAS.cpu_variants())
    @testset "ULMBLAS_VARIANT=sme without SME2" begin
        @test script_passes("nosme.jl", "ULMBLAS_VARIANT" => "sme")
    end
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
