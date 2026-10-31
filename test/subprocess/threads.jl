#  Run in a fresh process with several threads by runtests.jl: concurrent
#  calls of UlmBLAS.mul! (right after loading, before any other call:
#  initialization of the memory pools of ulmBLAS), then after
#  UlmBLAS.forward() concurrent BLAS routines of ulmBLAS, routines that
#  stay with OpenBLAS and UlmBLAS.mul!, each task with its own output.
#  This tests that the routines can be called concurrently (ulmBLAS itself
#  computes each call in one thread).
using LinearAlgebra, Test, Random
using UlmBLAS

const TYPES = (Float32, Float64, ComplexF32, ComplexF64)
const SIZES = (7, 33, 64, 150, 260)

#  a job: inputs, a function computing into its own output, the expected
#  result (computed in advance, by generic Julia code or OpenBLAS)
struct Job
    run::Function
    expected::Any
    tol::Float64
end
rtol(T) = 200 * eps(real(T))

function strided_job(rng, T, n)
    P = rand(rng, T, 2n, n); A = view(P, 2:2:2n, :); B = rand(rng, T, n, n)
    C = zeros(T, n, n)
    return Job(() -> (UlmBLAS.mul!(C, A, transpose(B)); copy(C)),
               Matrix(A) * transpose(B), rtol(T) * n)
end

@testset "threads" begin
    @test Threads.nthreads() > 1
    rng = Xoshiro(1)
    #  phase 1: UlmBLAS.mul! only, the first calls are concurrent
    jobs = [strided_job(rng, T, n) for T in TYPES for n in SIZES for _ in 1:3]
    ids = Vector{Int}(undef, length(jobs))
    results = Vector{Any}(undef, length(jobs))
    @sync for (i, j) in enumerate(jobs)
        Threads.@spawn begin
            ids[i] = Threads.threadid()
            r = nothing
            for _ in 1:4
                r = j.run()
            end
            results[i] = r
        end
    end
    @test length(unique(ids)) > 1
    @test all(isapprox(results[i], jobs[i].expected; rtol=jobs[i].tol) for i in eachindex(jobs))

    #  phase 2: after forward(), different routines at the same time
    UlmBLAS.forward()
    jobs = Job[]
    for T in TYPES, n in SIZES
        A = rand(rng, T, n, n); B = rand(rng, T, n, n); x = rand(rng, T, n)
        U = UpperTriangular(A + n * I)
        push!(jobs, strided_job(rng, T, n))
        Ag, Bg = copy(A), copy(B)
        ref = [sum(Ag[i, l] * Bg[l, j] for l in 1:n) for i in 1:n, j in 1:n]
        push!(jobs, Job(() -> Ag * Bg, ref, rtol(T) * n))                 # gemm
        push!(jobs, Job(() -> A * x, [sum(A[i, l] * x[l] for l in 1:n) for i in 1:n],
                        rtol(T) * n))                                   # gemv
        Bt = rand(rng, T, n, 5)
        push!(jobs, Job(() -> U * (U \ Bt), Bt, rtol(T) * n^2))          # trsm, trmm
        y = rand(rng, T, n)
        push!(jobs, Job(() -> dot(x, y), sum(conj.(x) .* y), rtol(T) * n))   # OpenBLAS
        if isdefined(BLAS, :gemmt!)
            Cg = zeros(T, n, n)
            push!(jobs, Job(() -> triu(BLAS.gemmt!('U', 'N', 'N', one(T), A, B,
                                                   zero(T), copy(Cg))),
                            triu(ref), rtol(T) * n))                      # OpenBLAS
        end
        push!(jobs, Job(() -> (F = lu(A + n * I); F.L * F.U), (A + n * I)[lu(A + n * I).p, :],
                        rtol(T) * n^2))                                  # LAPACK
    end
    results = Vector{Any}(undef, length(jobs))
    order = shuffle(rng, collect(eachindex(jobs)))
    @sync for i in order
        Threads.@spawn begin
            r = nothing
            for _ in 1:3
                r = jobs[i].run()
            end
            results[i] = r
        end
    end
    bad = [i for i in eachindex(jobs) if !isapprox(results[i], jobs[i].expected; rtol=jobs[i].tol)]
    @test isempty(bad)
end
