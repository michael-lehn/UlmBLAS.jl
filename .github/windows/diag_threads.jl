#  test/subprocess/threads.jl with progress output (Windows diagnostics)
using LinearAlgebra, Random, UlmBLAS
step(s) = (println(round(time() - T0; digits=1), "s ", s); flush(stdout))
const T0 = time()
step("threads: $(Threads.nthreads()), library $(UlmBLAS.library_path())")
const TYPES = (Float32, Float64, ComplexF32, ComplexF64)
const SIZES = (7, 33, 64, 150, 260)
done = Threads.Atomic{Int}(0)
rng = Xoshiro(1)
jobs = []
for T in TYPES, n in SIZES, _ in 1:3
    P = rand(rng, T, 2n, n); A = view(P, 2:2:2n, :); B = rand(rng, T, n, n); C = zeros(T, n, n)
    push!(jobs, (T, n, () -> UlmBLAS.mul!(C, A, transpose(B))))
end
step("phase 1: $(length(jobs)) tasks of UlmBLAS.mul!")
@sync for (T, n, f) in jobs
    Threads.@spawn begin
        for _ in 1:4
            f()
        end
        k = Threads.atomic_add!(done, 1) + 1
        step("  done $k ($T, n = $n) on thread $(Threads.threadid())")
    end
end
step("phase 1 finished")
UlmBLAS.forward()
step("forward() done")
done[] = 0
jobs = []
for T in TYPES, n in SIZES
    A = rand(rng, T, n, n); B = rand(rng, T, n, n); x = rand(rng, T, n)
    push!(jobs, ("gemm $T $n", () -> A * B))
    push!(jobs, ("gemv $T $n", () -> A * x))
    push!(jobs, ("trsm $T $n", () -> UpperTriangular(A + n * I) \ B))
    push!(jobs, ("lu $T $n", () -> lu(A + n * I)))
end
step("phase 2: $(length(jobs)) tasks")
@sync for (name, f) in shuffle(rng, jobs)
    Threads.@spawn begin
        for _ in 1:3
            f()
        end
        k = Threads.atomic_add!(done, 1) + 1
        step("  done $k $name on thread $(Threads.threadid())")
    end
end
step("phase 2 finished")
