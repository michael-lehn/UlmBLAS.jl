#
#  julia --project=lapack lapack/profile.jl OUT.tsv [n ...]: where the time
#  of Netlib's DGETRF goes, with ulmBLAS and with OpenBLAS's BLAS.
#
#  The LBT entries of the BLAS routines DGETRF calls (dgemm, dtrsm, idamax,
#  dscal; verify.jl shows that there are no others) are replaced by
#  wrappers that time each call (time_ns) and call the previous target.
#  For each n, matrix kind (MATRIX, default dense,coldom) and the two Netlib
#  configurations, DGETRF is called repeatedly (BUDGET seconds, default
#  0.5, at least 5 and at most 2000 calls, after 2 for warm-up); per call
#  the time in each routine is recorded, and the rest of the call ("lapack":
#  DGETRF, DGETRF2, DLASWP, ILAENV inside liblapack, plus the cost of the
#  wrappers).  Reported: medians over the calls.  "total" is the time of a
#  DGETRF with the wrappers, "plain" without them (the same as bench.jl):
#  plain - total is the cost of the wrappers (two time_ns per BLAS call).
#
#  Columns: n, matrix, config, routine, calls (per DGETRF), median_ns.
#
include("common.jl")
using Statistics, Printf

const SIZES = length(ARGS) >= 2 ? parse.(Int, ARGS[2:end]) :
    [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
     1536, 2048, 3072, 4096]
const BUDGET = parse(Float64, get(ENV, "BUDGET", "0.5"))
const MATRICES = Symbol.(split(get(ENV, "MATRIX", "dense,coldom"), ','))
const ROUTINES = ["dgemm_", "dtrsm_", "idamax_", "dscal_"]
const TARGET = Vector{Ptr{Cvoid}}(undef, 4)
const NS = zeros(UInt64, 4)
const CALLS = zeros(Int, 4)

const P = Ptr{BlasInt}
const PD = Ptr{Float64}
const PC = Ptr{UInt8}

function t_dgemm(ta::PC, tb::PC, m::P, n::P, k::P, alpha::PD, A::PD, lda::P,
                 B::PD, ldb::P, beta::PD, C::PD, ldc::P, l1::Csize_t, l2::Csize_t)
    t0 = time_ns()
    ccall(TARGET[1], Cvoid,
          (PC, PC, P, P, P, PD, PD, P, PD, P, PD, PD, P, Csize_t, Csize_t),
          ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, l1, l2)
    NS[1] += time_ns() - t0; CALLS[1] += 1
    return
end
function t_dtrsm(side::PC, uplo::PC, ta::PC, diag::PC, m::P, n::P, alpha::PD,
                 A::PD, lda::P, B::PD, ldb::P, l1::Csize_t, l2::Csize_t,
                 l3::Csize_t, l4::Csize_t)
    t0 = time_ns()
    ccall(TARGET[2], Cvoid,
          (PC, PC, PC, PC, P, P, PD, PD, P, PD, P, Csize_t, Csize_t, Csize_t, Csize_t),
          side, uplo, ta, diag, m, n, alpha, A, lda, B, ldb, l1, l2, l3, l4)
    NS[2] += time_ns() - t0; CALLS[2] += 1
    return
end
function t_idamax(n::P, x::PD, incx::P)
    t0 = time_ns()
    r = ccall(TARGET[3], BlasInt, (P, PD, P), n, x, incx)
    NS[3] += time_ns() - t0; CALLS[3] += 1
    return r
end
function t_dscal(n::P, alpha::PD, x::PD, incx::P)
    t0 = time_ns()
    ccall(TARGET[4], Cvoid, (P, PD, PD, P), n, alpha, x, incx)
    NS[4] += time_ns() - t0; CALLS[4] += 1
    return
end

wrappers() = [
    @cfunction(t_dgemm, Cvoid,
        (PC, PC, P, P, P, PD, PD, P, PD, P, PD, PD, P, Csize_t, Csize_t)),
    @cfunction(t_dtrsm, Cvoid,
        (PC, PC, PC, PC, P, P, PD, PD, P, PD, P, Csize_t, Csize_t, Csize_t, Csize_t)),
    @cfunction(t_idamax, BlasInt, (P, PD, P)),
    @cfunction(t_dscal, Cvoid, (P, PD, PD, P))]

#  per DGETRF: time of the call and of each routine, calls of each routine
function run(A, B, ipiv; spied)
    nsamples = 0; start = time()
    total = Float64[]; parts = [Float64[] for _ in ROUTINES]; calls = zeros(Int, 4)
    k = 0
    while k < 7 || (time() - start < BUDGET && k < 2002)
        k += 1
        copyto!(B, A); fill!(NS, 0); fill!(CALLS, 0)
        t0 = time_ns()
        dgetrf!(B, ipiv)
        t = time_ns() - t0
        k <= 2 && continue        # warm-up
        push!(total, t)
        for i in 1:4
            push!(parts[i], NS[i])
        end
        calls .= CALLS
    end
    return total, parts, calls
end

describe()
configure(:openblas)
println("BLAS threads: ", BLAS.get_num_threads(), ", Julia threads: ", Threads.nthreads())
w = wrappers()
open(ARGS[1], "w") do io
    println(io, join(["n", "matrix", "config", "routine", "calls", "median_ns"], '\t'))
    for (k, n) in enumerate(SIZES), kind in MATRICES
        A = test_matrix(kind, n); B = similar(A); ipiv = zeros(BlasInt, n)
        order = isodd(k) ? (:netlib_ulmblas, :netlib_openblas) : (:netlib_openblas, :netlib_ulmblas)
        for c in order
            configure(c)
            plain, _, _ = run(A, B, ipiv; spied=false)
            for i in 1:4
                TARGET[i] = BLAS.lbt_get_forward(ROUTINES[i], :ilp64)
            end
            total, parts, calls = try
                foreach(i -> BLAS.lbt_set_forward(ROUTINES[i], w[i], :ilp64), 1:4)
                run(A, B, ipiv; spied=true)
            finally
                foreach(i -> BLAS.lbt_set_forward(ROUTINES[i], TARGET[i], :ilp64), 1:4)
            end
            rest = total .- sum(parts)
            row(r, ncalls, v) = println(io, join([n, kind, c, r, ncalls, median(v)], '\t'))
            row("plain", 1, plain); row("total", 1, total)
            for i in 1:4
                row(chop(ROUTINES[i]), calls[i], parts[i])
            end
            row("lapack", 1, rest)
            flush(io)
            @printf("n = %5d %-7s %-16s plain %11.0f  total %11.0f  dgemm %11.0f  dtrsm %10.0f  idamax %9.0f  dscal %9.0f  lapack %10.0f ns\n",
                    n, kind, c, median(plain), median(total), median.(parts)..., median(rest))
        end
    end
end
configure(:openblas)
