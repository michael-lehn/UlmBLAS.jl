#
#  julia --project=lapack lapack/verify.jl: which library executes the BLAS
#  calls of DGETRF in each configuration.
#
#  1. Static: for every BLAS routine Netlib LAPACK imports (nm -u), the
#     library libblastrampoline (LBT) forwards it to (lbt_get_forward and
#     dladdr), and the same for the LAPACK routines of DGETRF.
#  2. Dynamic: the LBT entries of dgemm, dtrsm, idamax, dscal and dswap are
#     replaced by counters that call the previous target, and so are those
#     of dgetrf, dgetrf2, dlaswp and ilaenv.  One DGETRF (n = 1000) then
#     shows which calls go through LBT and where they end; the flops of
#     DGEMM and DTRSM are summed.
#
#  Fails if, with Netlib LAPACK and ulmBLAS, a DGEMM or DTRSM call of DGETRF
#  does not reach ulmBLAS.
#
include("common.jl")
using Printf

const SPIED = ["dgemm_", "dtrsm_", "idamax_", "dscal_", "dswap_",
               "dgetrf_", "dgetrf2_", "dlaswp_", "ilaenv_"]
const TARGET = Dict{String,Ptr{Cvoid}}()
const CALLS = Dict(s => 0 for s in SPIED)
const FLOPS = Dict("dgemm_" => 0.0, "dtrsm_" => 0.0)

const P = Ptr{BlasInt}
const PD = Ptr{Float64}
const PC = Ptr{UInt8}

function spy_dgemm(ta::PC, tb::PC, m::P, n::P, k::P, alpha::PD, A::PD, lda::P,
                   B::PD, ldb::P, beta::PD, C::PD, ldc::P, l1::Csize_t, l2::Csize_t)
    CALLS["dgemm_"] += 1
    FLOPS["dgemm_"] += 2.0*unsafe_load(m)*unsafe_load(n)*unsafe_load(k)
    ccall(TARGET["dgemm_"], Cvoid,
          (PC, PC, P, P, P, PD, PD, P, PD, P, PD, PD, P, Csize_t, Csize_t),
          ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, l1, l2)
end
function spy_dtrsm(side::PC, uplo::PC, ta::PC, diag::PC, m::P, n::P, alpha::PD,
                   A::PD, lda::P, B::PD, ldb::P, l1::Csize_t, l2::Csize_t,
                   l3::Csize_t, l4::Csize_t)
    CALLS["dtrsm_"] += 1
    mm, nn = unsafe_load(m), unsafe_load(n)
    FLOPS["dtrsm_"] += unsafe_load(side) == UInt8('L') ? 1.0*nn*mm^2 : 1.0*mm*nn^2
    ccall(TARGET["dtrsm_"], Cvoid,
          (PC, PC, PC, PC, P, P, PD, PD, P, PD, P, Csize_t, Csize_t, Csize_t, Csize_t),
          side, uplo, ta, diag, m, n, alpha, A, lda, B, ldb, l1, l2, l3, l4)
end
function spy_idamax(n::P, x::PD, incx::P)
    CALLS["idamax_"] += 1
    ccall(TARGET["idamax_"], BlasInt, (P, PD, P), n, x, incx)
end
function spy_dscal(n::P, alpha::PD, x::PD, incx::P)
    CALLS["dscal_"] += 1
    ccall(TARGET["dscal_"], Cvoid, (P, PD, PD, P), n, alpha, x, incx)
end
function spy_dswap(n::P, x::PD, incx::P, y::PD, incy::P)
    CALLS["dswap_"] += 1
    ccall(TARGET["dswap_"], Cvoid, (P, PD, P, PD, P), n, x, incx, y, incy)
end
function spy_dgetrf(m::P, n::P, A::PD, lda::P, ipiv::P, info::P)
    CALLS["dgetrf_"] += 1
    ccall(TARGET["dgetrf_"], Cvoid, (P, P, PD, P, P, P), m, n, A, lda, ipiv, info)
end
function spy_dgetrf2(m::P, n::P, A::PD, lda::P, ipiv::P, info::P)
    CALLS["dgetrf2_"] += 1
    ccall(TARGET["dgetrf2_"], Cvoid, (P, P, PD, P, P, P), m, n, A, lda, ipiv, info)
end
function spy_dlaswp(n::P, A::PD, lda::P, k1::P, k2::P, ipiv::P, incx::P)
    CALLS["dlaswp_"] += 1
    ccall(TARGET["dlaswp_"], Cvoid, (P, PD, P, P, P, P, P), n, A, lda, k1, k2, ipiv, incx)
end
function spy_ilaenv(ispec::P, name::PC, opts::PC, n1::P, n2::P, n3::P, n4::P,
                    l1::Csize_t, l2::Csize_t)
    CALLS["ilaenv_"] += 1
    ccall(TARGET["ilaenv_"], BlasInt, (P, PC, PC, P, P, P, P, Csize_t, Csize_t),
          ispec, name, opts, n1, n2, n3, n4, l1, l2)
end

spies() = Dict(
    "dgemm_" => @cfunction(spy_dgemm, Cvoid,
        (PC, PC, P, P, P, PD, PD, P, PD, P, PD, PD, P, Csize_t, Csize_t)),
    "dtrsm_" => @cfunction(spy_dtrsm, Cvoid,
        (PC, PC, PC, PC, P, P, PD, PD, P, PD, P, Csize_t, Csize_t, Csize_t, Csize_t)),
    "idamax_" => @cfunction(spy_idamax, BlasInt, (P, PD, P)),
    "dscal_" => @cfunction(spy_dscal, Cvoid, (P, PD, PD, P)),
    "dswap_" => @cfunction(spy_dswap, Cvoid, (P, PD, P, PD, P)),
    "dgetrf_" => @cfunction(spy_dgetrf, Cvoid, (P, P, PD, P, P, P)),
    "dgetrf2_" => @cfunction(spy_dgetrf2, Cvoid, (P, P, PD, P, P, P)),
    "dlaswp_" => @cfunction(spy_dlaswp, Cvoid, (P, PD, P, P, P, P, P)),
    "ilaenv_" => @cfunction(spy_ilaenv, BlasInt,
        (P, PC, PC, P, P, P, P, Csize_t, Csize_t)))

set_forward(name, addr) = BLAS.lbt_set_forward(name, addr, :ilp64)

#  the BLAS routines Netlib LAPACK imports (ILP64 names without "64_")
function imported_blas()
    out = read(`nm -u $LAPACK_LIBRARY`, String)
    names = String[]
    for l in eachline(IOBuffer(out))
        m = match(r"_?([a-z0-9]+_)64_$", strip(l))
        m === nothing || push!(names, m[1])
    end
    return sort(unique(names))
end

short(path) = basename(path)

function static_report(c)
    println("\n[$c] forward targets of LBT")
    by = Dict{String,Vector{String}}()
    for s in imported_blas()
        push!(get!(by, short(forward_target(s)), String[]), s)
    end
    println("  BLAS routines imported by Netlib LAPACK ($(sum(length, values(by)))):")
    for (lib, syms) in sort(collect(by); by=first)
        println("    ", rpad(lib, 28), length(syms), length(syms) <= 12 ? "  " * join(syms, " ") : "")
    end
    for s in ("dgetrf_", "dgetrf2_", "dlaswp_", "ilaenv_", "dgemm_", "dtrsm_")
        println("  ", rpad(s, 10), " -> ", forward_target(s))
    end
    return by
end

function dynamic_report(c; n=1000)
    for s in SPIED
        TARGET[s] = BLAS.lbt_get_forward(s, :ilp64)
        CALLS[s] = 0
    end
    FLOPS["dgemm_"] = FLOPS["dtrsm_"] = 0.0
    sp = spies()
    targets = Dict(s => short(library_of(TARGET[s])) for s in SPIED)
    A = randn(n, n); B = copy(A); ipiv = zeros(BlasInt, n)
    try
        foreach(s -> set_forward(s, sp[s]), SPIED)
        info = dgetrf!(B, ipiv)
        info == 0 || error("INFO = $info")
    finally
        foreach(s -> set_forward(s, TARGET[s]), SPIED)
    end
    total = 2n^3/3 - n^2/2 + 5n/6
    println("  one DGETRF, n = $n: calls through LBT (and where they end)")
    for s in SPIED
        @printf("    %-9s %6d   %s\n", s, CALLS[s], CALLS[s] > 0 ? targets[s] : "")
    end
    @printf("  flops in DGEMM %.1f%%, in DTRSM %.1f%% of DGETRF's %.3g\n",
            100FLOPS["dgemm_"]/total, 100FLOPS["dtrsm_"]/total, total)
    return copy(CALLS), targets
end

describe()
results = Dict()
for c in CONFIGS
    configure(c)
    println("\n", "="^72, "\n", c, ": ", BLAS.get_config())
    static_report(c)
    results[c] = dynamic_report(c)
end
configure(:openblas)

calls, targets = results[:netlib_ulmblas]
ulm = basename(UlmBLAS.library_path())
ok = calls["dgemm_"] > 0 && calls["dtrsm_"] > 0 &&
     targets["dgemm_"] == ulm && targets["dtrsm_"] == ulm
println("\nNetlib LAPACK + ulmBLAS: DGEMM and DTRSM of DGETRF run in $ulm: ", ok)
ok || exit(1)
