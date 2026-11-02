#
#  The configurations of the experiment, set in libblastrampoline (LBT):
#
#      :openblas         Julia's default: LAPACK and BLAS of OpenBLAS
#      :netlib_ulmblas   Netlib LAPACK, BLAS of ulmBLAS (UlmBLAS.forward())
#      :netlib_openblas  Netlib LAPACK, BLAS of OpenBLAS
#
#  Netlib LAPACK is the library of build.sh, LAPACK_LIBRARY (default
#  lapack/liblapack/lib/liblapack.{so,dylib}).  Its BLAS calls go through
#  LBT, so they run in the library LBT forwards them to.
#
using LinearAlgebra, Libdl, Random, UlmBLAS, OpenBLAS_jll
using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS: libblas

const LAPACK_LIBRARY = get(ENV, "LAPACK_LIBRARY",
    joinpath(@__DIR__, "liblapack", "lib", "liblapack." * Libdl.dlext))
const CONFIGS = (:openblas, :netlib_ulmblas, :netlib_openblas)

function configure(c::Symbol)
    isfile(LAPACK_LIBRARY) || error("no Netlib LAPACK: $LAPACK_LIBRARY (lapack/build.sh)")
    BLAS.lbt_forward(OpenBLAS_jll.libopenblas_path; clear=true, suffix_hint="64_")
    BLAS.set_num_threads(1)
    c === :openblas && return
    c === :netlib_ulmblas && UlmBLAS.forward()
    n = BLAS.lbt_forward(LAPACK_LIBRARY; clear=false, suffix_hint="64_")
    n > 0 || error("no routine forwarded from $LAPACK_LIBRARY")
    c in CONFIGS || error("unknown configuration $c")
    return
end

#  DGETRF through LBT: A is overwritten by L and U, ipiv by the pivots;
#  returns INFO
function dgetrf!(A::StridedMatrix{Float64}, ipiv::Vector{BlasInt})
    m, n = size(A)
    stride(A, 1) == 1 || error("A needs unit stride")
    length(ipiv) >= min(m, n) || error("ipiv too short")
    info = Ref{BlasInt}(0)
    ccall((:dgetrf_64_, libblas), Cvoid,
          (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
           Ptr{BlasInt}, Ref{BlasInt}),
          m, n, A, max(1, stride(A, 2)), ipiv, info)
    return Int(info[])
end

#  file of the library containing address p (dladdr)
struct DlInfo
    fname::Cstring
    fbase::Ptr{Cvoid}
    sname::Cstring
    saddr::Ptr{Cvoid}
end
function library_of(p::Ptr{Cvoid})
    p == C_NULL && return "(none)"
    info = Ref{DlInfo}()
    ccall(:dladdr, Cint, (Ptr{Cvoid}, Ref{DlInfo}), p, info) == 0 && return "(unknown)"
    return unsafe_string(info[].fname)
end

#  the library LBT forwards the ILP64 routine name (e.g. "dgemm_") to
forward_target(name) = library_of(BLAS.lbt_get_forward(name, :ilp64))

function describe()
    println("Julia:        ", VERSION, ", ", Sys.cpu_info()[1].model,
            " (", Sys.CPU_THREADS, " threads)")
    println("UlmBLAS:      ", UlmBLAS.library_path(), " (variant ", UlmBLAS.chosen[], ")")
    println("OpenBLAS:     ", OpenBLAS_jll.libopenblas_path)
    println("Netlib:       ", realpath(LAPACK_LIBRARY))
    build = joinpath(dirname(dirname(LAPACK_LIBRARY)), "build.txt")
    isfile(build) && println("              ", readline(build))
end

#  the n×n test matrices of bench.jl and coldom.jl, reproducible (seed n):
#
#      :dense    randn
#      :coldom   the same off-diagonal entries, a[i,i] = ±1.25 c_i (c_i: sum
#                of |a[j,i]|, j ≠ i; the sign of the random a[i,i]): strictly
#                diagonally dominant by columns.  This makes partial
#                pivoting choose the diagonal (column dominance is preserved
#                by the elimination), so ipiv[i] == i.
#      :rowdom   a[i,i] = ±1.25 r_i (r_i: sum of |a[i,j]|, j ≠ i): dominant
#                by rows, which does not prevent row interchanges (e.g.
#                [2 1; 3 4])
function test_matrix(kind::Symbol, n)
    A = randn(Xoshiro(n), n, n)
    kind === :dense && return A
    r = [sum(abs, A[i, :]) - abs(A[i, i]) for i in 1:n]
    c = [sum(abs, A[:, i]) - abs(A[i, i]) for i in 1:n]
    d = kind === :coldom ? c : kind === :rowdom ? r : error("unknown matrix $kind")
    for i in 1:n
        A[i, i] = (A[i, i] < 0 ? -1.25 : 1.25)*max(d[i], 1.0)
    end
    return A
end
