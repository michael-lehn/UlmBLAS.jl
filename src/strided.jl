#
#  GEMM with arbitrary strides: UlmBLAS.mul!(C, A, B, α, β) calls
#  ulmblas_?gemm_strided (ulmblas_strided.h of ulmBLAS) with the strides of
#  Julia's arrays, without copies.
#

using LinearAlgebra: BlasFloat, Adjoint, Transpose

#  integers: wraparound as Julia's arithmetic; UInt64 uses the function of
#  Int64 (same bits)
const IntTypes = Union{Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32,
                       UInt64}
const GemmTypes = Union{BlasFloat, IntTypes}

#  addresses of ulmblas_?gemm_strided in the library in use (C_NULL if the
#  library does not have it, e.g. the integer ones before 0.3)
const gemm_strided_ptr = Dict{DataType,Ptr{Cvoid}}()

#  addresses of the GEMM for 8-bit integers (C Int32) and 16-bit floating
#  point (C Float32) by the name of the C function
const gemm_mixed_ptr = Dict{String,Ptr{Cvoid}}()

function load_strided(path)
    #  Windows: with Julia 1.13, dlopen of a file that is no DLL did not
    #  return (GitHub Actions); a DLL starts with "MZ"
    if Sys.iswindows() && open(io -> read(io, 2), path) != b"MZ"
        error("$path is no DLL")
    end
    h = Libdl.dlopen(path)
    for (T, c) in ((Float32, "s"), (Float64, "d"),
                   (ComplexF32, "c"), (ComplexF64, "z"),
                   (Int8, "i8"), (UInt8, "i8"), (Int16, "i16"),
                   (UInt16, "i16"), (Int32, "i32"), (UInt32, "i32"),
                   (Int64, "i64"), (UInt64, "i64"))
        gemm_strided_ptr[T] = Libdl.dlsym_e(h, "ulmblas_$(c)gemm_strided")
    end
    for name in ("i8i32", "sb", "sh")
        f = "ulmblas_$(name)gemm_strided"
        gemm_mixed_ptr[f] = Libdl.dlsym_e(h, f)
    end
    return nothing
end

@noinline function missing_mixed(f)
    error("UlmBLAS: the library $(library[]) has no $f (built before " *
          "it was added, or no library loaded)")
end

#  the address of the C function f (error if the library has none)
@inline function mixed_ptr(f)
    p = get(gemm_mixed_ptr, f, C_NULL)
    p == C_NULL && missing_mixed(f)
    return p
end

@inline function check_sizes(C, A, B)
    m, n = size(C)
    k = size(A, 2)
    if size(A, 1) != m || size(B, 1) != k || size(B, 2) != n
        throw(DimensionMismatch(lazy"C has size $(size(C)), A $(size(A)), " *
                                lazy"B $(size(B))"))
    end
    return m, n, k
end

@inline function check_out(rC, cC, m, n)
    if (m > 1 && rC == 0) || (n > 1 && cC == 0)
        throw(ArgumentError("UlmBLAS.mul!: C has a zero stride"))
    end
end

#  A strided matrix, its transpose or adjoint, and transpose or adjoint of
#  these (e.g. adjoint(transpose(A)) = conj(A))
const Strided1{T} = Union{StridedMatrix{T},
                          Transpose{T,<:StridedMatrix{T}},
                          Adjoint{T,<:StridedMatrix{T}}}
const StridedOp{T} = Union{Strided1{T},
                           Transpose{T,<:Strided1{T}},
                           Adjoint{T,<:Strided1{T}}}

#  C: a strided matrix or its transpose (row major), not an adjoint
const StridedOut{T} = Union{StridedMatrix{T},
                            Transpose{T,<:StridedMatrix{T}}}

#  (array whose pointer is element (1,1), row stride, column stride, conj)
@inline layout(A::StridedMatrix) = (A, stride(A, 1), stride(A, 2), false)
@inline function layout(A::Transpose)
    P, r, c, cj = layout(parent(A))
    return (P, c, r, cj)
end
@inline function layout(A::Adjoint{T}) where {T}
    P, r, c, cj = layout(parent(A))
    return (P, c, r, cj ⊻ (T <: Complex))
end

@noinline function missing_strided(T)
    error("UlmBLAS: the library $(library[]) has no " *
          "ulmblas_?gemm_strided for $T (built before the strided " *
          "interface, or no library loaded)")
end

#  the C functions: real types take alpha, beta by value and no conj flags
for T in (Float32, Float64, Int8, Int16, Int32, Int64)
    @eval @inline function gemm_strided!(f, m, n, k, α::$T, A::Ptr{$T}, rA,
                                         cA, ::Bool, B::Ptr{$T}, rB, cB,
                                         ::Bool, β::$T, C::Ptr{$T}, rC, cC)
        ccall(f, Cvoid,
              (Int64, Int64, Int64, $T, Ptr{$T}, Int64, Int64, Ptr{$T},
               Int64, Int64, $T, Ptr{$T}, Int64, Int64),
              m, n, k, α, A, rA, cA, B, rB, cB, β, C, rC, cC)
    end
end

#  unsigned integers: the function of the signed type (same bits; the C
#  ABI may expect a sign-extended alpha, beta)
for (U, S) in ((UInt8, Int8), (UInt16, Int16), (UInt32, Int32),
               (UInt64, Int64))
    @eval @inline function gemm_strided!(f, m, n, k, α::$U, A::Ptr{$U}, rA,
                                         cA, cjA::Bool, B::Ptr{$U}, rB, cB,
                                         cjB::Bool, β::$U, C::Ptr{$U}, rC, cC)
        gemm_strided!(f, m, n, k, reinterpret($S, α), Ptr{$S}(A), rA, cA,
                      cjA, Ptr{$S}(B), rB, cB, cjB, reinterpret($S, β),
                      Ptr{$S}(C), rC, cC)
    end
end

#  complex types: alpha, beta by pointer, conjugation of A and B
for T in (ComplexF32, ComplexF64)
    @eval @inline function gemm_strided!(f, m, n, k, α::$T, A::Ptr{$T}, rA,
                                         cA, conjA::Bool, B::Ptr{$T}, rB, cB,
                                         conjB::Bool, β::$T, C::Ptr{$T}, rC,
                                         cC)
        ccall(f, Cvoid,
              (Int64, Int64, Int64, Ref{$T}, Cint, Ptr{$T}, Int64, Int64,
               Cint, Ptr{$T}, Int64, Int64, Ref{$T}, Ptr{$T}, Int64, Int64),
              m, n, k, α, conjA, A, rA, cA, conjB, B, rB, cB, β, C, rC, cC)
    end
end

"""
    UlmBLAS.mul!(C, A, B, α=true, β=false) -> C

`C = α A B + β C` for `Float32`, `Float64`, `ComplexF32`, `ComplexF64` and
the integer types `Int8` ... `Int64`, `UInt8` ... `UInt64`, with the GEMM of
ulmBLAS for arbitrary strides, without copies.

Integers wrap around as in Julia (`A*B` gives the same result).  There are
no tuned micro kernels for integers yet; ulmBLAS uses its generic one.

- `C`: a strided matrix (`Matrix`, a `view` with ranges, ...) or its
  `transpose` (row major), without zero strides (except for a dimension of
  length 1; `ArgumentError` otherwise).
- `A`, `B`: strided matrices, or `transpose`/`adjoint` of strided
  matrices (`adjoint` conjugates complex elements), also nested once
  (`adjoint(transpose(X))`).  Any strides are
  allowed: non-unit, negative, zero (e.g. a repeated row), row major.
- All three have the same element type; `α` and `β` are converted to it
  (`InexactError` if that is not possible, e.g. `α = 0.5` for `Int`).
- Other element types, mixed types and other matrix types (e.g. a `view`
  with a vector of indices, an `adjoint` `C`) are not supported and raise
  a `MethodError`; there is no fallback.

Contract (not checked, for speed):

- `C` must not overlap with `A` or `B` (otherwise the result is
  unspecified; only the elements of `C` are written).  For possibly
  overlapping operands use [`UlmBLAS.mul_aliased!`](@ref).

`A` and `B` may overlap each other or themselves.  As in BLAS, `C` is not
read for `β = 0`, and `A`, `B` are not read for `α = 0`.

This is not a method of `LinearAlgebra.mul!`: `LinearAlgebra.mul!` keeps
its own behavior (BLAS only for unit row strides).
"""
function mul!(C::StridedOut{T}, A::StridedOp{T}, B::StridedOp{T},
              α::Number=true, β::Number=false) where {T<:GemmTypes}
    m, n = size(C)
    k = size(A, 2)
    if size(A, 1) != m || size(B, 1) != k || size(B, 2) != n
        throw(DimensionMismatch(lazy"C has size $(size(C)), A $(size(A)), " *
                                lazy"B $(size(B))"))
    end
    f = get(gemm_strided_ptr, T, C_NULL)
    f == C_NULL && missing_strided(T)
    PA, rA, cA, conjA = layout(A)
    PB, rB, cB, conjB = layout(B)
    PC, rC, cC, _     = layout(C)
    if (m > 1 && rC == 0) || (n > 1 && cC == 0)
        throw(ArgumentError("UlmBLAS.mul!: C has a zero stride"))
    end
    GC.@preserve PA PB PC begin
        gemm_strided!(f, m, n, k, T(α), pointer(PA), rA, cA, conjA,
                      pointer(PB), rB, cB, conjB, T(β), pointer(PC), rC, cC)
    end
    return C
end

"""
    UlmBLAS.mul_aliased!(C, A, B, α=true, β=false) -> C

Like [`UlmBLAS.mul!`](@ref), but correct also if `C` overlaps with `A` or
`B`: then `α A B` is computed into a temporary matrix first (one
allocation of the size of `C`).  Without overlap (`Base.mightalias`) it is
`UlmBLAS.mul!`.

Also available as `UlmBLAS.mul_iansmba!` ("I Am Not Sure, Might Be
Aliased").
"""
function mul_aliased!(C::StridedOut{T}, A::StridedOp{T}, B::StridedOp{T},
                      α::Number=true, β::Number=false) where {T<:GemmTypes}
    if !(Base.mightalias(C, A) || Base.mightalias(C, B))
        return mul!(C, A, B, α, β)
    end
    AB = mul!(Matrix{T}(undef, size(C)), A, B, α, false)
    if iszero(β)
        copyto!(C, AB)
    else
        C .= AB .+ T(β) .* C
    end
    return C
end

const mul_iansmba! = mul_aliased!

const Int8or = Union{Int8,UInt8}

"""
    UlmBLAS.mul!(C::Int32 matrix, A, B, α=1, β=0) -> C

8-bit integers: `C = α A B + β C` with `A` and `B` of element type `Int8` or
`UInt8` (each its own) and `C` of element type `Int32`, exact modulo 2^32
(as Julia's `Int32` arithmetic).  The same layouts as for the floating-point
types (strided matrices, `transpose`, `adjoint`, any strides).  ulmBLAS uses
the int8 instructions of the CPU (Apple M4/M5 SME2: about twice the speed of
`Float32`; ARM dot products, x86 AVX-VNNI, AVX512-VNNI, AVX2).
"""
function mul!(C::StridedOut{Int32}, A::StridedOp{TA}, B::StridedOp{TB},
              α::Integer=1, β::Integer=0) where {TA<:Int8or,TB<:Int8or}
    m, n, k = check_sizes(C, A, B)
    f = mixed_ptr("ulmblas_i8i32gemm_strided")
    PA, rA, cA, _ = layout(A)
    PB, rB, cB, _ = layout(B)
    PC, rC, cC, _ = layout(C)
    check_out(rC, cC, m, n)
    GC.@preserve PA PB PC begin
        ccall(f, Cvoid,
              (Int64, Int64, Int64, Int32, Cint, Ptr{Cvoid}, Int64, Int64,
               Cint, Ptr{Cvoid}, Int64, Int64, Int32, Ptr{Int32}, Int64,
               Int64),
              m, n, k, α % Int32, TA === UInt8, pointer(PA), rA, cA,
              TB === UInt8, pointer(PB), rB, cB, β % Int32, pointer(PC), rC,
              cC)
    end
    return C
end

#  A and B 16-bit floating point (bits as UInt16), C Float32: f the C function
function gemm16!(f, C, A, B, α, β)
    m, n, k = check_sizes(C, A, B)
    PA, rA, cA, _ = layout(A)
    PB, rB, cB, _ = layout(B)
    PC, rC, cC, _ = layout(C)
    check_out(rC, cC, m, n)
    GC.@preserve PA PB PC begin
        ccall(mixed_ptr(f), Cvoid,
              (Int64, Int64, Int64, Float32, Ptr{Cvoid}, Int64, Int64,
               Ptr{Cvoid}, Int64, Int64, Float32, Ptr{Float32}, Int64, Int64),
              m, n, k, Float32(α), pointer(PA), rA, cA, pointer(PB), rB, cB,
              Float32(β), pointer(PC), rC, cC)
    end
    return C
end

"""
    UlmBLAS.mul!(C::Float32 matrix, A::Float16 matrix, B::Float16 matrix,
                 α=true, β=false) -> C

IEEE half precision: `C = α A B + β C` with `A`, `B` of element type
`Float16` and `C` of element type `Float32`, the products and sums in
`Float32` (`ulmblas_shgemm_strided`).  With the package BFloat16s also for
`BFloat16` (`ulmblas_sbgemm_strided`).  Layouts as for the other types.
"""
function mul!(C::StridedOut{Float32}, A::StridedOp{Float16},
              B::StridedOp{Float16}, α::Number=true, β::Number=false)
    return gemm16!("ulmblas_shgemm_strided", C, A, B, α, β)
end
