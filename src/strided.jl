#
#  GEMM with arbitrary strides: ulmBLAS.mul!(C, A, B, α, β) calls
#  ulmblas_?gemm_strided (ulmblas_strided.h of ulmBLAS) with the strides of
#  Julia's arrays, without copies.
#

using LinearAlgebra: BlasFloat, Adjoint, Transpose

#  addresses of ulmblas_{s,d,c,z}gemm_strided in the library in use
const gemm_strided_ptr = Dict{DataType,Ptr{Cvoid}}()

function load_strided(path)
    h = Libdl.dlopen(path)
    for (T, c) in ((Float32, 's'), (Float64, 'd'),
                   (ComplexF32, 'c'), (ComplexF64, 'z'))
        gemm_strided_ptr[T] = Libdl.dlsym_e(h, "ulmblas_$(c)gemm_strided")
    end
    return nothing
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
    error("ulmBLAS: the library $(library[]) has no " *
          "ulmblas_?gemm_strided for $T (built before the strided " *
          "interface, or no library loaded)")
end

#  the C functions: real types take alpha, beta by value and no conj flags
for T in (Float32, Float64)
    @eval @inline function gemm_strided!(f, m, n, k, α::$T, A::Ptr{$T}, rA,
                                         cA, ::Bool, B::Ptr{$T}, rB, cB,
                                         ::Bool, β::$T, C::Ptr{$T}, rC, cC)
        ccall(f, Cvoid,
              (Int64, Int64, Int64, $T, Ptr{$T}, Int64, Int64, Ptr{$T},
               Int64, Int64, $T, Ptr{$T}, Int64, Int64),
              m, n, k, α, A, rA, cA, B, rB, cB, β, C, rC, cC)
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
    ulmBLAS.mul!(C, A, B, α=true, β=false) -> C

`C = α A B + β C` for `Float32`, `Float64`, `ComplexF32`, `ComplexF64`, with
the GEMM of ulmBLAS for arbitrary strides, without copies.

- `C`: a strided matrix (`Matrix`, a `view` with ranges, ...) or its
  `transpose` (row major).
- `A`, `B`: strided matrices, or `transpose`/`adjoint` of strided
  matrices (`adjoint` conjugates complex elements), also nested once
  (`adjoint(transpose(X))`).  Any strides are
  allowed: non-unit, negative, zero (e.g. a repeated row), row major.
- All three have the same element type; `α` and `β` are converted to it.

Contract (not checked, for speed):

- `C` must not overlap with `A` or `B`.  For possibly overlapping operands
  use [`ulmBLAS.mul_aliased!`](@ref).
- The elements of `C` must be distinct memory locations (no zero stride
  of `C`, except for a dimension of length 1).

`A` and `B` may overlap each other or themselves.  As in BLAS, `C` is not
read for `β = 0`, and `A`, `B` are not read for `α = 0`.

This is not a method of `LinearAlgebra.mul!`: `LinearAlgebra.mul!` keeps
its own behavior (BLAS only for unit row strides).
"""
function mul!(C::StridedOut{T}, A::StridedOp{T}, B::StridedOp{T},
              α::Number=true, β::Number=false) where {T<:BlasFloat}
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
    GC.@preserve PA PB PC begin
        gemm_strided!(f, m, n, k, T(α), pointer(PA), rA, cA, conjA,
                      pointer(PB), rB, cB, conjB, T(β), pointer(PC), rC, cC)
    end
    return C
end

"""
    ulmBLAS.mul_aliased!(C, A, B, α=true, β=false) -> C

Like [`ulmBLAS.mul!`](@ref), but correct also if `C` overlaps with `A` or
`B`: then `α A B` is computed into a temporary matrix first (one
allocation of the size of `C`).  Without overlap (`Base.mightalias`) it is
`ulmBLAS.mul!`.

Also available as `ulmBLAS.mul_iansmba!` ("I Am Not Sure, Might Be
Aliased").
"""
function mul_aliased!(C::StridedOut{T}, A::StridedOp{T}, B::StridedOp{T},
                      α::Number=true, β::Number=false) where {T<:BlasFloat}
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
