#  UlmBLAS.mul! and UlmBLAS.mul_aliased!: arbitrary strides, transpose and
#  adjoint, alpha and beta, against a reference without BLAS

#  reference: α A B + β C in plain loops (A, B with op already applied)
function mul_ref(A, B, C, α, β)
    T = eltype(C)
    R = Matrix{T}(undef, size(C))
    for j in axes(B, 2), i in axes(A, 1)
        s = zero(T)
        for l in axes(A, 2)
            s += A[i, l] * B[l, j]
        end
        R[i, j] = iszero(β) ? α * s : α * s + β * C[i, j]
    end
    return R
end

#  an r x c operand with the given layout, as a view into a larger parent
#  (the parent is returned too, to check the elements outside the view)
function operand(::Type{T}, layout, r, c) where {T}
    if layout === :col
        P = rand(T, r + 3, c + 2);   V = view(P, 2:r+1, 2:c+1)
    elseif layout === :row                   # row major: transpose of a view
        P = rand(T, c + 2, r + 3);   V = transpose(view(P, 2:c+1, 2:r+1))
    elseif layout === :stride2
        P = rand(T, 2r + 3, c + 1);  V = view(P, 2:2:2r, 1:c)
    elseif layout === :negrow
        P = rand(T, r + 2, c + 1);   V = view(P, r+1:-1:2, 1:c)
    elseif layout === :negboth
        P = rand(T, 3r + 3, 2c + 2); V = view(P, 3r:-3:3, 2c:-2:2)
    elseif layout === :zerorow               # every row the same
        P = rand(T, 3, c + 1);       V = view(P, StepRangeLen(2, 0, r), 1:c)
    else # :zerocol                          # every column the same
        P = rand(T, r + 1, 3);       V = view(P, 1:r, StepRangeLen(2, 0, c))
    end
    return P, V
end

wrapped(op, M) = op === :N ? M : op === :T ? transpose(M) : adjoint(M)

#  op(stored) with size r x c (the stored matrix is c x r for T and C)
function operand_op(::Type{T}, layout, op, r, c) where {T}
    P, V = op === :N ? operand(T, layout, r, c) : operand(T, layout, c, r)
    return P, wrapped(op, V)
end

tol(T, k) = (T <: Complex ? 8 : 4) * (k + 2) * eps(real(T))

#  one case; returns true if the result is correct and no element of the
#  parent of C outside C changed
function check_case(f!, T, m, n, k, lA, opA, lB, opB, lC, α, β)
    PA, A = operand_op(T, lA, opA, m, k)
    PB, B = operand_op(T, lB, opB, k, n)
    PC, C = operand(T, lC, m, n)
    if iszero(α)                     # A and B must not be read
        fill!(PA, T(NaN)); fill!(PB, T(NaN))
    end
    if iszero(β)                     # C must not be read
        C .= T(NaN)
    end
    C0 = copy(C); PC0 = copy(PC)
    Af = iszero(α) ? zeros(T, m, k) : Matrix(A)
    Bf = iszero(α) ? zeros(T, k, n) : Matrix(B)
    ref = mul_ref(Af, Bf, C0, T(α), T(β))
    bound = abs(α) .* mul_ref(abs.(Af), abs.(Bf), zeros(real(T), m, n), 1, 0)
    iszero(β) || (bound .+= abs(β) .* abs.(C0))

    f!(C, A, B, α, β)

    R = copy(C)
    ok = all(isfinite, R) &&
         all(abs.(R .- ref) .<= tol(T, max(k, 1)) .* bound .+ floatmin(real(T)))
    C .= C0                          # restore C: then the parent is unchanged
    return ok && isequal(PC, PC0)
end

const LAYOUTS = (:col, :row, :stride2, :negrow, :negboth, :zerorow, :zerocol)
const OUT_LAYOUTS = (:col, :row, :stride2, :negrow, :negboth)

@testset "strided $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    ops = T <: Complex ? (:N, :T, :C) : (:N, :T)
    αβ = ((1, 0), (T(-1.5) + (T <: Complex ? T(0.5im) : 0), T(0.25)),
          (0, T(2)), (2, 1))
    #  every layout and op of A and B with column-major C, small and medium
    for (m, n, k) in ((1, 1, 1), (3, 2, 5), (17, 13, 29), (64, 33, 70))
        for lA in LAYOUTS, lB in LAYOUTS, opA in ops, opB in ops
            i = hash((lA, lB, opA, opB)) % length(αβ) + 1
            @test check_case(UlmBLAS.mul!, T, m, n, k, lA, opA, lB, opB,
                             :col, αβ[i]...)
        end
    end
    #  every layout of C, every α, β
    for lC in OUT_LAYOUTS, (α, β) in αβ
        @test check_case(UlmBLAS.mul!, T, 23, 19, 31, :stride2, ops[end],
                         :negrow, :N, lC, α, β)
    end
    #  large (several blocks of the GEMM algorithm)
    for (lA, lB, lC) in ((:col, :col, :col), (:stride2, :negboth, :row),
                         (:zerorow, :row, :negrow))
        @test check_case(UlmBLAS.mul!, T, 301, 263, 517, lA, ops[end], lB,
                         :N, lC, αβ[2]...)
    end
    #  empty and k = 0
    for (m, n, k) in ((0, 5, 3), (4, 0, 3), (4, 5, 0))
        @test check_case(UlmBLAS.mul!, T, m, n, k, :negrow, :N, :stride2,
                         :N, :stride2, 2, 1)
    end
end

@testset "strided: no allocations" begin
    A = view(rand(200, 200), 1:2:200, 1:100); B = rand(100, 100)
    C = zeros(100, 100)
    UlmBLAS.mul!(C, A, B', 2, 0.5)
    @test (@allocated UlmBLAS.mul!(C, A, B', 2, 0.5)) == 0
    Z = rand(ComplexF64, 50, 50); W = rand(ComplexF64, 50, 50)
    D = zeros(ComplexF64, 50, 50)
    UlmBLAS.mul!(D, Z', W, 1im, 0)
    @test (@allocated UlmBLAS.mul!(D, Z', W, 1im, 0)) == 0
end

@testset "strided: errors" begin
    @test_throws DimensionMismatch UlmBLAS.mul!(zeros(3, 4), rand(3, 2),
                                                rand(3, 4))
    #  no fallback for other types
    @test_throws MethodError UlmBLAS.mul!(zeros(Int, 2, 2), ones(Int, 2, 2),
                                          ones(Int, 2, 2))
    @test_throws MethodError UlmBLAS.mul!(zeros(2, 2), rand(Float32, 2, 2),
                                          rand(2, 2))
    #  other matrix types: a view with a vector of indices, an adjoint C
    A = rand(4, 4)
    @test_throws MethodError UlmBLAS.mul!(zeros(2, 4), view(A, [1, 3], :), A)
    @test_throws MethodError UlmBLAS.mul!(adjoint(zeros(ComplexF64, 4, 4)),
                                          rand(ComplexF64, 4, 4),
                                          rand(ComplexF64, 4, 4))
    #  C with a zero stride
    @test_throws ArgumentError UlmBLAS.mul!(view(zeros(4, 4), StepRangeLen(1, 0, 4), 1:4), A, A)
    @test_throws ArgumentError UlmBLAS.mul!(view(zeros(4, 4), 1:4, StepRangeLen(1, 0, 4)), A, A)
    #  ... but not for a dimension of length 1
    C = zeros(4, 4)
    @test UlmBLAS.mul!(view(C, StepRangeLen(2, 0, 1), 1:4), A[1:1, :], A) ≈ A[1:1, :] * A
end

@testset "mul_aliased! $T" for T in (Float64, ComplexF64)
    for (α, β) in ((1, 0), (T(2), T(-0.5)))
        #  C is A
        A = rand(T, 40, 40); B = rand(T, 40, 40); A0 = copy(A)
        UlmBLAS.mul_aliased!(A, A, B, α, β)
        @test A ≈ mul_ref(A0, B, A0, T(α), T(β))
        #  C overlaps B (shifted view of the same parent)
        P = rand(T, 60, 40); P0 = copy(P)
        C = view(P, 1:40, 1:40); Bv = view(P, 11:50, 1:40); Aa = rand(T, 40, 40)
        UlmBLAS.mul_iansmba!(C, Aa, Bv, α, β)
        @test C ≈ mul_ref(Aa, P0[11:50, :], P0[1:40, :], T(α), T(β))
        #  no overlap: same as mul!
        C1 = rand(T, 40, 40); C2 = copy(C1)
        UlmBLAS.mul_aliased!(C1, A0, B, α, β)
        UlmBLAS.mul!(C2, A0, B, α, β)
        @test C1 == C2
    end
end
