#  UlmBLAS.mul! for 8-bit integers (C Int32) and 16-bit floating point (C
#  Float32): the layouts of strided.jl, against references in plain loops

function check_int8(TA, TB, m, n, k, lA, opA, lB, opB, lC, α, β)
    _, A = operand_op(TA, lA, opA, m, k)
    _, B = operand_op(TB, lB, opB, k, n)
    PC, C = operand(Int32, lC, m, n)
    C0 = copy(C); PC0 = copy(PC)
    ref = Matrix{Int32}(undef, m, n)
    for j in 1:n, i in 1:m
        s = Int32(0)
        for l in 1:k
            s += Int32(A[i, l]) * Int32(B[l, j])
        end
        ref[i, j] = Int32(α) * s + (iszero(β) ? Int32(0) : Int32(β) * C0[i, j])
    end
    UlmBLAS.mul!(C, A, B, α, β)
    ok = C == ref
    C .= C0
    return ok && PC == PC0
end

function check_16(T, m, n, k, lA, opA, lB, opB, lC, α, β)
    _, A = operand_op(T, lA, opA, m, k)
    _, B = operand_op(T, lB, opB, k, n)
    PC, C = operand(Float32, lC, m, n)
    iszero(β) && (C .= NaN32)
    C0 = copy(C); PC0 = copy(PC)
    Af = Float64.(Float32.(Matrix(A))); Bf = Float64.(Float32.(Matrix(B)))
    ref = α .* (Af * Bf) .+ (iszero(β) ? 0 : β .* Float64.(C0))
    bound = abs(α) .* (abs.(Af) * abs.(Bf)) .+
            (iszero(β) ? 0 : abs(β) .* abs.(Float64.(C0)))
    UlmBLAS.mul!(C, A, B, α, β)
    ok = all(isfinite, C) &&
         all(abs.(C .- ref) .<= 4 * (k + 2) * eps(Float32) .* bound .+
                                floatmin(Float32))
    C .= C0
    return ok && isequal(PC, PC0)
end

@testset "Int8/UInt8 -> Int32" begin
    αβ = ((1, 0), (-3, 5), (2, 1), (0, 7))
    for TA in (Int8, UInt8), TB in (Int8, UInt8)
        for (m, n, k) in ((1, 1, 1), (3, 2, 5), (17, 13, 29), (70, 33, 131))
            for lA in LAYOUTS, lB in LAYOUTS, opA in (:N, :T), opB in (:N, :T)
                i = hash((lA, lB, opA, opB)) % length(αβ) + 1
                @test check_int8(TA, TB, m, n, k, lA, opA, lB, opB, :col,
                                 αβ[i]...)
            end
        end
        for lC in OUT_LAYOUTS, (α, β) in αβ
            @test check_int8(TA, TB, 23, 19, 31, :stride2, :T, :negrow, :N,
                             lC, α, β)
        end
        @test check_int8(TA, TB, 301, 263, 517, :col, :N, :row, :N, :negrow,
                         -3, 5)
    end
end

@testset "$T -> Float32" for T in (Float16, BFloat16)
    αβ = ((1, 0), (-1.5f0, 0.25f0), (2, 1))
    for (m, n, k) in ((1, 1, 1), (3, 2, 5), (17, 13, 29), (64, 33, 70))
        for lA in LAYOUTS, lB in LAYOUTS, opA in (:N, :T), opB in (:N, :T)
            i = hash((lA, lB, opA, opB)) % length(αβ) + 1
            @test check_16(T, m, n, k, lA, opA, lB, opB, :col, αβ[i]...)
        end
    end
    for lC in OUT_LAYOUTS, (α, β) in αβ
        @test check_16(T, 23, 19, 31, :stride2, :T, :negrow, :N, lC, α, β)
    end
    @test check_16(T, 301, 263, 517, :col, :N, :row, :N, :negrow, -1.5f0,
                   0.25f0)
end
