#
#  UlmBLAS.mul! for BFloat16 (package BFloat16s): A, B BFloat16, C Float32
#  with ulmblas_sbgemm_strided (products and sums in Float32).
#
module UlmBLASBFloat16sExt

using UlmBLAS: UlmBLAS, StridedOp, StridedOut, gemm16!
using BFloat16s: BFloat16

function UlmBLAS.mul!(C::StridedOut{Float32}, A::StridedOp{BFloat16},
                      B::StridedOp{BFloat16}, α::Number=true,
                      β::Number=false)
    return gemm16!("ulmblas_sbgemm_strided", C, A, B, α, β)
end

end
