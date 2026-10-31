#  Run in a fresh process by runtests.jl: an invalid argument to a routine
#  of ulmBLAS is reported (xerbla) and does not end the process.
using LinearAlgebra, Test
using UlmBLAS

@testset "invalid argument" begin
    A = rand(4, 4); C = zeros(4, 4)
    #  dgemm with lda = 1 < m = 4
    ccall((:dgemm_64_, BLAS.libblastrampoline), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ref{Int64},
           Ref{Float64}, Ptr{Float64}, Ref{Int64}, Ptr{Float64}, Ref{Int64},
           Ref{Float64}, Ptr{Float64}, Ref{Int64}, Csize_t, Csize_t),
          'N', 'N', 4, 4, 4, 1.0, A, 1, A, 4, 0.0, C, 4, 1, 1)
    @test iszero(C)
    @test A * A ≈ [sum(A[i, l] * A[l, j] for l in 1:4) for i in 1:4, j in 1:4]
end
