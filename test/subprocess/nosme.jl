#  Run in a fresh process by runtests.jl on Apple silicon without SME2, with
#  ULMBLAS_VARIANT=sme: the request is refused, neon is used and the sme
#  library is not loaded.
using Libdl, Test
using UlmBLAS

@testset "no sme library loaded" begin
    @test ENV["ULMBLAS_VARIANT"] == "sme"
    @test UlmBLAS.chosen[] == "neon"
    @test basename.(filter(contains("libulmblas_"), dllist())) == ["libulmblas_neon.dylib"]
end
