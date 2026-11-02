#
#  LD_DEBUG=bindings julia --project=lapack lapack/bindings.jl 2>&1 |
#      grep 'liblapack.*64_' (Linux): the dynamic linker binds the BLAS
#  routines Netlib LAPACK calls during DGETRF to libblastrampoline, and the
#  LAPACK routines it calls (dgetrf2_64_, dlaswp_64_, ilaenv_64_, ...) to
#  liblapack itself.
#
include("common.jl")
configure(:netlib_ulmblas)
A = randn(200, 200)
dgetrf!(A, zeros(BlasInt, 200)) == 0 || error("singular")
configure(:openblas)
