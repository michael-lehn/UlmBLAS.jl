#  Windows diagnostics (run with ULMBLAS_LIBRARY): xerbla, invalid
#  arguments through libblastrampoline, and the complex return style of
#  libblastrampoline for ulmBLAS (zdotc, zdotu).
using Libdl, LinearAlgebra, UlmBLAS
step(s) = (println(s); flush(stdout))
lib = UlmBLAS.library_path()
step("julia $VERSION, library $lib")
mode = get(ARGS, 1, "all")
if mode in ("all", "xerbla")
    ccall((:xerbla_64_, lib), Cvoid, (Ptr{UInt8}, Ref{Int64}, Csize_t), "DGEMM ", 8, 6)
    step("xerbla_64_ returned")
end
UlmBLAS.forward()
for l in BLAS.get_config().loaded_libs
    step("loaded: $(basename(l.libname)) $(l.interface) complex_retstyle=$(l.complex_retstyle)")
end
if mode in ("all", "dgemm")
    A = rand(4, 4); C = zeros(4, 4)
    ccall((:dgemm_64_, BLAS.libblastrampoline), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ref{Int64},
           Ref{Float64}, Ptr{Float64}, Ref{Int64}, Ptr{Float64}, Ref{Int64},
           Ref{Float64}, Ptr{Float64}, Ref{Int64}, Csize_t, Csize_t),
          'N', 'N', 4, 4, 4, 1.0, A, 1, A, 4, 0.0, C, 4, 1, 1)
    step("dgemm with lda = 1 returned, C zero: $(iszero(C))")
end
if mode in ("all", "zdot")
    #  zdotc through libblastrampoline against ulmBLAS's zdotc_64_ (called
    #  directly, C ABI) and the value of the library loaded before
    h = Libdl.dlopen(lib)
    x = rand(ComplexF64, 100_003); y = rand(ComplexF64, 100_003)
    direct = ccall(Libdl.dlsym(h, :zdotc_64_), ComplexF64,
                   (Ref{Int64}, Ptr{ComplexF64}, Ref{Int64}, Ptr{ComplexF64}, Ref{Int64}),
                   length(x), x, 1, y, 1)
    viaLBT = BLAS.dotc(x, y)
    step("zdotc: direct $direct, via LBT $viaLBT, bitwise equal: $(direct === viaLBT)")
    ob = BLAS.get_config().loaded_libs[1].libname
    hob = Libdl.dlopen(ob)
    zdot(f) = ccall(f, ComplexF64,
                    (Ref{Int64}, Ptr{ComplexF64}, Ref{Int64}, Ptr{ComplexF64}, Ref{Int64}),
                    length(x), x, 1, y, 1)
    openblas = zdot(Libdl.dlsym(hob, :zdotc_64_))
    step("zdotc of OpenBLAS ($(basename(ob))) directly: $openblas, equal to via LBT: $(openblas === viaLBT)")
    #  the same for ddot (normal return style): routed to ulmBLAS
    xd = real.(x); yd = real.(y)
    dd(f) = ccall(f, Float64, (Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}),
                  length(xd), xd, 1, yd, 1)
    step("ddot: ulmBLAS $(dd(Libdl.dlsym(h, :ddot_64_))), OpenBLAS $(dd(Libdl.dlsym(hob, :ddot_64_))), " *
         "via LBT (ccall ddot_64_) $(ccall((:ddot_64_, BLAS.libblastrampoline), Float64, (Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}), length(xd), xd, 1, yd, 1))")
    step("forward zdotc_: $(BLAS.lbt_get_forward("zdotc_", :ilp64)), dlsym: $(Libdl.dlsym(h, :zdotc_64_))")
end
step("done")
