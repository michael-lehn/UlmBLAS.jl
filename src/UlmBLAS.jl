"""
    UlmBLAS

Use the BLAS of [ulmBLAS](https://github.com/michael-lehn/ulmBLAS) in Julia.

`using UlmBLAS` loads the shared library of ulmBLAS for this CPU, but does
not change the BLAS of Julia:

- [`UlmBLAS.mul!`](@ref) (GEMM with arbitrary strides) works right away.
- [`UlmBLAS.forward()`](@ref forward) loads the library into
  libblastrampoline, the BLAS forwarding layer of Julia, on top of the
  libraries already loaded (usually OpenBLAS): the BLAS routines of
  ulmBLAS then replace those of OpenBLAS, and LAPACK and every routine
  ulmBLAS does not provide stay with OpenBLAS.

ulmBLAS selects its micro kernels when it is compiled.  The package
therefore contains one library per kind of CPU and picks the one for this
CPU when it is loaded (see [`variant`](@ref)).  For development, the
environment variable `ULMBLAS_LIBRARY` (path of a library built with
`make lbt`) overrides the choice, and `ULMBLAS_VARIANT` selects another
variant of the package's libraries (if the CPU can run it).
"""
module UlmBLAS

using Libdl, LinearAlgebra, ulmBLAS_jll
using Base.BinaryPlatforms.CPUID

const LIBNAME = Sys.isapple() ? "libulmblas_lbt.dylib" : "libulmblas_lbt.so"

#  path and variant of the library in use (set by __init__)
const library = Ref{String}("")
const chosen  = Ref{String}("")

#  macOS: the value of an integer sysctl (sysctlbyname of libSystem, no
#  process), nothing if there is none
function sysctl_int(name)
    Sys.isapple() || return nothing
    value = Ref{Int64}(0)
    len = Ref{Csize_t}(sizeof(Int64))
    r = ccall(:sysctlbyname, Cint,
              (Cstring, Ptr{Int64}, Ptr{Csize_t}, Ptr{Cvoid}, Csize_t),
              name, value, len, C_NULL, 0)
    r == 0 || return nothing
    len[] == 8 && return Int(value[])
    len[] == 4 && return Int(value[] % Int32)    # little endian
    return nothing
end

#  AArch64 (macOS): the variants for the values sysctl(name) reports.  The
#  SME kernels need SME2, FEAT_SME_F64F64 and a streaming vector length of
#  512 bits (64 bytes).  macOS sets the largest vector length the CPU
#  supports and reports the length it reads back (RDSVL) as
#  hw.optional.arm.sme_max_svl_b; processes cannot change it.  neon if any
#  value is missing or different.
function arm_variants(sysctl)
    sme = sysctl("hw.optional.arm.FEAT_SME2") == 1 &&
          sysctl("hw.optional.arm.FEAT_SME_F64F64") == 1 &&
          sysctl("hw.optional.arm.sme_max_svl_b") == 64
    return sme ? ["sme", "neon"] : ["neon"]
end

is_amd() = occursin("AMD", Sys.cpu_info()[1].model)
has(feature) = CPUID.test_cpu_feature(feature)

"""
    variants()

The builds of ulmBLAS in ulmBLAS_jll this CPU can run, the preferred one
first ([`cpu_variants`](@ref) for which ulmBLAS_jll has a library).
"""
variants() = filter(has_library, cpu_variants())

#  ulmBLAS_jll has a library of variant v for this platform (ulmBLAS_jll 0.2
#  has no sme library: SME2 needs a newer LLVMCompilerRT_jll)
has_library(v) = isdefined(ulmBLAS_jll, Symbol("libulmblas_", v, "_path"))

"""
    cpu_variants()

The builds of ulmBLAS this CPU can run, the preferred one first:

- AArch64: `"sme"` (Apple M4 and later: SME2 with FEAT_SME_F64F64 and a
  streaming vector length of 512 bits), `"neon"`;
- x86-64: `"zen4"` (AMD with AVX-512 F, DQ and VL: Zen 4, Zen 5) or
  `"skylakex"` (Intel with AVX-512 F, DQ and VL), `"haswell"` (AVX2 and
  FMA), `"x86_64"` (SSE3).  With AVX-512 both `"zen4"` and `"skylakex"` run.
"""
function cpu_variants()
    if Sys.ARCH === :aarch64
        return Sys.isapple() ? arm_variants(sysctl_int) : ["neon"]
    elseif Sys.ARCH === :x86_64
        return x86_variants(has, is_amd())
    end
    return String[]
end

#  x86-64: the variants for the features has(CPUID.JL_X86_...) reports, AMD
#  preferring zen4.  The AVX-512 libraries are compiled with AVX-512 F and DQ,
#  and the vendored assembly kernels also use AVX-512 VL.
function x86_variants(has, amd)
    v = String[]
    if has(CPUID.JL_X86_avx512f) && has(CPUID.JL_X86_avx512dq) &&
       has(CPUID.JL_X86_avx512vl)
        append!(v, amd ? ["zen4", "skylakex"] : ["skylakex", "zen4"])
    end
    has(CPUID.JL_X86_avx2) && has(CPUID.JL_X86_fma) && push!(v, "haswell")
    has(CPUID.JL_X86_sse3) && push!(v, "x86_64")
    return v
end

"""
    variant()

The build of ulmBLAS for this CPU (the first of [`variants`](@ref)), or
`nothing` if there is none.
"""
variant() = (v = variants(); isempty(v) ? nothing : first(v))

#  the library of variant v from ulmBLAS_jll (nothing if there is none)
function artifact_library(v)
    ulmBLAS_jll.is_available() && has_library(v) || return nothing
    path = getfield(ulmBLAS_jll, Symbol("libulmblas_", v, "_path"))
    return isfile(path) ? path : nothing
end

#  the library to use: ULMBLAS_LIBRARY, or the variant ULMBLAS_VARIANT
#  (if this CPU runs it) or variant() from the artifact; nothing (with a
#  warning) if there is none
function find_library()
    path = get(ENV, "ULMBLAS_LIBRARY", "")
    if !isempty(path)
        chosen[] = "ULMBLAS_LIBRARY"
        isfile(path) && return path
        @warn "UlmBLAS: ULMBLAS_LIBRARY=$path does not exist"
        return nothing
    end
    v = get(ENV, "ULMBLAS_VARIANT", "")
    if !isempty(v) && !(v in variants())
        @warn "UlmBLAS: ULMBLAS_VARIANT=$v does not run on this CPU " *
              "(variants: $(join(variants(), ", ")))"
        v = ""
    end
    isempty(v) && (v = variant())
    if v === nothing
        @warn "UlmBLAS: no build of ulmBLAS for $(Sys.ARCH)"
        return nothing
    end
    chosen[] = v
    lib = artifact_library(v)
    lib === nothing && @warn "UlmBLAS: no library for this platform " *
        "(variant $v); set ULMBLAS_LIBRARY to a library built with `make lbt`"
    return lib
end

"""
    library_path()

Path of the ulmBLAS library in use (`""` if there is none).
"""
library_path() = library[]

"""
    forward() -> number of routines forwarded

Make ulmBLAS the BLAS of Julia: load the library of ulmBLAS into
libblastrampoline on top of the libraries already loaded
(`BLAS.lbt_forward(library_path(); clear=false)`).  Only the routines
ulmBLAS provides are replaced; LAPACK and every other routine (e.g.
`?gemmt`, `?axpby`, the CBLAS dot products) keep their previous library,
usually OpenBLAS.  Calling `forward()` again changes nothing.

There is deliberately no `clear` option: after `clear=true`
libblastrampoline would leave the routines ulmBLAS does not have without
an implementation (calls only print an error message and return without
a result).  To undo `forward()`, load the previous library again with
`clear=true`, e.g. `BLAS.lbt_forward(OpenBLAS_jll.libopenblas_path;
clear=true)`.
"""
#  libblastrampoline does not synchronize lbt_forward: concurrent calls
#  write its list of loaded libraries (record_library_load in config.c of
#  LBT 5.15) without a lock.  forward() calls are serialized by this lock
#  (calls of BLAS.lbt_forward by other code are not).
const FORWARD_LOCK = ReentrantLock()

#  A library built with LAPACK (OPENMP=1 LAPACK=1) has ?getrf, ?potrf, ...
#  and its own number of threads.  Its thread functions are registered with
#  libblastrampoline before forwarding (libblastrampoline appends the suffix
#  64_ itself); then BLAS.get_num_threads and BLAS.set_num_threads include
#  ulmBLAS, and ulmBLAS takes the number of threads Julia gives BLAS.
const THREADS_REGISTERED = Ref(false)

function has_thread_functions(path)
    h = Libdl.dlopen(path; throw_error=false)
    h === nothing && return false
    return Libdl.dlsym(h, :ulmblas_get_num_threads_64_; throw_error=false) !== nothing
end

function forward()
    path = library[]
    isempty(path) && error("UlmBLAS: no library of ulmBLAS loaded")
    n = lock(FORWARD_LOCK) do
        threads = BLAS.get_num_threads()
        lapack = has_thread_functions(path)
        if lapack && !THREADS_REGISTERED[]
            ccall((:lbt_register_thread_interface, LinearAlgebra.libblastrampoline),
                  Cvoid, (Cstring, Cstring),
                  "ulmblas_get_num_threads_", "ulmblas_set_num_threads_")
            THREADS_REGISTERED[] = true
        end
        k = BLAS.lbt_forward(path; clear=false, suffix_hint="64_")
        lapack && BLAS.set_num_threads(threads)
        k
    end
    n > 0 || error("UlmBLAS: no BLAS routines forwarded from $path")
    return n
end

function __init__()
    path = find_library()
    path === nothing && return
    #  a library that cannot be loaded (e.g. built for a newer glibc): as
    #  without a library, the BLAS of Julia stays unchanged
    try
        load_strided(path)
    catch err
        @warn "UlmBLAS: library $path cannot be loaded" exception = err
        return
    end
    library[] = path
end

include("strided.jl")

end # module
