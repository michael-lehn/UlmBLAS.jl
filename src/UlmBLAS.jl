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

using Artifacts, Libdl, LinearAlgebra
using Base.BinaryPlatforms.CPUID

const LIBNAME = Sys.isapple() ? "libulmblas_lbt.dylib" :
                Sys.iswindows() ? "libulmblas_lbt.dll" : "libulmblas_lbt.so"

#  path and variant of the library in use (set by __init__)
const library = Ref{String}("")
const chosen  = Ref{String}("")

#  Apple: SME2 (M4 and later) is reported by sysctl
function has_sme2()
    try
        return strip(read(`sysctl -n hw.optional.arm.FEAT_SME2`, String)) == "1"
    catch
        return false
    end
end

is_amd() = occursin("AMD", Sys.cpu_info()[1].model)
has(feature) = CPUID.test_cpu_feature(feature)

"""
    variants()

The builds of ulmBLAS this CPU can run, the preferred one first:

- AArch64: `"sme"` (Apple M4 and later, SME2 kernels), `"neon"`;
- x86-64: `"zen4"` (AMD with AVX-512 F and DQ: Zen 4, Zen 5) or
  `"skylakex"` (Intel with AVX-512 F and DQ), `"haswell"` (AVX2 and FMA),
  `"x86_64"` (SSE3).  With AVX-512 both `"zen4"` and `"skylakex"` run.
"""
function variants()
    if Sys.ARCH === :aarch64
        return Sys.isapple() && has_sme2() ? ["sme", "neon"] : ["neon"]
    elseif Sys.ARCH === :x86_64
        v = String[]
        if has(CPUID.JL_X86_avx512f) && has(CPUID.JL_X86_avx512dq)
            append!(v, is_amd() ? ["zen4", "skylakex"] : ["skylakex", "zen4"])
        end
        has(CPUID.JL_X86_avx2) && has(CPUID.JL_X86_fma) && push!(v, "haswell")
        has(CPUID.JL_X86_sse3) && push!(v, "x86_64")
        return v
    end
    return String[]
end

"""
    variant()

The build of ulmBLAS for this CPU (the first of [`variants`](@ref)), or
`nothing` if there is none.
"""
variant() = (v = variants(); isempty(v) ? nothing : first(v))

#  the library of variant v in the artifact of this platform (installed by
#  Pkg together with the package; nothing if there is none)
function artifact_library(v)
    toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    isfile(toml) || return nothing
    meta = artifact_meta("ulmblas", toml)
    meta === nothing && return nothing
    path = joinpath(artifact_path(Base.SHA1(meta["git-tree-sha1"])),
                    "lib", v, LIBNAME)
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

function forward()
    path = library[]
    isempty(path) && error("UlmBLAS: no library of ulmBLAS loaded")
    n = lock(FORWARD_LOCK) do
        BLAS.lbt_forward(path; clear=false, suffix_hint="64_")
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
