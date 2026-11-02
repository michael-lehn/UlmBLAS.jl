#  The CPU, the library UlmBLAS selected and the identity of the artifact of
#  ulmBLAS_jll it was loaded from; fails if they do not match.
using Pkg, UlmBLAS, ulmBLAS_jll

toml = Pkg.Artifacts.load_artifacts_toml(joinpath(pkgdir(ulmBLAS_jll), "Artifacts.toml"))
meta = only(m for m in toml["ulmBLAS"] if m["os"] == "macos" && m["arch"] == "aarch64")
dir = ulmBLAS_jll.artifact_dir
tree = bytes2hex(Pkg.GitTools.tree_hash(dir))
sme = Dict(n => UlmBLAS.sysctl_int("hw.optional.arm.$n")
           for n in ("FEAT_SME2", "FEAT_SME_F64F64", "sme_max_svl_b"))

println("CPU:          ", Sys.cpu_info()[1].model, " (", Sys.CPU_THREADS, " threads)")
println("SME sysctls:  ", join(("$k = $(sme[k])" for k in sort!(collect(keys(sme)))), ", "))
println("variants:     ", join(UlmBLAS.variants(), ", "))
println("selected:     ", UlmBLAS.chosen[], " ", UlmBLAS.library_path())
println("artifact:     ", dir)
println("tree hash:    ", tree, " (Artifacts.toml ", meta["git-tree-sha1"], ")")
println("sha256:       ", only(meta["download"])["sha256"])
println("url:          ", only(meta["download"])["url"])

@assert tree == meta["git-tree-sha1"] == basename(dir)
@assert startswith(UlmBLAS.library_path(), dir)
@assert basename(UlmBLAS.library_path()) == "libulmblas_$(UlmBLAS.chosen[]).dylib"
@assert UlmBLAS.chosen[] == first(UlmBLAS.arm_variants(UlmBLAS.sysctl_int))
