#!/bin/bash
# Experiment: build LLVMCompilerRT 20.1.8 (same changes as crt.sh) for one platform.
set -euxo pipefail
W=$GITHUB_WORKSPACE
cd $W/ygg/L/LLVMCompilerRT
sed -n '/^version = /p' build_tarballs.jl
sed -i 's/^version = v"17.0.6"/version = v"20.1.8"/' build_tarballs.jl
sed -i 's/11b8d09dcf92a0f91c5c82defb5ad9ff4acf5cf073a80c317204baa922d136b4/15277402f6fd63397c0917a5c7171cda82d16d226094b828c1ed0f58f73b9c69/' build_tarballs.jl
sed -i 's/807f069c54dc20cb47b21c1f6acafdd9c649f3ae015609040d6182cab01140f4/3319203cfd1172bbac50f06fa68e318af84dcb5d65353310c0586354069d6634/' build_tarballs.jl
sed -i 's/BuildDependency(PackageSpec(name="LLVM_full_jll"; version))/BuildDependency(PackageSpec(name="LLVM_full_jll", version=string(version)))/' build_tarballs.jl
sed -i '/atomic_patch -p1 ..\/patches\/do-not-codesign.patch/d; /# We.ll codesign during audit/d' build_tarballs.jl
sed -i 's|    -DCMAKE_LIBTOOL=$(which libtool) \\|    -DCMAKE_LIBTOOL=$(which libtool) \\\n    -DCODESIGN=OFF \\|' build_tarballs.jl
sed -i 's|julia_compat="1.6", preferred_gcc_version=v"8", preferred_llvm_version=version)|julia_compat="1.6", preferred_gcc_version=v"8", preferred_llvm_version=version,\n               lock_microarchitecture=false)|' build_tarballs.jl
if [[ "${CRT_MUSL_NO_SANITIZERS:-}" == 1 ]]; then
    sed -i 's|^FLAGS=()$|FLAGS=()\nif [[ "${target}" == *-musl* ]]; then\n    FLAGS+=(-DCOMPILER_RT_BUILD_SANITIZERS=OFF)\nfi|' build_tarballs.jl
fi
git -C $W/ygg diff
julia --project=$W/ygg/.ci/bb1_project build_tarballs.jl --verbose $1
mkdir -p $W/out && tar -tzf products/LLVMCompilerRT.v*.tar.gz | grep -E '\.(a|so|dylib|dll)$' | sort > $W/out/files-$1.txt
wc -l $W/out/files-$1.txt
