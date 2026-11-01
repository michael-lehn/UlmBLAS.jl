#!/bin/bash
# Experiment: LLVMCompilerRT 20.1.8 (SME routines for Darwin) and the ulmBLAS
# recipe linking libclang_rt.osx.a from it instead of bundling compiler-rt.
set -euxo pipefail
W=$GITHUB_WORKSPACE
P="--project=$W/ygg/.ci/bb1_project"
cd $W/ygg/L/LLVMCompilerRT
sed -i 's/^version = v"17.0.6"/version = v"20.1.8"/' build_tarballs.jl
sed -i 's/11b8d09dcf92a0f91c5c82defb5ad9ff4acf5cf073a80c317204baa922d136b4/15277402f6fd63397c0917a5c7171cda82d16d226094b828c1ed0f58f73b9c69/' build_tarballs.jl
sed -i 's/807f069c54dc20cb47b21c1f6acafdd9c649f3ae015609040d6182cab01140f4/3319203cfd1172bbac50f06fa68e318af84dcb5d65353310c0586354069d6634/' build_tarballs.jl
sed -i 's/BuildDependency(PackageSpec(name="LLVM_full_jll"; version))/BuildDependency(PackageSpec(name="LLVM_full_jll", version=string(version)))/' build_tarballs.jl
git -C $W/ygg diff
julia $P build_tarballs.jl --verbose --deploy=local aarch64-apple-darwin
mkdir -p $W/out/crt && tar -xzf products/*.tar.gz -C $W/out/crt
find $W/out/crt -name '*.a' | head
J=$HOME/.julia/dev/LLVMCompilerRT_jll
git -C $J init -q && git -C $J add -A && git -C $J -c user.name=ci -c user.email=ci@localhost commit -qm jll
cd $W/ygg/U/ulmBLAS
python3 - <<PY
p='build_tarballs.jl'; s=open(p).read()
a=s.index('    # The SME support routines'); b=s.index('                  "15277402')
s=s[:a]+s[s.index('\n',b)+1:]
a=s.index('        if [[ \${v} == sme ]]; then'); b=s.index('        fi\n',a)
s=s[:a]+'        [[ \${v} == sme ]] && link+=" -L\${libdir}/darwin -lclang_rt.osx"\n'+s[b+len('        fi\n'):]
a=s.index('if [[ \${target} == *-apple-* ]]; then\n    install_license'); b=s.index('fi\n',a)
s=s[:a]+s[b+3:]
s=s.replace('dependencies = Dependency[]','dependencies = [\n    BuildDependency(PackageSpec(name="LLVMCompilerRT_jll", path="$J");\n                    platforms=filter(Sys.isapple, platforms)),\n]')
open(p,'w').write(s)
PY
cat build_tarballs.jl
julia $P build_tarballs.jl --verbose aarch64-apple-darwin
mkdir -p $W/out/ulm && tar -xzf products/*.tar.gz -C $W/out/ulm && ls -l $W/out/ulm/lib
