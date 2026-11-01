#!/bin/bash
# Experiment: build a recipe that depends on LLVMCompilerRT_jll without a
# version bound, either as registered (17.0.6) or against LLVMCompilerRT
# 20.1.8 built by crt-platform.sh.
#   crt-consumer.sh RECIPE_DIR PLATFORM CRT_PLATFORM 17|20
set -euxo pipefail
W=$GITHUB_WORKSPACE
if [[ "$4" == 20 ]]; then
    CRT_MUSL_NO_SANITIZERS=1 CRT_DEPLOY=local bash $W/pkg/.github/experiments/crt-platform.sh $3
    J=$HOME/.julia/dev/LLVMCompilerRT_jll
    git -C $J init -q && git -C $J add -A && git -C $J -c user.name=ci -c user.email=ci@localhost commit -qm jll
    sed -i "s|BuildDependency(\"LLVMCompilerRT_jll\"|BuildDependency(Pkg.PackageSpec(name=\"LLVMCompilerRT_jll\", path=\"$J\")|" $W/ygg/$1/build_tarballs.jl
    sed -i '1i import Pkg' $W/ygg/$1/build_tarballs.jl
fi
git -C $W/ygg diff -- $1
cd $W/ygg/$1
julia --project=$W/ygg/.ci/bb1_project build_tarballs.jl --verbose $2
mkdir -p $W/out && ls -l products | tee $W/out/products-$4.txt
