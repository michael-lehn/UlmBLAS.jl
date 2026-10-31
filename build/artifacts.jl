#
#  julia build/artifacts.jl VERSION DIR=TRIPLET ...
#
#  Packs the output directories of build.sh (one per platform) as artifacts
#  and binds them in Artifacts.toml (artifact "ulmblas", one entry per
#  platform), with the release assets of this repository as download
#  location:
#
#      https://github.com/michael-lehn/ulmBLAS.jl/releases/download/vVERSION/
#          ulmBLAS.vVERSION.TRIPLET.tar.gz
#
#  The tarballs are written to dist/; upload them to the release vVERSION.
#  TRIPLET: aarch64-apple-darwin or x86_64-linux-gnu.
#
#      julia build/artifacts.jl 0.1.0 out-mac=aarch64-apple-darwin \
#                                     out-linux=x86_64-linux-gnu
#
using Pkg.Artifacts, Base.BinaryPlatforms, SHA

const REPO = "https://github.com/michael-lehn/ulmBLAS.jl"

version = ARGS[1]
toml = joinpath(@__DIR__, "..", "Artifacts.toml")
dist = joinpath(@__DIR__, "..", "dist")
mkpath(dist)

for arg in ARGS[2:end]
    dir, triplet = split(arg, "=")
    platform = parse(Platform, triplet)
    hash = create_artifact() do art
        cp(dir, art; force=true)
    end
    name = "ulmBLAS.v$version.$triplet.tar.gz"
    tarball = joinpath(dist, name)
    archive_artifact(hash, tarball)
    sha = bytes2hex(open(sha256, tarball))
    url = "$REPO/releases/download/v$version/$name"
    bind_artifact!(toml, "ulmblas", hash; platform=platform, force=true,
                   download_info=[(url, sha)])
    println("$triplet: git-tree-sha1 $hash, $name sha256 $sha")
end
