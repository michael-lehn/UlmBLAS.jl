#
#  julia build/artifacts.jl TAG DIR=TRIPLET ...
#
#  Packs the output directories of build.sh (one per platform) as artifacts
#  and binds them in Artifacts.toml (artifact "ulmblas", one entry per
#  platform), with the assets of the release TAG of this repository as
#  download location:
#
#      https://github.com/michael-lehn/UlmBLAS.jl/releases/download/TAG/
#          TAG.TRIPLET.tar.gz
#
#  TAG names the libraries, not a version of the package, e.g.
#  libulmblas-d738372b for libraries built from ulmBLAS d738372b.  The
#  tarballs are written to dist/; upload them to the release TAG.
#  TRIPLET: aarch64-apple-darwin or x86_64-linux-gnu.
#
#      julia build/artifacts.jl libulmblas-d738372b out-mac=aarch64-apple-darwin \
#                                                   out-linux=x86_64-linux-gnu
#
#  With BASEURL=file:///path/to/dist the tarballs are bound with that
#  location instead, to test the installation before the release exists.
#
using Pkg.Artifacts, Base.BinaryPlatforms, SHA

const REPO = "https://github.com/michael-lehn/UlmBLAS.jl"

tag = ARGS[1]
toml = joinpath(@__DIR__, "..", "Artifacts.toml")
dist = joinpath(@__DIR__, "..", "dist")
base = get(ENV, "BASEURL", "$REPO/releases/download/$tag")
mkpath(dist)

for arg in ARGS[2:end]
    dir, triplet = split(arg, "=")
    platform = parse(Platform, triplet)
    hash = create_artifact() do art
        cp(dir, art; force=true)
    end
    name = "$tag.$triplet.tar.gz"
    tarball = joinpath(dist, name)
    archive_artifact(hash, tarball)
    sha = bytes2hex(open(sha256, tarball))
    bind_artifact!(toml, "ulmblas", hash; platform=platform, force=true,
                   download_info=[("$base/$name", sha)])
    println("$triplet: git-tree-sha1 $hash, $name sha256 $sha")
end
