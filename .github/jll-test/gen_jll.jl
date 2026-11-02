#  gen_jll.jl META_JSON TARBALL_DIR URL_PREFIX: the package ulmBLAS_jll for
#  the tarballs in TARBALL_DIR, as Yggdrasil's .ci/bb1_project/register_package.jl
#  generates it, without GitHub: the package goes to ~/.julia/dev, the
#  download URLs are URL_PREFIX/<tarball>.
using BinaryBuilder, BinaryBuilderBase, Pkg

meta_json, tarballs, prefix = ARGS
objs = [BinaryBuilder.JSON.parse(line; dicttype=Dict)
        for line in eachline(meta_json) if !isempty(strip(line))]
objs_unmerged = deepcopy(objs)
merged = BinaryBuilder.merge_json_objects(objs)
BinaryBuilder.cleanup_merged_object!(merged)
BinaryBuilder.cleanup_merged_object!.(objs_unmerged)
for json_obj in [merged, objs_unmerged...]
    json_obj["dependencies"] = Dependency[dep for dep in json_obj["dependencies"]
                                          if BinaryBuilderBase.is_runtime_dependency(dep)]
end
for (i, obj) in enumerate(objs_unmerged)
    BinaryBuilder.rebuild_jll_package(obj; download_dir=tarballs,
                                      build_meta_dir=mktempdir(),
                                      upload_prefix=prefix,
                                      build_version=VersionNumber(string(merged["version"])),
                                      verbose=true, from_scratch=(i == 1))
end
