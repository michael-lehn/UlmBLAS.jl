#  The subprocess scripts of test/runtests.jl one at a time, as runtests.jl
#  runs them (same environment, threads), each with a time limit: which one
#  does not finish?  (Windows diagnostics)
using UlmBLAS
const DIR = joinpath(dirname(pathof(UlmBLAS)), "..", "test", "subprocess")
function run_limited(file, env...; threads=1, limit=600)
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$threads
           --project=$(Base.active_project()) $(joinpath(DIR, file))`
    out = IOBuffer(); err = IOBuffer()
    t0 = time()
    p = run(pipeline(ignorestatus(addenv(cmd, env...)); stdout=out, stderr=err); wait=false)
    while !process_exited(p) && time() - t0 < limit
        sleep(1)
    end
    done = process_exited(p)
    done || kill(p)
    sleep(2)
    println("== $file $(join(env, " ")) threads=$threads: ",
            done ? "exit code $(p.exitcode)" : "NOT FINISHED (killed)",
            " after $(round(time() - t0; digits=1)) s")
    o = String(take!(out)); e = String(take!(err))
    (done && p.exitcode == 0) || println("stdout (last 3000):\n", last(o, 3000), "\nstderr (last 3000):\n", last(e, 3000))
    flush(stdout)
end
println("julia $VERSION, variants $(UlmBLAS.variants()), library $(UlmBLAS.library_path())")
run_limited("errors.jl")
run_limited("threads.jl"; threads=4)
for v in UlmBLAS.variants()
    run_limited("variant.jl", "ULMBLAS_VARIANT" => v)
end
run_limited("init.jl", "ULMBLAS_LIBRARY" => "/nonexistent/libulmblas_lbt")
run_limited("init.jl", "ULMBLAS_LIBRARY" => joinpath(DIR, "..", "runtests.jl"))
run_limited("linalg.jl", "LINALG_FILES" => "blas.jl matmul.jl"; limit=1500)
