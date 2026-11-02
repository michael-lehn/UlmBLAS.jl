#
#  Each BLAS routine within Netlib's DGETRF: ulmBLAS against OpenBLAS's
#  BLAS, from the results of profile.jl, one panel per machine:
#
#      julia plot_profile.jl OUTDIR "Ice Lake=il.profile.tsv" "Zen 4=zen4.profile.tsv"
#
#  writes OUTDIR/profile-ROUTINE.svg for dgemm, dtrsm, idamax, dscal and
#  lapack (the rest of DGETRF: DLASWP and the code of DGETRF and DGETRF2,
#  the same code in both configurations), random matrices: top, the time
#  in the routine per DGETRF with OpenBLAS's BLAS and with ulmBLAS (medians,
#  log scale); bottom, the ratio ulmBLAS / OpenBLAS (above 1 = slower with
#  ulmBLAS); against n (linear).  (The column-dominant matrices make the
#  same BLAS calls and give the same curves.)  MATRIX=coldom plots those.
#  Only Julia's standard library is used.
#
const ROUTINES = ["dgemm", "dtrsm", "idamax", "dscal", "lapack"]
const NAMES = Dict("dgemm" => "DGEMM", "dtrsm" => "DTRSM", "idamax" => "IDAMAX",
                   "dscal" => "DSCAL", "lapack" => "Rest of DGETRF (DLASWP, LAPACK code)")
const MATRIX = get(ENV, "MATRIX", "dense")
const CONFIGS = ["netlib_openblas", "netlib_ulmblas"]
const LABELS = Dict("netlib_openblas" => "OpenBLAS's BLAS", "netlib_ulmblas" => "ulmBLAS")
# categorical slots 1 and 2 (light, dark), fixed order
const COLORS      = ["#2a78d6", "#eb6834"]
const COLORS_DARK = ["#3987e5", "#d95926"]

outdir = ARGS[1]
machines = map(ARGS[2:end]) do a
    name, path = split(a, '='; limit=2)
    lines = readlines(path)
    header = split(lines[1], '\t')
    col(c) = findfirst(==(c), header)
    t = Dict{Tuple{String,String,String,Int},Float64}()
    calls = Dict{Tuple{String,String,Int},Int}()
    for l in lines[2:end]
        f = split(l, '\t')
        n = parse(Int, f[col("n")])
        t[(f[col("matrix")], f[col("config")], f[col("routine")], n)] =
            parse(Float64, f[col("median_ns")])
        calls[(f[col("matrix")], f[col("routine")], n)] = parse(Int, f[col("calls")])
    end
    sizes = sort(unique(last.(keys(t))))
    (name=String(name), t=t, calls=calls, sizes=sizes)
end

fmt_time(ns) = ns < 1e3 ? string(round(Int, ns), " ns") :
               ns < 1e6 ? string(round(ns/1e3, digits=1), " µs") :
               string(round(ns/1e6, digits=1), " ms")
ratio(m, k, rt, n) = m.t[(k, "netlib_ulmblas", rt, n)]/m.t[(k, "netlib_openblas", rt, n)]

#  one panel at (x0, y0): x = n (linear), y = value(m, config or nothing, n)
function panel(p, m, x0, y0, pw, ph, series, value, ylog, yticks, ytick_label, ylab, tip; ref=nothing)
    r(v) = round(v, digits=1)
    xmax = 512*ceil(Int, last(m.sizes)/512)
    lx(n) = x0 + pw*n/xmax
    f = ylog ? log : identity
    ly(y) = y0 + ph*(1 - (f(y) - f(first(yticks)))/(f(last(yticks)) - f(first(yticks))))
    for y in yticks
        p("""<line class="grid" x1="$x0" x2="$(x0 + pw)" y1="$(r(ly(y)))" y2="$(r(ly(y)))"/>""")
        p("""<text class="t2" x="$(x0 - 7)" y="$(r(ly(y)) + 4)" font-size="11" text-anchor="end">$(ytick_label(y))</text>""")
    end
    ref === nothing || p("""<line class="ref" x1="$x0" x2="$(x0 + pw)" y1="$(r(ly(ref)))" y2="$(r(ly(ref)))"/>""")
    for n in 0:(xmax > 2048 ? 1024 : 512):xmax
        p("""<text class="t2" x="$(r(lx(n)))" y="$(y0 + ph + 16)" font-size="11" text-anchor="middle">$n</text>""")
    end
    p("""<line class="axis" x1="$x0" x2="$(x0 + pw)" y1="$(y0 + ph)" y2="$(y0 + ph)"/>""")
    p("""<text class="t2" font-size="12" text-anchor="middle" transform="translate($(x0 - 62) $(y0 + ph/2)) rotate(-90)">$ylab</text>""")
    for (i, c) in series
        pts = [(n, value(m, c, n)) for n in m.sizes]
        p("""<path class="s$i" d="$(join(["$(q == 1 ? 'M' : 'L')$(r(lx(n))) $(r(ly(y)))" for (q, (n, y)) in enumerate(pts)], " "))"/>""")
        for (n, y) in pts
            p("""<circle class="m$i" cx="$(r(lx(n)))" cy="$(r(ly(y)))" r="3.5"><title>$(tip(m, c, n))</title></circle>""")
        end
    end
end

ticks_log(lo, hi) = [10.0^k for k in floor(Int, log10(lo)):ceil(Int, log10(hi))]
fmt_tick(ns) = ns < 1e3 ? "$(round(Int, ns)) ns" : ns < 1e6 ? "$(round(Int, ns/1e3)) µs" :
               ns < 1e9 ? "$(round(Int, ns/1e6)) ms" : "$(round(Int, ns/1e9)) s"

function figure(rt)
    pw, ph, left, gapx, top, gapy = 380, 230, 90, 100, 92, 80
    np = length(machines)
    W = left + np*pw + (np - 1)*gapx + 24
    H = top + 2ph + gapy + 70
    io = IOBuffer()
    p(x...) = print(io, x..., '\n')
    p("""<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" viewBox="0 0 $W $H" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif">""")
    p("<title>$(NAMES[rt]) within Netlib's DGETRF: ulmBLAS against OpenBLAS's BLAS</title>")
    p("<style>")
    p("  .surface { fill: #fcfcfb; } .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; }")
    p("  .grid { stroke: #e3e2de; stroke-width: 1; } .axis { stroke: #8f8e88; stroke-width: 1; }")
    p("  .ref { stroke: #8f8e88; stroke-width: 1.5; stroke-dasharray: 5 4; }")
    for i in 1:2
        p("  .s$i { stroke: $(COLORS[i]); fill: none; stroke-width: 2; } .m$i { fill: $(COLORS[i]); stroke: #fcfcfb; stroke-width: 2; } .k$i { fill: $(COLORS[i]); }")
    end
    p("  .s3 { stroke: #52514e; fill: none; stroke-width: 2; } .m3 { fill: #52514e; stroke: #fcfcfb; stroke-width: 2; }")
    p("  @media (prefers-color-scheme: dark) {")
    p("    .surface { fill: #1a1a19; } .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; }")
    p("    .grid { stroke: #34332f; } .axis, .ref { stroke: #6b6a65; }")
    for i in 1:2
        p("    .s$i { stroke: $(COLORS_DARK[i]); } .m$i { fill: $(COLORS_DARK[i]); stroke: #1a1a19; } .k$i { fill: $(COLORS_DARK[i]); }")
    end
    p("    .s3 { stroke: #c3c2b7; } .m3 { fill: #c3c2b7; stroke: #1a1a19; }")
    p("  }")
    p("</style>")
    p("""<rect class="surface" x="0" y="0" width="$W" height="$H" rx="8"/>""")
    what = MATRIX == "dense" ? "random matrices" : "column-dominant matrices"
    p("""<text class="t1" x="24" y="28" font-size="15" font-weight="600">$(NAMES[rt]) within Netlib's DGETRF ($what): ulmBLAS against OpenBLAS's BLAS</text>""")
    lx0 = 24
    for (i, c) in enumerate(CONFIGS)
        p("""<rect class="k$i" x="$lx0" y="44" width="12" height="12" rx="2"/>""")
        p("""<text class="t2" x="$(lx0 + 17)" y="54" font-size="12">$(LABELS[c])</text>""")
        lx0 += 17 + 6.4*length(LABELS[c]) + 24
    end
    p("""<text class="t2" x="$lx0" y="54" font-size="12">(bottom: time ulmBLAS / time OpenBLAS)</text>""")
    t(m, c, n) = m.t[(MATRIX, c, rt, n)]
    rat(m, _, n) = t(m, "netlib_ulmblas", n)/t(m, "netlib_openblas", n)
    times = [t(m, c, n) for m in machines for c in CONFIGS for n in m.sizes]
    v = [rat(m, nothing, n) for m in machines for n in m.sizes]
    hi = max(maximum(v), 1.0); lo = min(minimum(v), 1.0)
    step = hi - lo > 6 ? 2.0 : hi - lo > 3 ? 1.0 : hi - lo > 1.2 ? 0.5 : hi - lo > 0.5 ? 0.25 : hi - lo > 0.2 ? 0.1 : 0.05
    rticks = collect(step*floor(lo/step):step:step*ceil(hi/step))
    callinfo(m, n) = rt == "lapack" ? "" : ", $(m.calls[(MATRIX, rt, n)]) calls"
    for (j, m) in enumerate(machines)
        x0 = left + (j - 1)*(pw + gapx)
        p("""<text class="t1" x="$x0" y="$(top - 14)" font-size="13" font-weight="600">$(m.name)</text>""")
        panel(p, m, x0, top, pw, ph, enumerate(CONFIGS), t, true,
              ticks_log(minimum(times), maximum(times)), fmt_tick, "time per DGETRF",
              (m, c, n) -> "$(NAMES[rt]), $(LABELS[c]), n = $n: $(fmt_time(t(m, c, n))) per DGETRF$(callinfo(m, n))")
        panel(p, m, x0, top + ph + gapy, pw, ph, [(3, nothing)], rat, false,
              rticks, y -> "$(round(y, digits=2))×", "ulmBLAS / OpenBLAS",
              (m, _, n) -> "$(NAMES[rt]), n = $n: $(round(rat(m, nothing, n), digits=2))× ($(fmt_time(t(m, "netlib_openblas", n))) → $(fmt_time(t(m, "netlib_ulmblas", n))))"; ref=1.0)
        p("""<text class="t2" x="$(x0 + pw/2)" y="$(top + 2ph + gapy + 34)" font-size="12" text-anchor="middle">n (n × n matrix)</text>""")
    end
    p("""<text class="t2" x="24" y="$(H - 12)" font-size="11">Netlib LAPACK 3.12.1, the same code in both configurations; every BLAS call timed (median per DGETRF); one thread.</text>""")
    p("</svg>")
    file = joinpath(outdir, MATRIX == "dense" ? "profile-$rt.svg" : "profile-$MATRIX-$rt.svg")
    write(file, String(take!(io)))
    println("wrote ", file)
end

foreach(figure, ROUTINES)
