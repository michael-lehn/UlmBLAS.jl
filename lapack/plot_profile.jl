#
#  Where Netlib LAPACK + ulmBLAS loses against Netlib LAPACK + OpenBLAS's
#  BLAS, from the results of profile.jl, one panel per machine:
#
#      julia plot_profile.jl OUTDIR "Ice Lake=il.profile.tsv" "Zen 4=zen4.profile.tsv"
#
#  writes OUTDIR/profile-MATRIX-small.svg (n <= 512) and
#  OUTDIR/profile-MATRIX-large.svg (n >= 512), MATRIX default dense: for
#  each routine the time it takes longer with ulmBLAS (median per DGETRF),
#  in % of the time of DGETRF with OpenBLAS's BLAS (without the timing
#  wrappers), against n (linear).  The parts add up to the slowdown of
#  DGETRF; below 0 = that part is faster with ulmBLAS.  "rest of DGETRF" is
#  the time outside the BLAS calls (DLASWP and the code of DGETRF and
#  DGETRF2 in liblapack, the same code in both configurations).  Only
#  Julia's standard library is used.
#
const ROUTINES = ["dgemm", "dtrsm", "idamax", "dscal", "lapack"]
const LABELS = Dict("dgemm" => "DGEMM", "dtrsm" => "DTRSM", "idamax" => "IDAMAX",
                    "dscal" => "DSCAL", "lapack" => "rest of DGETRF")
# categorical slots 1 to 5 (light, dark), fixed order
const COLORS      = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
const COLORS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]

outdir = ARGS[1]
const MATRIX = get(ENV, "MATRIX", "dense")
machines = map(ARGS[2:end]) do a
    name, path = split(a, '='; limit=2)
    lines = readlines(path)
    header = split(lines[1], '\t')
    col(c) = findfirst(==(c), header)
    t = Dict{Tuple{String,String,Int},Float64}()
    for l in lines[2:end]
        f = split(l, '\t')
        f[col("matrix")] == MATRIX || continue
        t[(f[col("config")], f[col("routine")], parse(Int, f[col("n")]))] =
            parse(Float64, f[col("median_ns")])
    end
    sizes = sort(unique(last.(keys(t))))
    loss(r, n) = t[("netlib_ulmblas", r, n)] - t[("netlib_openblas", r, n)]
    share = Dict((r, n) => 100*loss(r, n)/t[("netlib_openblas", "plain", n)]
                 for r in ROUTINES for n in sizes)
    (name=String(name), t=t, sizes=sizes, share=share, loss=loss)
end

fmt_time(ns) = abs(ns) < 1e3 ? string(round(Int, ns), " ns") :
               abs(ns) < 1e6 ? string(round(ns/1e3, digits=1), " µs") :
               string(round(ns/1e6, digits=1), " ms")

function figure(part, nmin, nmax)
    W, pw, ph, left, gapx, top = 0, 380, 300, 84, 100, 92
    ms = [(m..., sizes=filter(n -> nmin <= n <= nmax, m.sizes)) for m in machines]
    np = length(ms)
    W = left + np*pw + (np - 1)*gapx + 110
    H = top + ph + 80
    io = IOBuffer()
    p(x...) = print(io, x..., '\n')
    r(v) = round(v, digits=1)
    what = MATRIX == "dense" ? "random matrices" : "matrices diagonally dominant by columns"
    p("""<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" viewBox="0 0 $W $H" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif">""")
    p("<title>Netlib DGETRF: where ulmBLAS loses time against OpenBLAS's BLAS</title>")
    p("<style>")
    p("  .surface { fill: #fcfcfb; } .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; }")
    p("  .grid { stroke: #e3e2de; stroke-width: 1; } .axis, .ref { stroke: #8f8e88; stroke-width: 1; }")
    for i in 1:5
        p("  .s$i { stroke: $(COLORS[i]); fill: none; stroke-width: 2; } .m$i { fill: $(COLORS[i]); stroke: #fcfcfb; stroke-width: 2; } .k$i { fill: $(COLORS[i]); }")
    end
    p("  @media (prefers-color-scheme: dark) {")
    p("    .surface { fill: #1a1a19; } .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; }")
    p("    .grid { stroke: #34332f; } .axis, .ref { stroke: #6b6a65; }")
    for i in 1:5
        p("    .s$i { stroke: $(COLORS_DARK[i]); } .m$i { fill: $(COLORS_DARK[i]); stroke: #1a1a19; } .k$i { fill: $(COLORS_DARK[i]); }")
    end
    p("  }")
    p("</style>")
    p("""<rect class="surface" x="0" y="0" width="$W" height="$H" rx="8"/>""")
    p("""<text class="t1" x="24" y="28" font-size="15" font-weight="600">Netlib DGETRF, $what: $(part == "small" ? "n ≤ 512" : "n ≥ 512"): extra time with ulmBLAS per routine</text>""")
    lx0 = 24
    for (i, rt) in enumerate(ROUTINES)
        p("""<rect class="k$i" x="$lx0" y="44" width="12" height="12" rx="2"/>""")
        p("""<text class="t2" x="$(lx0 + 17)" y="54" font-size="12">$(LABELS[rt])</text>""")
        lx0 += 17 + 7*length(LABELS[rt]) + 22
    end

    allv = [m.share[(rt, n)] for m in ms for rt in ROUTINES for n in m.sizes]
    raw = maximum(allv) - min(0, minimum(allv))
    step = raw > 300 ? 100 : raw > 150 ? 50 : raw > 60 ? 20 : raw > 30 ? 10 : 5
    lo = minimum(allv) > -step/4 ? 0.0 : step*floor(minimum(allv)/step); hi = step*ceil(maximum(allv)/step)
    yticks = collect(step*floor(lo/step):step:step*ceil(hi/step))
    for (k, m) in enumerate(ms)
        x0 = left + (k - 1)*(pw + gapx)
        xmin = nmin; xmax = 512*ceil(Int, last(m.sizes)/512)
        lx(n) = x0 + pw*(n - xmin)/(xmax - xmin)
        ly(v) = top + ph*(1 - (v - first(yticks))/(last(yticks) - first(yticks)))
        p("""<text class="t1" x="$x0" y="$(top - 14)" font-size="13" font-weight="600">$(m.name)</text>""")
        for v in yticks
            p("""<line class="grid" x1="$x0" x2="$(x0 + pw)" y1="$(r(ly(v)))" y2="$(r(ly(v)))"/>""")
            p("""<text class="t2" x="$(x0 - 7)" y="$(r(ly(v)) + 4)" font-size="11" text-anchor="end">$(Int(v))%</text>""")
        end
        p("""<line class="ref" x1="$x0" x2="$(x0 + pw)" y1="$(r(ly(0)))" y2="$(r(ly(0)))"/>""")
        for n in xmin:(xmax - xmin > 2048 ? 512 : xmax - xmin > 600 ? 256 : 128):xmax
            p("""<text class="t2" x="$(r(lx(n)))" y="$(top + ph + 16)" font-size="11" text-anchor="middle">$n</text>""")
        end
        p("""<line class="axis" x1="$x0" x2="$(x0 + pw)" y1="$(top + ph)" y2="$(top + ph)"/>""")
        p("""<text class="t2" x="$(x0 + pw/2)" y="$(top + ph + 34)" font-size="12" text-anchor="middle">n (n × n matrix)</text>""")
        p("""<text class="t2" font-size="12" text-anchor="middle" transform="translate($(x0 - 50) $(top + ph/2)) rotate(-90)">extra time, % of DGETRF with OpenBLAS's BLAS</text>""")
        for (i, rt) in enumerate(ROUTINES)
            pts = [(n, m.share[(rt, n)]) for n in m.sizes]
            p("""<path class="s$i" d="$(join(["$(j == 1 ? 'M' : 'L')$(r(lx(n))) $(r(ly(v)))" for (j, (n, v)) in enumerate(pts)], " "))"/>""")
            for (n, v) in pts
                p("""<circle class="m$i" cx="$(r(lx(n)))" cy="$(r(ly(v)))" r="3.5"><title>$(LABELS[rt]), n = $n: $(fmt_time(m.loss(rt, n))) more per DGETRF ($(round(v, digits=1))% of $(fmt_time(m.t[("netlib_openblas", "plain", n)])))</title></circle>""")
            end
        end
    end
    p("""<text class="t2" x="24" y="$(H - 14)" font-size="11">Time with ulmBLAS minus time with OpenBLAS's BLAS, per routine (median per DGETRF, each call timed); in % of DGETRF with OpenBLAS's BLAS; the parts add up to the slowdown.</text>""")
    p("</svg>")
    file = joinpath(outdir, "profile-$MATRIX-$part.svg")
    write(file, String(take!(io)))
    println("wrote ", file)
end

figure("small", 0, 512)
figure("large", 512, typemax(Int))
