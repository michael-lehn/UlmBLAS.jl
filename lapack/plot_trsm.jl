#
#  DGETRF of Netlib LAPACK with ulmBLAS, with ulmBLAS but OpenBLAS's DTRSM,
#  and with OpenBLAS's BLAS, from the results of bench.jl
#  (CONFIGS=netlib_ulmblas,netlib_ulmblas_obtrsm,netlib_openblas), one
#  panel per machine:
#
#      julia plot_trsm.jl OUTDIR "Ice Lake=il.tsv" "Zen 4=zen4.tsv"
#
#  writes OUTDIR/trsm-MATRIX.svg for MATRIX = dense, coldom: time of
#  Netlib LAPACK + OpenBLAS's BLAS / time (medians), against n (linear);
#  below 1 = slower than with OpenBLAS's BLAS.  Only Julia's standard
#  library is used.
#
const SERIES = ["netlib_ulmblas", "netlib_ulmblas_obtrsm"]
const LABELS = Dict("netlib_ulmblas" => "ulmBLAS",
                    "netlib_ulmblas_obtrsm" => "ulmBLAS with OpenBLAS's DTRSM")
const BASE = "netlib_openblas"
# categorical slots 2 and 1 (ulmBLAS keeps the color of the other plots)
const COLORS      = ["#eb6834", "#2a78d6"]
const COLORS_DARK = ["#d95926", "#3987e5"]

outdir = ARGS[1]
machines = map(ARGS[2:end]) do a
    name, path = split(a, '='; limit=2)
    lines = readlines(path)
    header = split(lines[1], '\t')
    col(c) = findfirst(==(c), header)
    t = Dict{Tuple{String,String,Int},Float64}()
    for l in lines[2:end]
        f = split(l, '\t')
        t[(f[col("matrix")], f[col("config")], parse(Int, f[col("n")]))] =
            parse(Float64, f[col("median_ns")])
    end
    (name=String(name), t=t, sizes=sort(unique(last.(keys(t)))))
end

fmt_time(ns) = ns < 1e3 ? string(round(Int, ns), " ns") :
               ns < 1e6 ? string(round(ns/1e3, digits=1), " µs") :
               ns < 1e9 ? string(round(ns/1e6, digits=1), " ms") :
               string(round(ns/1e9, digits=2), " s")

function figure(kind)
    pw, ph, left, gapx, top = 380, 300, 84, 100, 92
    np = length(machines)
    W = left + np*pw + (np - 1)*gapx + 24
    H = top + ph + 80
    io = IOBuffer()
    p(x...) = print(io, x..., '\n')
    r(v) = round(v, digits=1)
    what = kind == "dense" ? "random matrices" : "column-dominant matrices (no row interchanges)"
    sp(m, c, n) = m.t[(kind, BASE, n)]/m.t[(kind, c, n)]
    p("""<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" viewBox="0 0 $W $H" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif">""")
    p("<title>Netlib DGETRF with OpenBLAS's DTRSM in ulmBLAS</title>")
    p("<style>")
    p("  .surface { fill: #fcfcfb; } .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; }")
    p("  .grid { stroke: #e3e2de; stroke-width: 1; } .axis { stroke: #8f8e88; stroke-width: 1; }")
    p("  .ref { stroke: #8f8e88; stroke-width: 1.5; stroke-dasharray: 5 4; }")
    for i in 1:2
        p("  .s$i { stroke: $(COLORS[i]); fill: none; stroke-width: 2; } .m$i { fill: $(COLORS[i]); stroke: #fcfcfb; stroke-width: 2; } .k$i { fill: $(COLORS[i]); }")
    end
    p("  @media (prefers-color-scheme: dark) {")
    p("    .surface { fill: #1a1a19; } .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; }")
    p("    .grid { stroke: #34332f; } .axis, .ref { stroke: #6b6a65; }")
    for i in 1:2
        p("    .s$i { stroke: $(COLORS_DARK[i]); } .m$i { fill: $(COLORS_DARK[i]); stroke: #1a1a19; } .k$i { fill: $(COLORS_DARK[i]); }")
    end
    p("  }")
    p("</style>")
    p("""<rect class="surface" x="0" y="0" width="$W" height="$H" rx="8"/>""")
    p("""<text class="t1" x="24" y="28" font-size="15" font-weight="600">Netlib DGETRF, $what: speedup over Netlib + OpenBLAS's BLAS</text>""")
    lx0 = 24
    for (i, c) in enumerate(SERIES)
        p("""<rect class="k$i" x="$lx0" y="44" width="12" height="12" rx="2"/>""")
        p("""<text class="t2" x="$(lx0 + 17)" y="54" font-size="12">Netlib LAPACK + $(LABELS[c])</text>""")
        lx0 += 17 + 6.4*length("Netlib LAPACK + " * LABELS[c]) + 24
    end
    v = [sp(m, c, n) for m in machines for c in SERIES for n in m.sizes]
    hi = max(maximum(v), 1.0)
    step = hi > 1.2 ? 0.25 : 0.1
    yticks = collect(0.0:step:step*ceil(hi/step))
    for (j, m) in enumerate(machines)
        x0 = left + (j - 1)*(pw + gapx)
        xmax = 512*ceil(Int, last(m.sizes)/512)
        lx(n) = x0 + pw*n/xmax
        ly(y) = top + ph*(1 - y/last(yticks))
        p("""<text class="t1" x="$x0" y="$(top - 14)" font-size="13" font-weight="600">$(m.name)</text>""")
        for y in yticks
            p("""<line class="grid" x1="$x0" x2="$(x0 + pw)" y1="$(r(ly(y)))" y2="$(r(ly(y)))"/>""")
            p("""<text class="t2" x="$(x0 - 7)" y="$(r(ly(y)) + 4)" font-size="11" text-anchor="end">$(round(y, digits=2))×</text>""")
        end
        p("""<line class="ref" x1="$x0" x2="$(x0 + pw)" y1="$(r(ly(1)))" y2="$(r(ly(1)))"/>""")
        for n in 0:(xmax > 2048 ? 1024 : 512):xmax
            p("""<text class="t2" x="$(r(lx(n)))" y="$(top + ph + 16)" font-size="11" text-anchor="middle">$n</text>""")
        end
        p("""<line class="axis" x1="$x0" x2="$(x0 + pw)" y1="$(top + ph)" y2="$(top + ph)"/>""")
        p("""<text class="t2" x="$(x0 + pw/2)" y="$(top + ph + 34)" font-size="12" text-anchor="middle">n (n × n matrix)</text>""")
        p("""<text class="t2" font-size="12" text-anchor="middle" transform="translate($(x0 - 58) $(top + ph/2)) rotate(-90)">time with OpenBLAS's BLAS / time</text>""")
        for (i, c) in enumerate(SERIES)
            pts = [(n, sp(m, c, n)) for n in m.sizes]
            p("""<path class="s$i" d="$(join(["$(q == 1 ? 'M' : 'L')$(r(lx(n))) $(r(ly(y)))" for (q, (n, y)) in enumerate(pts)], " "))"/>""")
            for (n, y) in pts
                p("""<circle class="m$i" cx="$(r(lx(n)))" cy="$(r(ly(y)))" r="3.5"><title>Netlib + $(LABELS[c]), n = $n: $(round(y, digits=2))× ($(fmt_time(m.t[(kind, c, n)])) vs. $(fmt_time(m.t[(kind, BASE, n)])))</title></circle>""")
            end
        end
    end
    p("""<text class="t2" x="24" y="$(H - 14)" font-size="11">Complete DGETRF calls, median, one thread; the same Netlib LAPACK 3.12.1 in all configurations; 1 = Netlib LAPACK + OpenBLAS's BLAS.</text>""")
    p("</svg>")
    file = joinpath(outdir, "trsm-$kind.svg")
    write(file, String(take!(io)))
    println("wrote ", file)
end

foreach(figure, ("dense", "coldom"))
