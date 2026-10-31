#
#  Bar chart (SVG) of the results of gemm_threads.jl:
#
#      julia plot_gemm_threads.jl results/2026-10-31-apple-m5.tsv [SVG [TITLE]]
#
#  writes results/2026-10-31-apple-m5.svg: 8 groups (BLAS threads 1 to 4 x
#  n = 1000, 1500) of 4 bars, GFLOPS = 2n^3 / minimum time (linear axis
#  from zero), as in the output of gemm_threads.jl.
#  Only Julia's standard library is used.
#

const CASES = ["ulmBLAS.mul!, strided", "LinearAlgebra.mul!, strided",
               "LinearAlgebra.mul!, contiguous", "ulmBLAS.mul!, contiguous"]
# categorical slots 1 to 4 (light, dark), fixed order
const COLORS      = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
const COLORS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]

tsv = length(ARGS) >= 1 ? ARGS[1] :
      joinpath(@__DIR__, "results", "2026-10-31-apple-m5.tsv")
svg = length(ARGS) >= 2 ? ARGS[2] : replace(tsv, r"\.tsv$" => ".svg")
label = length(ARGS) >= 3 ? ARGS[3] : "DGEMM on an Apple M5"

#  GFLOPS by (case, n, BLAS threads)
gf = Dict{Tuple{String,Int,Int},Float64}()
lines = readlines(tsv)
header = split(lines[1], '\t')
col(name) = findfirst(==(name), header)
for l in lines[2:end]
    f = split(l, '\t')
    n = parse(Int, f[col("n")])
    gf[(f[col("case")], n, parse(Int, f[col("BLAS_threads")]))] =
        2.0*n^3/parse(Float64, f[col("min_ns")])
end
groups = [(t, n) for t in 1:4 for n in (1000, 1500)]

#  geometry
W, H = 900, 470
left, right, top, bottom = 70, 20, 78, 92
pw, ph = W - left - right, H - top - bottom
ymax = 100*ceil(maximum(values(gf))/100)    # axis range in GFLOPS (linear)
y(v) = top + ph*(1 - v/ymax)
gw = pw/length(groups)                       # width per group
bw, gap = 16, 2                              # bar width, gap between bars

io = IOBuffer()
p(s...) = print(io, s..., '\n')
p("""<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" viewBox="0 0 $W $H" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif">""")
p("<title>GEMM on an Apple M5: GFLOPS, higher is better</title>")
p("<style>")
p("  .surface { fill: #fcfcfb; } .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; }")
p("  .grid { stroke: #e3e2de; stroke-width: 1; } .axis { stroke: #8f8e88; stroke-width: 1; }")
for i in 1:4
    p("  .s$i { fill: $(COLORS[i]); }")
end
p("  @media (prefers-color-scheme: dark) {")
p("    .surface { fill: #1a1a19; } .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; }")
p("    .grid { stroke: #34332f; } .axis { stroke: #6b6a65; }")
for i in 1:4
    p("    .s$i { fill: $(COLORS_DARK[i]); }")
end
p("  }")
p("</style>")
p("""<rect class="surface" x="0" y="0" width="$W" height="$H" rx="8"/>""")

#  title and legend
p("""<text class="t1" x="$left" y="26" font-size="15" font-weight="600">$label: GFLOPS (higher is better)</text>""")
lx = left
for (i, c) in enumerate(CASES)
    p("""<rect class="s$i" x="$(round(lx, digits=1))" y="42" width="12" height="12" rx="2"/>""")
    p("""<text class="t2" x="$(round(lx + 17, digits=1))" y="52" font-size="12">$c</text>""")
    global lx += 17 + 6.6*length(c) + 26
end

#  grid and y axis
for d in 0:100:Int(ymax)
    yy = round(y(d), digits=1)
    p("""<line class="grid" x1="$left" x2="$(W - right)" y1="$yy" y2="$yy"/>""")
    p("""<text class="t2" x="$(left - 8)" y="$(yy + 4)" font-size="11" text-anchor="end">$d</text>""")
end
p("""<text class="t2" font-size="12" text-anchor="middle" transform="translate(22 $(top + ph/2)) rotate(-90)">GFLOPS</text>""")

#  bars: rectangles from zero up, 4px rounded top
for (g, (t, n)) in enumerate(groups)
    x0 = left + (g - 1)*gw + (gw - (4*bw + 3*gap))/2
    for (i, c) in enumerate(CASES)
        v  = gf[(c, n, t)]
        x  = x0 + (i - 1)*(bw + gap)
        yt = y(v); yb = top + ph; r = 4
        p("""<path class="s$i" d="M$(x) $(yb) V$(round(yt + r, digits=1)) Q$(x) $(round(yt, digits=1)) $(x + r) $(round(yt, digits=1)) H$(x + bw - r) Q$(x + bw) $(round(yt, digits=1)) $(x + bw) $(round(yt + r, digits=1)) V$(yb) Z"><title>$c, $n × $n, $t BLAS thread$(t > 1 ? "s" : ""): $(round(v, digits=1)) GFLOPS</title></path>""")
    end
    cx = left + (g - 0.5)*gw
    p("""<text class="t1" x="$cx" y="$(top + ph + 18)" font-size="12" text-anchor="middle">$t Thread$(t > 1 ? "s" : "")</text>""")
    p("""<text class="t2" x="$cx" y="$(top + ph + 33)" font-size="12" text-anchor="middle">$n × $n</text>""")
end
p("""<line class="axis" x1="$left" x2="$(W - right)" y1="$(top + ph)" y2="$(top + ph)"/>""")
p("""<text class="t2" x="$left" y="$(H - 18)" font-size="11">Threads: OpenBLAS threads (BLAS.set_num_threads). ulmBLAS and Julia's generic strided multiplication always use one thread.</text>""")
p("</svg>")

write(svg, String(take!(io)))
println("wrote ", svg)
