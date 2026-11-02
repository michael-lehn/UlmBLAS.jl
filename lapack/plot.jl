#
#  Line charts (SVG) of the results of bench.jl, one panel per machine:
#
#      julia plot.jl OUTDIR "Ice Lake=results/il.tsv" "Zen 4=results/zen4.tsv" ...
#
#  writes OUTDIR/getrf-time.svg (median time of a complete DGETRF, log-log)
#  and OUTDIR/getrf-speedup.svg (time of OpenBLAS / time, log n axis,
#  above 1 = faster than Julia's default).  Only Julia's standard library
#  is used.
#
const CONFIGS = ["openblas", "netlib_ulmblas", "netlib_openblas"]
const LABELS = Dict("openblas" => "OpenBLAS (Julia's default LAPACK and BLAS)",
                    "netlib_ulmblas" => "Netlib LAPACK + ulmBLAS",
                    "netlib_openblas" => "Netlib LAPACK + OpenBLAS BLAS")
const SHORT = Dict("openblas" => "OpenBLAS", "netlib_ulmblas" => "Netlib + ulmBLAS",
                   "netlib_openblas" => "Netlib + OpenBLAS BLAS")
# categorical slots 1 to 3 (light, dark), fixed order
const COLORS      = ["#2a78d6", "#eb6834", "#1baf7a"]
const COLORS_DARK = ["#3987e5", "#d95926", "#199e70"]

outdir = ARGS[1]
machines = map(ARGS[2:end]) do a
    name, path = split(a, '='; limit=2)
    lines = readlines(path)
    header = split(lines[1], '\t')
    col(c) = findfirst(==(c), header)
    t = Dict{Tuple{String,Int},Float64}()
    for l in lines[2:end]
        f = split(l, '\t')
        t[(f[col("config")], parse(Int, f[col("n")]))] = parse(Float64, f[col("median_ns")])
    end
    sizes = sort(unique(last.(keys(t))))
    (name=String(name), t=t, sizes=sizes)
end

fmt(v) = isinteger(v) || v >= 100 ? string(round(Int, v)) : v >= 10 ? string(round(v, digits=1)) :
         string(round(v, digits=2))
function fmt_time(ns)
    ns < 1e3 && return fmt(ns) * " ns"
    ns < 1e6 && return fmt(ns/1e3) * " µs"
    ns < 1e9 && return fmt(ns/1e6) * " ms"
    return fmt(ns/1e9) * " s"
end

function header(io, W, H, title)
    p(s...) = print(io, s..., '\n')
    p("""<svg xmlns="http://www.w3.org/2000/svg" width="$W" height="$H" viewBox="0 0 $W $H" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif">""")
    p("<title>$title</title>")
    p("<style>")
    p("  .surface { fill: #fcfcfb; } .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; }")
    p("  .grid { stroke: #e3e2de; stroke-width: 1; } .axis { stroke: #8f8e88; stroke-width: 1; }")
    p("  .ref { stroke: #8f8e88; stroke-width: 1.5; stroke-dasharray: 5 4; }")
    for i in 1:3
        p("  .s$i { stroke: $(COLORS[i]); fill: none; stroke-width: 2; } .m$i { fill: $(COLORS[i]); stroke: #fcfcfb; stroke-width: 2; } .k$i { fill: $(COLORS[i]); }")
    end
    p("  @media (prefers-color-scheme: dark) {")
    p("    .surface { fill: #1a1a19; } .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; }")
    p("    .grid { stroke: #34332f; } .axis, .ref { stroke: #6b6a65; }")
    for i in 1:3
        p("    .s$i { stroke: $(COLORS_DARK[i]); } .m$i { fill: $(COLORS_DARK[i]); stroke: #1a1a19; } .k$i { fill: $(COLORS_DARK[i]); }")
    end
    p("  }")
    p("</style>")
    p("""<rect class="surface" x="0" y="0" width="$W" height="$H" rx="8"/>""")
end

function legend(io, x, y, configs)
    for c in configs
        i = findfirst(==(c), CONFIGS)
        print(io, """<rect class="k$i" x="$x" y="$(y - 10)" width="12" height="12" rx="2"/>""", '\n')
        print(io, """<text class="t2" x="$(x + 17)" y="$y" font-size="12">$(LABELS[c])</text>""", '\n')
        x += 17 + 6.4*length(LABELS[c]) + 24
    end
end

#  one panel: x = n (log), y = value (log or linear) for each configuration
function panel(io, m, x0, y0, pw, ph, configs, value, ylog, yticks, ytick_label, tip; ref=nothing)
    p(s...) = print(io, s..., '\n')
    xs = m.sizes
    lx(n) = x0 + pw*(log(n) - log(first(xs)))/(log(last(xs)) - log(first(xs)))
    ylo, yhi = first(yticks), last(yticks)
    f = ylog ? log : identity
    ly(v) = y0 + ph*(1 - (f(v) - f(ylo))/(f(yhi) - f(ylo)))
    r(v) = round(v, digits=1)
    p("""<text class="t1" x="$x0" y="$(y0 - 14)" font-size="13" font-weight="600">$(m.name)</text>""")
    for v in yticks
        p("""<line class="grid" x1="$x0" x2="$(x0 + pw)" y1="$(r(ly(v)))" y2="$(r(ly(v)))"/>""")
        p("""<text class="t2" x="$(x0 - 7)" y="$(r(ly(v)) + 4)" font-size="11" text-anchor="end">$(ytick_label(v))</text>""")
    end
    for n in xs
        (n in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)) || continue
        p("""<text class="t2" x="$(r(lx(n)))" y="$(y0 + ph + 16)" font-size="11" text-anchor="middle">$n</text>""")
    end
    p("""<line class="axis" x1="$x0" x2="$(x0 + pw)" y1="$(y0 + ph)" y2="$(y0 + ph)"/>""")
    p("""<text class="t2" x="$(x0 + pw/2)" y="$(y0 + ph + 34)" font-size="12" text-anchor="middle">n (n × n matrix)</text>""")
    ref === nothing || p("""<line class="ref" x1="$x0" x2="$(x0 + pw)" y1="$(r(ly(ref)))" y2="$(r(ly(ref)))"/>""")
    for c in configs
        i = findfirst(==(c), CONFIGS)
        pts = [(n, value(m, c, n)) for n in xs if haskey(m.t, (c, n))]
        d = join(["$(k == 1 ? 'M' : 'L')$(r(lx(n))) $(r(ly(v)))" for (k, (n, v)) in enumerate(pts)], " ")
        p("""<path class="s$i" d="$d"/>""")
        for (n, v) in pts
            p("""<circle class="m$i" cx="$(r(lx(n)))" cy="$(r(ly(v)))" r="4"><title>$(SHORT[c]), n = $n: $(tip(m, c, n))</title></circle>""")
        end
    end
end

ticks_log(lo, hi) = [10.0^k for k in floor(Int, log10(lo)):ceil(Int, log10(hi))]

function figure(file, title, configs, value, ylog, yticks, ylab, ytick_label, tip, note; ref=nothing)
    np = length(machines)
    pw, ph = 380, 300
    left, gapx, top = 84, 100, 92
    W = left + np*pw + (np - 1)*gapx + 24
    H = top + ph + 80
    io = IOBuffer()
    header(io, W, H, title)
    print(io, """<text class="t1" x="24" y="28" font-size="15" font-weight="600">$title</text>""", '\n')
    legend(io, 24, 54, configs)
    for (k, m) in enumerate(machines)
        x0 = left + (k - 1)*(pw + gapx)
        panel(io, m, x0, top, pw, ph, configs, value, ylog, yticks, ytick_label, tip; ref)
        print(io, """<text class="t2" font-size="12" text-anchor="middle" transform="translate($(x0 - 66) $(top + ph/2)) rotate(-90)">$ylab</text>""", '\n')
    end
    print(io, """<text class="t2" x="24" y="$(H - 14)" font-size="11">$note</text>""", '\n')
    print(io, "</svg>\n")
    write(joinpath(outdir, file), String(take!(io)))
    println("wrote ", joinpath(outdir, file))
end

note = "Median of repeated calls, one thread, same matrices (randn) and Julia version for all configurations."

alltimes = [v for m in machines for v in values(m.t)]
figure("getrf-time.svg", "DGETRF (LU with partial pivoting): median time, lower is better",
       CONFIGS, (m, c, n) -> m.t[(c, n)], true,
       ticks_log(minimum(alltimes), maximum(alltimes)), "time per call",
       fmt_time, (m, c, n) -> fmt_time(m.t[(c, n)]), note)

speedup(m, c, n) = m.t[("openblas", n)]/m.t[(c, n)]
s = [speedup(m, c, n) for m in machines for c in CONFIGS[2:3] for n in m.sizes if haskey(m.t, (c, n))]
step = maximum(s) > 2 ? 0.5 : 0.25
yt = collect(0.0:step:step*ceil(maximum(s)/step))
figure("getrf-speedup.svg", "DGETRF: speedup over Julia's default (OpenBLAS), higher is better",
       CONFIGS[2:3], speedup, false, yt, "time of OpenBLAS / time",
       v -> string(v, "×"), (m, c, n) -> string(round(speedup(m, c, n), digits=2), "× (",
       fmt_time(m.t[(c, n)]), " vs. ", fmt_time(m.t[("openblas", n)]), ")"), note; ref=1.0)
