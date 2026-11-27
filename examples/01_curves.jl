# Plane and space curves: a first walk through the package.
#
# Each block computes the connected components of V(G) ∖ V(f) for a curve V(G)
# and a numerator f, and checks the answer against what can be seen by hand.
# Every block runs in seconds (after the first call has compiled everything).

using ConnectedComponents
using Random

Random.seed!(1)   # the centre c and the starting points are random

# ---------------------------------------------------------------------------
# two concentric circles, nothing removed
# ---------------------------------------------------------------------------
@var x[1:3]
G = [(x[1]^2 + x[2]^2 - 1) * (x[1]^2 + x[2]^2 - 9)]
r = RoutingFunction(1, x[1:2])         # f = 1 removes nothing; pass the variables
C = connected_components(r, G)
# 2 components (the two circles), each with χ = 0
display(C)

# ---------------------------------------------------------------------------
# the same, step by step
# ---------------------------------------------------------------------------
# build the cache once and hand it to every routine: it holds the compiled
# systems, so `(r, G)` arguments rebuild all of it on every call.
cache = RoutingCache(r, G)

# 1. routing points: the critical points of r on V(G) ∖ V(f)
pts = routing_points(cache)

# 2. their Morse indices. index 0 = a local maximum of |r|; on a circle every
#    maximum comes with a minimum, of index 1.
idxs = routing_point_indices(cache, pts)
println(length(pts), " routing points with indices ", idxs)

# 3. connect: from every point of positive index, follow the gradient flow of |r|
#    up to the maxima it reaches. A is the reachability matrix.
A, pts = find_connectivity_matrix(cache, pts)

# 4. group into components
C = connected_components(cache, pts, A)
for c in C
    println(c)                  # the points, their indices and χ = ∑ (-1)^index
end
println("χ of the whole curve: ", euler_characteristic(C))

# ---------------------------------------------------------------------------
# an elliptic curve: an oval and an unbounded branch
# ---------------------------------------------------------------------------
E = [x[2]^2 - x[1] * (x[1] - 1) * (x[1] + 1)]
C = connected_components(RoutingFunction(1, x[1:2]), E)
println(sort([c.euler_characteristic for c in C]))     # [0, 1]: a circle and a line

# remove the two points where it meets the circle of radius 3: the unbounded branch
# is cut into three arcs, the oval is untouched
C = connected_components(RoutingFunction(x[1]^2 + x[2]^2 - 9, x[1:2]), E)
println(sort([c.euler_characteristic for c in C]))     # [0, 1, 1, 1]

# ---------------------------------------------------------------------------
# the twisted cubic with the origin removed
# ---------------------------------------------------------------------------
# a curve in ℝ³ cut out by two equations. f = xyz vanishes on it only at the
# origin, which splits it in two.
T = [x[1]^3 - x[3], x[1]^2 - x[2]]
C = connected_components(RoutingFunction(x[1] * x[2] * x[3], x[1:3]), T)
println(sort([c.euler_characteristic for c in C]))     # [1, 1]

# ---------------------------------------------------------------------------
# a quartic curve with the coordinate axes removed
# ---------------------------------------------------------------------------
# the curve of Figure 1 of "Smooth Connectivity in Real Algebraic Varieties". It
# is singular at the origin, which lies on the axes, so removing the axes removes
# the singular point too.
Q = [x[1]^4 + x[2]^4 - (x[1] - x[2])^2 * (x[1] + x[2])]
C = connected_components(RoutingFunction(x[1] * x[2], x[1:2]), Q)
println(sort([c.euler_characteristic for c in C]))     # [1, 1, 1, 1]: four arcs


# ---------------------------------------------------------------------------
# the nodal cubic with the origin removed, plotted by component
# ---------------------------------------------------------------------------
# 4y² = x²(x + 4) has a node at the origin. Removing it cuts the curve into three
# arcs: the loop through (-4, 0) and the two unbounded branches.
# Plotting needs CairoMakie in a reachable environment (e.g. the global one).
# `save` is qualified: HomotopyContinuation exports one too.
using CairoMakie

N = [4 * x[2]^2 - x[1]^2 * (x[1] + 4)]
r = RoutingFunction(x[1]^2 + x[2]^2, x[1:2])   # f vanishes only at the origin
# (not f = xy: it also vanishes at (-4, 0), which would cut the loop in two)
cache = RoutingCache(r, N)
C = connected_components(cache)
println(sort([c.euler_characteristic for c in C]))     # [1, 1, 1]: three arcs

# the curve is parametrised by t ↦ (t² - 4, t(t² - 4)/2), with the origin at t = ±2,
# so the arcs are t < -2, -2 < t < 2 and t > 2. Ask the package which component
# each arc lies on, from a point a third of the way along it (not the middle of
# the loop: (-4, 0) is a routing point, where the flow cannot start).
nodal(t) = [t^2 - 4, t * (t^2 - 4) / 2]
arcs = [range(-2.9, -2.0, 400), range(-2.0, 2.0, 800), range(2.0, 2.9, 400)]
labels = [component_index(cache, C, nodal(a[end ÷ 3])) for a in arcs]
println("component of each arc: ", labels)

colours = [colorant"#0072B2", colorant"#009E73", colorant"#CC79A7"]  # never orange
linewidth = 9

fig = Figure(size = (900, 1000), backgroundcolor = :white)
ax = Axis(fig[1, 1], aspect = DataAspect())
hidedecorations!(ax)
hidespines!(ax)
for (a, k) in zip(arcs, labels)
    P = nodal.(a)
    lines!(ax, first.(P), last.(P); color = colours[k], linewidth = linewidth)
end
scatter!(ax, [0.0], [0.0]; color = colorant"#E69F00", markersize = 4* linewidth,
         strokewidth = 0)                    # the removed origin
CairoMakie.save(joinpath(@__DIR__, "nodal_cubic.png"), fig; px_per_unit = 4)
