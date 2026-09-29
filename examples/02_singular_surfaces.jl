# Surfaces with singular points.
#
# The method needs V(G) to be smooth away from V(f). When V(G) has singular points,
# put them inside V(f): multiplying f by `singular_locus(G, vars)` (for a single
# equation g that is ‖∇g‖²) removes every singular point along with the rest of V(f).

using ConnectedComponents
using Random

Random.seed!(2)
@var x[1:3]

# ---------------------------------------------------------------------------
# the "ding dong" x² + y² = z² - z³
# ---------------------------------------------------------------------------
# a cone point at the origin joins a bell (0 < z < 1, a disc) to an infinite
# funnel (z < 0, a cylinder). Removing the cone point separates them.
g = x[1]^2 + x[2]^2 - x[3]^2 + x[3]^3
f = singular_locus([g], x[1:3])          # = ‖∇g‖², zero exactly at the origin on V(g)
r = RoutingFunction(f, x[1:3])
C = connected_components(r, [g])
println(sort([c.euler_characteristic for c in C]))     # [0, 1]

# ---------------------------------------------------------------------------
# "Chubs" x⁴ + y⁴ + z⁴ - (x² + y² + z²) + 1/2 = 0   (section 6 of the paper)
# ---------------------------------------------------------------------------
# singular at the 12 points (±1/√2, ±1/√2, 0) and permutations. Removing them
# leaves 8 pieces, each a sphere with three points removed (χ = -1). There are 60 to
# 80 routing points -- how many depends on the centre c, the Euler characteristics do
# not -- and it takes about a minute.
g = x[1]^4 + x[2]^4 + x[3]^4 - (x[1]^2 + x[2]^2 + x[3]^2) + 1 / 2
r = RoutingFunction(singular_locus([g], x[1:3]), x[1:3])
cache = RoutingCache(r, [g])
C = connected_components(cache)
println(length(C), " components with χ = ", [c.euler_characteristic for c in C])
println("routing points by index: ",
        Dict(k => length(v) for (k, v) in sort_routing_points_by_index(cache, reduce(vcat, [c.points for c in C]))))
