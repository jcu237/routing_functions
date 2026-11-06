# Surfaces with singular points.
#
# The method needs V(G) to be smooth away from V(f). When V(G) has singular points,
# put them inside V(f). These surfaces use the fallback `jacobian_rank_locus(G, vars)`
# (for a single equation g that is ‖∇g‖²). It removes every singular point, and here
# nothing else, because each g has no repeated factor and changes sign. A polynomial of
# your own is usually smaller: on the ding dong, f = x[3] gives the same result.

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
f = jacobian_rank_locus([g], x[1:3])          # = ‖∇g‖², zero exactly at the origin on V(g)
r = RoutingFunction(f, x[1:3])
C = connected_components(r, [g])
println(sort([c.euler_characteristic for c in C]))     # [0, 1]

# ---------------------------------------------------------------------------
# "Chubs" x⁴ + y⁴ + z⁴ - (x² + y² + z²) + 1/2 = 0   
# ---------------------------------------------------------------------------
# singular at the 12 points (±1/√2, ±1/√2, 0) and permutations. Removing them
# leaves 8 pieces, each a sphere with three points removed (χ = -1).
g = x[1]^4 + x[2]^4 + x[3]^4 - (x[1]^2 + x[2]^2 + x[3]^2) + 1 / 2
r = RoutingFunction(jacobian_rank_locus([g], x[1:3]), x[1:3])
cache = RoutingCache(r, [g])
C = connected_components(cache)
println(length(C), " components with χ = ", [c.euler_characteristic for c in C])
println("routing points by index: ",
        Dict(k => length(v) for (k, v) in sort_routing_points_by_index(cache, reduce(vcat, [c.points for c in C]))))


# ---------------------------------------------------------------------------
# Cayley Cubic x^3 + y^3 + z^3 + 1 - 0.25*(x + y + z + 1)^3
# Removing the 4 singular points leaves 3 connected components.
# ---------------------------------------------------------------------------
g = x[1]^3+x[2]^3+x[3]^3+1-0.25*(x[1]+x[2]+x[3]+1)^3
r = RoutingFunction(jacobian_rank_locus([g], x[1:3]), x[1:3])
cache = RoutingCache(r, [g])
C = connected_components(cache)
println(length(C), " components with χ = ", [c.euler_characteristic for c in C])
println("routing points by index: ",
        Dict(k => length(v) for (k, v) in sort_routing_points_by_index(cache, reduce(vcat, [c.points for c in C]))))


# ---------------------------------------------------------------------------
# Barth Sextic (40 components? must check)
# ---------------------------------------------------------------------------
g = 4*(((1+sqrt(5))/2)^2*x[1]^2-x[2]^2)*(((1+sqrt(5))/2)^2*x[2]^2-x[3]^2)*(((1+sqrt(5))/2)^2*x[3]^2-x[1]^2)-1*(1+2*((1+sqrt(5))/2))*(x[1]^2+x[2]^2+x[3]^2-1)^2
r = RoutingFunction(jacobian_rank_locus([g], x[1:3]), x[1:3])
cache = RoutingCache(r, [g])
C = connected_components(cache)
println(length(C), " components with χ = ", [c.euler_characteristic for c in C])
println("routing points by index: ",
        Dict(k => length(v) for (k, v) in sort_routing_points_by_index(cache, reduce(vcat, [c.points for c in C]))))


# ---------------------------------------------------------------------------
# Togliatti-quintic (13 components? must check)
# ---------------------------------------------------------------------------
g = (1-sqrt(5-sqrt(5))/2*x[3])*(x[1]^2+x[2]^2-1+(1+3*sqrt(5))/4*x[3]^2)^2-3.8496061120482923113983843766*(x[1]-x[3])*(cos(2*3.14159265358979/5)*x[1]-sin(2*3.14159265358979/5)*x[2]-x[3])*(cos(4*3.14159265358979/5)*x[1]-sin(4*3.14159265358979/5)*x[2]-x[3])*(cos(6*3.14159265358979/5)*x[1]-sin(6*3.14159265358979/5)*x[2]-x[3])*(cos(8*3.14159265358979/5)*x[1]-sin(8*3.14159265358979/5)*x[2]-x[3]) 
r = RoutingFunction(jacobian_rank_locus([g], x[1:3]), x[1:3])
cache = RoutingCache(r, [g])
C = connected_components(cache)
println(length(C), " components with χ = ", [c.euler_characteristic for c in C])
println("routing points by index: ",
        Dict(k => length(v) for (k, v) in sort_routing_points_by_index(cache, reduce(vcat, [c.points for c in C]))))

