# Positive Landau discriminants: the parachute diagram with equal masses.
#
# α = (α1 : α2 : α3 : α4) are the Schwinger parameters and (s, m, M) the kinematic
# parameters (the parachute's F does not involve t). The incidence variety Y of the
# Landau equations projects onto the Landau discriminant. The question here is which
# irreducible components of Y contain a point with α real and positive and with a
# positive real mass ratio M/m. We work in the charts α4 = 1 and m = 1, so this means
# α1, α2, α3 > 0 and M > 0.

using ConnectedComponents
using LinearAlgebra
using Random

@var α[1:3] y s M
α1, α2, α3 = α
p = [s, M]              # kinematic parameters in the chart m = 1
vars = [α; y; p]

# The Landau equations ∂F/∂αₑ = 0 (e = 1, …, 4) in the charts α4 = 1, m = 1, computed
# with Landau.jl. The last equation, with U the first Symanzik polynomial at α4 = 1,
# forces α1α2α3·U ≠ 0. That removes the coordinate hyperplanes and the spurious
# components in U = 0.
U = (α1 + α2)*(α3 + 1) + α3
G = System([
    M*α3 - 2*α1*α3 - 2*α1 + α2*α3*s - 2*α2*α3 + α2*s - 2*α2 - α3^2 - 3*α3 - 1,
    M*α3 + α1*α3*s - 2*α1*α3 + α1*s - 2*α1 - 2*α2*α3 - 2*α2 - α3^2 - 3*α3 - 1,
    M*α1 + M*α2 - α1^2 + α1*α2*s - 2*α1*α2 - 2*α1*α3 - 3*α1 - α2^2 - 2*α2*α3 - 3*α2 - 2*α3 - 1,
    M*α1*α3 + M*α2*α3 - α1^2 + α1*α2*s - 2*α1*α2 - 3*α1*α3 - 2*α1 - α2^2 - 3*α2*α3 - 2*α2 - α3^2 - 2*α3,
    y*α1*α2*α3*U - 1,
], vars)

# Routing function r = f/g^d. Every connected component of V(G) \ V(f) contains a
# routing point, and the factors α1α2α3·M in f keep the signs of α and M constant on
# each one. So every positive piece of V(G) gets a routing point.
#
# ConnectedComponents also needs V(G) to be smooth of dimension 1 off V(f), so f must
# vanish wherever the jacobian of G drops rank on V(G). positive_landau.m2 computes
# that locus:
#   * the surface M = 1, α3 = -1, which is a whole component of V(G) of dimension 2;
#   * the point where the line M = 9 crosses the quadric component (the curve over the
#     quadric M^2 - 10M + 9 + 4s = 0 of the discriminant), at α = (1/2, 1/2, 1);
#   * the point where the line M = 1 crosses the quadric component, at α = (-1/2, -1/2, 1).
# M - 1 vanishes on the surface and the second crossing. (α1 - α2)^2 + (M - 9)^2
# vanishes on the real points of V(G) only at the first crossing, and as a sum of
# squares it does not change any signs.
f =M * (M - 1) * ((α1 - α2)^2 + (M - 9)^2)
r = RoutingFunction(f, vars)
cache = RoutingCache(r, G)

Random.seed!(1)
rp = routing_points(cache; nstarts = 200, box = 10.0)

# keep the routing points with α > 0 and M > 0
is_positive(q) = all(>(0), q[[1, 2, 3, 6]])     # α1, α2, α3, M
positive_rp = filter(is_positive, rp)

println(length(positive_rp), " of ", length(rp), " routing points are positive")
for (j, q) in enumerate(positive_rp)
    println("   $j:  α = ", round.([q[1:3]; 1], sigdigits = 6),
            ",  s = ", round(q[5], sigdigits = 6), ",  M = ", round(q[6], sigdigits = 6))
end

# Which irreducible components of V(G) do the positive routing points lie on?
# witness_sets maps each dimension d to the witness sets of the components of that
# dimension, and membership tests all of positive_rp against one witness set at once.
N = nid(G)
for (d, Ws) in witness_sets(N), (i, W) in enumerate(Ws)
    in_W = membership(positive_rp, W; show_progress = false)
    for j in findall(in_W)
        println("positive routing point $j lies on component $i of dimension $d (degree $(degree(W)))")
    end
end

# V(G) has four components: the surface (dim 2, degree 7) and three curves of degrees 3, 3
# and 4. The 4 positive routing points lie on two of them, two on each:
#   * the line M = 9, s = 0, α3 = 1, α1 + α2 = 1 (degree 3 here because of y), one point on
#     each side of the crossing at α1 = 1/2;
#   * the quadric component α1 = α2 = 2/(M - 5), α3 = 1 (degree 4), one point on each side
#     of the crossing at M = 9.
# positive_landau.m2 lists the four components exactly.
