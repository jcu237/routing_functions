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
