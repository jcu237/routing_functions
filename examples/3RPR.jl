# A planar 3-RPR parallel manipulator.
#
# V(F) is the set of poses (position p, orientation ϕ = (cos θ, sin θ)) together
# with the two leg lengths c₁, c₂ that the third leg (fixed length c3) allows.
# Removing the singular poses -- where the jacobian of F in (p, ϕ) is singular --
# leaves the regions within which the manipulator can move without passing through
# a singularity.
#
# NOTE (to check against the source of these equations): two terms look like typos.
# F[2] should be |p - R(ϕ)(a₃, b₃)|² - c₁ with R(ϕ) = [ϕ₁ -ϕ₂; ϕ₂ ϕ₁], which expands
# to the line below with `-2*(a[3]*p[1] + b[3]*p[2])*ϕ[1]` where it has `...*p[1]`;
# and F[4] should be |p - (A₃, B₃)|² - c3, which has `B[3]*p[2]` where it has
# `B[3]*p[1]`. They are left as they were.
#
# V(F) is large: p and ϕ stay within about 30 of the origin, but the leg lengths c₁,
# c₂ reach 2200 and 760. The starting points of the flows must cover all of it
# (`box = 2000.0`): monodromy explores from the part of V(F) the flows found, and with
# a small box it never reaches the routing points far out.
#
# Expected: 2 components, one for each sign of the determinant, each with χ = 1
# (checked independently: V(F) is a torus -- p on the circle (p₁ - 15)² + p₂² = 208,
# ϕ on the unit circle, c determined by F[2], F[3] -- and a fine grid on it has two
# sign regions, each with three maxima of |r| and two saddles).
#
# The hardest example: most of those 10 routing points are far out, where |r| is below
# 1e-12. With the settings below the 2 components come out in most runs; when a far
# maximum is missed, one of them has χ = 0 instead of 1.

using ConnectedComponents
using LinearAlgebra
using Random

Random.seed!(4)

a = [0, 14, 7]
b = [0, 0, 10]
A = [0, 16, 9]
B = [0, 0, 6]
c3 = 100

@var p[1:2] ϕ[1:2] c[1:2]

F = [
    ϕ[1]^2 + ϕ[2]^2 - 1,
    p[1]^2 + p[2]^2 - 2*(a[3]*p[1] + b[3]*p[2])*p[1] + 2*(b[3]*p[1] - a[3]*p[2])*ϕ[2] + a[3]^2 + b[3]^2 - c[1],
    p[1]^2 + p[2]^2 - 2*A[2]*p[1] + 2*((a[2]-a[3])*p[1] - b[3]*p[2] + A[2]*a[3] - A[2]*a[2])*ϕ[1] + 2*(b[3]*p[1]+(a[2]-a[3])*p[2] - A[2]*b[3])*ϕ[2] + (a[2]-a[3])^2 + b[3]^2 + A[2]^2 - c[2],
    p[1]^2 + p[2]^2 - 2*(A[3]*p[1] + B[3]*p[1]) + A[3]^2 + B[3]^2 - c3
]

vars = vcat(p, ϕ, c)
JF = differentiate(F, vars)
r = RoutingFunction(det(JF[1:4, 1:4]), vars)
cache = RoutingCache(r, F)

# the routing points: flows from 400 starts spread over V(F), then monodromy
pts = routing_points(cache; box = 2000.0, nstarts = 400, max_attempts = 50, verbose = true)
println(length(pts), " routing points, indices ", sort(routing_point_indices(cache, pts)))

# the paths between them. the scale of the problem is large, so are the steps, and
# the saddles are ~2000 units from the maxima they join: allow long paths
C = connected_components(cache, pts; grad_step_size = 1.0, tol = 2.0, start_step_size = 1.1,
                         path_options = (tspan = (0.0, 1e5),))
display(C)
