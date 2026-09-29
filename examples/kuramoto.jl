# The Kuramoto model with three oscillators.
#
# Steady states of three coupled oscillators, with the phase of the third fixed at
# 0 and (sᵢ, cᵢ) = (sin θᵢ, cos θᵢ). The frequencies w₁, w₂ are determined by the
# phases, so V(G) is a torus in ℝ⁶. Removing the points where the jacobian in (s, c)
# is singular -- the fold where steady states are born and die -- splits it into the
# regions on which the steady state depends smoothly on the frequencies.
#
# Expected: 4 components. The fold is the curve cos θ₁ cos θ₂ + cos(θ₁ - θ₂)(cos θ₁ +
# cos θ₂) = 0, which cuts three discs out of the torus (around the in-phase state and
# the two splay states), leaving a torus minus three discs: χ = -3, 1, 1, 1, made of
# 6 maxima and 6 saddles.
#
# Runtime: under a minute. Section 2 skips monodromy, to show what it adds.

using ConnectedComponents
using LinearAlgebra
using Random

Random.seed!(3)

@var s[1:2] c[1:2] w[1:2]
freq1 = (s[1] * c[2] - c[1] * s[2]) + s[1] - 3 * w[1]
freq2 = (s[2] * c[1] - c[2] * s[1]) + s[2] - 3 * w[2]
norm1 = s[1]^2 + c[1]^2 - 1
norm2 = s[2]^2 + c[2]^2 - 1
G = [freq1, freq2, norm1, norm2]

J = differentiate(G, [s; c])
f = expand(det(J) / 4)

r = RoutingFunction(f, vcat(s, c, w))
cache = RoutingCache(r, G)

# ---------------------------------------------------------------------------
# 1. the whole pipeline
# ---------------------------------------------------------------------------
C = connected_components(cache; grad_step_size = 0.1, tol = 0.2, start_step_size = 0.5)
display(C)

# ---------------------------------------------------------------------------
# 2. flows only
# ---------------------------------------------------------------------------
# the gradient flows alone find the maxima, and enough saddles to join them: the
# component count is right. They miss some saddles, though, so the Euler
# characteristic of the large component comes out wrong (-2, -1 or 0 instead of -3).
pts = flow_to_routing_points(cache; nstarts = 200)
C = connected_components(cache, pts; grad_step_size = 0.1, tol = 0.2, start_step_size = 0.5)
display(C)
