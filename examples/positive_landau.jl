# Positive Landau discriminants.
#
# F is the second Symanzik polynomial of a two-loop graph with four propagators:
# cubic and homogeneous in the Schwinger parameters x = (x1,...,x4), linear in the
# kinematic parameters p = (s,m,M). The incidence variety
#
#     V = {(x,p) ∈ (ℂ*)^4 × ℂ^3 : F = ∂F/∂x1 = ... = ∂F/∂x4 = 0}
#
# is the object whose projection to p-space is the Landau discriminant. The
# question here is which of its irreducible components carry a point whose x is
# real and strictly positive -- the physical sheet.

using ConnectedComponents
using LinearAlgebra
using Random

@var x1 x2 x3 x4 s m M
X = [x1, x2, x3, x4]
p = [s, m, M]
Z = [X; p]

F = m*x2*x3^2 + m*x2^2*x3 + m*x3*x1^2 + m*x3^2*x1 + m*x4*x1^2 + m*x4*x2^2 +
    m*x4*x3^2 + m*x4^2*x1 + m*x4^2*x2 + m*x4^2*x3 - M*x4*x2*x3 - M*x4*x3*x1 +
    2*m*x2*x3*x1 + 2*m*x4*x2*x1 + 3*m*x4*x2*x3 + 3*m*x4*x3*x1 -
    s*x2*x3*x1 - s*x4*x2*x1

dF = differentiate(F, X)

# ---------------------------------------------------------------------------
# setting the system up so that the routing machinery can see it
# ---------------------------------------------------------------------------
#
# three things have to be dealt with before V is a variety this package can work
# on. all three are consequences of F being homogeneous in x and linear in p.
#
# (1) F = 0 is redundant. F is homogeneous of degree 3 in x, so Euler gives
#     3F = ∑ xᵢ ∂F/∂xᵢ and the equation F = 0 follows from the four partials.
#     Carrying it anyway makes the jacobian of G rank deficient at *every* point
#     of V(G), which leaves the Lagrange multipliers of the routing system
#     undetermined -- the routing system then has a positive-dimensional fibre
#     over every routing point and nothing downstream means anything.
@assert expand(3*F - sum(X .* dF)) == 0

# (2) V contains {p = 0} × (ℂ*)^4. F is linear in p, so F(x,0) ≡ 0 and the whole
#     x-Hessian vanishes there: the jacobian of (∂F/∂x1,...,∂F/∂x4) at p = 0 is
#     [0 | J(x)], of rank ≤ 3 < 4. That 4-dimensional component is therefore a
#     singular locus of the system, with the same undetermined-multiplier problem.
#     It is also the uninteresting component -- it contains positive x for trivial
#     reasons -- so it should be cut away rather than worked around.
#
# (3) V is a cone in x and a cone in p. Critical points of r would come in
#     2-parameter families, so the routing system would again be positive
#     dimensional.
#
# One affine chart in each of x and p fixes (2) and (3) at once. δ is taken with
# positive entries so that every x in the positive orthant scales into δ·x = 1,
# which is the whole point; ℓ is generic, and ℓ·p = 1 is what removes {p = 0}.
δ = [0.31, 0.57, 0.83, 1.19]
ℓ = [0.37, -0.62, 0.91]
G = [dF; sum(δ .* X) - 1; sum(ℓ .* p) - 1]

# 6 equations in 7 unknowns. Y = V(G) is one surface and six curves -- the seven
# irreducible components of V that have p ≠ 0, sliced. Slicing does not break them
# up: each component is a cone in both x and p, so it maps onto its slice by
# rescaling, and the image of an irreducible variety is irreducible.

# removing the coordinate hyperplanes is exactly what a routing function does:
# on each connected component of Y_ℝ \ {x1x2x3x4 = 0} the sign vector of x is
# constant, so a component of the real variety is positive or it is not, and one
# routing point per region settles it.
r = RoutingFunction(x1*x2*x3*x4, Z)
cache = RoutingCache(r, G)

# ---------------------------------------------------------------------------
# 1. a single positive point, as fast as possible
# ---------------------------------------------------------------------------
#
# `stop_when` is a predicate on a real routing point in ℝⁿ. The flow returns the
# first point satisfying it, and `routing_points` then skips monodromy entirely.
# That matters here: the routing system is 13 equations whose first block has
# degree 11, so its generic fibre is far out of reach of `monodromy_solve`.
#
# the threshold is not cosmetic. Newton lands on plenty of points with a
# coordinate at 1e-6, which sit on {x1x2x3x4 = 0} rather than in a positive
# region; asking for a definite margin keeps those out.
is_positive(P; tol = 1e-4) = all(>(tol), view(P, 1:4))

Random.seed!(1)
hit = routing_points(cache; stop_when = is_positive, nstarts = 200,
                     box = 3.0, verbose = true)
if isempty(hit)
    println("no positive routing point found; raise nstarts")
else
    q = hit[1]
    println("positive routing point")
    println("   x = ", round.(q[1:4], sigdigits = 6))
    println("   p = ", round.(q[5:7], sigdigits = 6))
    println("   |G(q)| = ", norm(Float64.(evaluate(G, Z => q))))
end

# ---------------------------------------------------------------------------
# 2. all of them: the sign vectors that occur on Y_ℝ
# ---------------------------------------------------------------------------
#
# Dropping `stop_when` and taking every routing point the flow reaches gives the
# census. The argument that this is complete, and not just a sample, is the one
# the routing function is built on: the flow follows sign(r)∇r, so |r| increases
# along it and the path can never cross {x1x2x3x4 = 0}, where r = 0. Every path
# therefore ends at a local maximum of |r| inside the region it started in, and
# every region of Y_ℝ \ {x1x2x3x4 = 0} has at least one such maximum, because
# |r| → 0 both on the boundary of the region and at infinity. So the sign vectors
# carried by routing points are exactly the sign vectors realised on Y_ℝ.
#
# What is *not* guaranteed is that a random start lands in every region; that is
# the same lower-bound caveat as the 27 lines example, and is why nstarts is high.
Random.seed!(2)
seeds = flow_to_routing_points(cache; nstarts = 4000, box = 3.0)
real_pts = [real.(z[1:7]) for z in seeds if maximum(abs ∘ imag, z[1:7]) < 1e-8]

census = Dict{NTuple{4,Int},Vector{Vector{Float64}}}()
for q in real_pts
    minimum(abs, view(q, 1:4)) > 1e-4 || continue
    push!(get!(() -> Vector{Float64}[], census, ntuple(i -> q[i] > 0 ? 1 : -1, 4)), q)
end
println("\n", length(real_pts), " real routing points, ", length(census), " sign vectors:")
for (sv, qs) in sort(collect(census); by = q -> -length(last(q)))
    println("   ", sv, "   ", length(qs), " points", sv == (1,1,1,1) ? "   <-- positive" : "")
end

positive = get(census, (1,1,1,1), Vector{Float64}[])

# ---------------------------------------------------------------------------
# 3. which irreducible components the positive points lie on
# ---------------------------------------------------------------------------
#
# A routing point says a positive region exists; the witness sets say which
# component it belongs to. Membership: slice the component with a random affine
# subspace of complementary dimension *through the point*. The slice meets the
# component in deg(W) points, and the point is one of them exactly when it lies
# on the component.
N = numerical_irreducible_decomposition(System(G; variables = Z))
comps = [(d, i, W) for d in sort(collect(keys(N.witness_sets)); rev = true)
                   for (i, W) in enumerate(N.witness_sets[d])]
println("\nsliced V has ", length(comps), " components: ",
        [(d, degree(W)) for (d, _, W) in comps])

# distance from q to the component: slice through q and take the nearest point of
# the slice. On the component that distance is the tracking error; off it, it is
# the distance to the nearest other branch, which is orders of magnitude larger.
function distance_to_component(q, W, d; tries = 5)
    best = Inf
    for _ = 1:tries
        A = randn(d, length(q))
        Wq = try
            witness_set(W, LinearSubspace(A, A * q); show_progress = false)
        catch
            continue
        end
        for sol in solutions(Wq)
            maximum(abs ∘ imag, sol) < 1e-6 || continue
            best = min(best, maximum(abs, real.(sol) .- q))
        end
    end
    return best
end

function which_component(q, comps; tol = 1e-5)
    ds = [distance_to_component(q, W, d) for (d, _, W) in comps]
    j = argmin(ds)
    return ds[j] < tol ? j : nothing
end

labels = [(d, i, degree(W)) for (d, i, W) in comps]
assign = [which_component(q, comps) for q in positive]
hits = sort(unique(filter(!isnothing, assign)))
println("\n", length(hits), " of the ", length(comps),
        " components of the sliced V carry a point with x > 0:")
for j in hits
    d, i, deg = labels[j]
    w = positive[findfirst(==(j), assign)]
    println("   component #$j:  dim $d, degree $deg")
    println("        x = ", round.(w[1:4], sigdigits = 6), "   p = ", round.(w[5:7], sigdigits = 6))
end
println("\nunassigned positive routing points: ", count(isnothing, assign), " of ", length(positive))

# ---------------------------------------------------------------------------
# the answer, and how it lines up with the count of seven
# ---------------------------------------------------------------------------
#
# Over ℂ, and in the torus, V has eight irreducible components: {p = 0} × (ℂ*)^4
# of dimension 4, and the seven with p ≠ 0 that the charts above leave -- one more
# of dimension 4 (degree 3 in the slice) and six of dimension 3 (degrees 4, 3, 1,
# 1, 1, 1). Two of the four lines are a complex conjugate pair and have no real
# points at all, so over ℚ they are a single prime; that merge turns eight into
# the seven of the email.
#
# Three of those seven contain a point with x real and strictly positive:
#
#   * {p = 0}, for the trivial reason that any positive x will do -- it is the
#     component the ℓ·p = 1 chart deliberately removes,
#   * the degree 3 curve, e.g. x = (0.5637, 0.5637, 0.2495, 0.2495),
#                               p = (0.6193, 0.1628, 0.9580),
#   * one of the degree 1 lines, on which s vanishes identically, e.g.
#     x = (0.2411, 0.1692, 0.4103, 0.4103), p = (0, 0.1321, 1.1889).
#
# Both nontrivial witnesses have x1 = x2 or x3 = x4: F is invariant under x1 ↔ x2
# and under x3 ↔ x4 separately, and the positive points sit on the fixed locus.
# That symmetry is visible in the m-coefficient, which factors as
#
#     (x1 + x2 + x3 + x4) · ((x1 + x2)(x3 + x4) + x3 x4),
#
# the first Symanzik polynomial times the sum of the Schwinger parameters.
#
# The other four are not positive, and each for a different reason: the dimension
# 4 component and the degree 4 curve have real points in seven and three sign
# classes respectively but never in (+,+,+,+); one degree 1 line is real but its
# positive interval is empty; the conjugate pair is not real at all.
