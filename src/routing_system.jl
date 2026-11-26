# the Lagrange system whose real solutions are the critical points of r|V(G), and the
# family of such systems that monodromy runs over.
#
# ∇r = grad_num / g^(d+1), so x is critical iff grad_num(x) = JG(x)ᵀμ for some μ, where
# μ = g^(d+1)λ for the multipliers λ of ∇r itself. using μ drops the degree of the first
# block from max(deg f + 1, 2d + deg G + 2) to max(deg f + 1, deg G). g ≥ 1 on ℝⁿ, so
# real solutions correspond one to one; the extra complex solutions have g = 0.
"""
    routing_system(r, G) -> System
    routing_system(cache) -> System

The polynomial system in `(x, μ)`

    grad_num(x) − JG(x)ᵀ μ = 0,     G(x) = 0,

where `grad_num = g ∇f − d f ∇g = g^(d+1) ∇r` is the numerator of `∇r`, and `μ` has one
entry per equation, with fresh variable names that cannot collide with the caller's.
For `G` as given, the solutions with `x` real and off `V(f)` are the routing points (the
critical points of `r` on `V(G) ∖ V(f)`) and the other real solutions lie on `V(f)`. The
`μ` are `g^(d+1)` times the Lagrange multipliers of `∇r`.

`routing_system(r, G)` is square only if `G` has as many equations as the codimension
of `V(G)`. `routing_system(cache)` uses the equations `cache.H` (see
[`RoutingCache`](@ref)), always square, so it also has real solutions off `V(G)`; the
routing points are those on `V(G)`.
"""
function routing_system(r::RoutingFunction, G)::System
    G = _as_expressions(G)
    # unique names, so the multipliers cannot collide with a user's own variables
    @unique_var μ[1:length(G)]
    eqns = vcat(r.grad_num - transpose(HC.differentiate(G, r.vars)) * μ, G)
    return System(eqns; variables = vcat(r.vars, μ))
end

# the centre family: the routing system of G (the equations H of the cache) with the
# centre c and constant a of g = ‖x - c‖² + a as parameters, in the order (c₁, …, cₙ, a).
# the routing system of r is the member at (r.c, 1).
#
# its solutions on X ∖ V(f) form one irreducible component when X is irreducible, so
# monodromy from any routing point reaches all the others (PIPELINE.md §3.2).
#
# the multipliers are projective, (μ₀ : μ̂) with μ = μ̂ / μ₀, normalised by a random
# chart ℓ₀μ₀ + ℓ·μ̂ = 1:
#
#     μ₀ grad_num − JGᵀμ̂ = 0,     G = 0,     ℓ₀μ₀ + ℓ·μ̂ = 1,
#
# in the variables (x, μ₀, μ̂). μ grows like ‖x‖^(deg f + 1) and reaches 10⁹ times x on
# a large variety (3RPR), making the jacobian numerically singular; the chart keeps the
# multipliers of every solution at size 1. see `_to_chart`.
function _centre_family(r::RoutingFunction, G::Vector{Expression}, chart::Vector{ComplexF64})::System
    n = length(r.vars)
    k = length(G)
    x = r.vars
    # plain names: HC's variable ordering cannot parse a subscript in a unique name
    @unique_var m0 m[1:k] c[1:n] a          # m0 = μ₀, m = μ̂
    g = sum((x[i] - c[i])^2 for i = 1:n) + a
    grad_num = g .* r.∇f .- (2 * r.d) .* r.f .* (x .- c)       # ∇g = 2(x - c)
    eqns = vcat(m0 .* grad_num - transpose(HC.differentiate(G, x)) * m, G,
                [chart[1] * m0 + sum(chart[j+1] * m[j] for j = 1:k) - 1])
    return System(eqns; variables = vcat(x, [m0], m), parameters = vcat(c, [a]))
end

# (x, μ) ↦ (x, μ₀, μ̂) on the chart, and back. nothing where the chart misses the point
# (ℓ₀ + ℓ·μ = 0) or the point is at infinity (μ₀ = 0).
function _to_chart(z::AbstractVector, n::Int64, chart::Vector{ComplexF64})
    μ = view(z, n+1:length(z))
    s = chart[1] + sum(chart[j+1] * μ[j] for j in eachindex(μ))
    (isfinite(s) && s != 0) || return nothing
    μ₀ = 1 / s
    return ComplexF64.(vcat(z[1:n], [μ₀], μ₀ .* μ))
end

function _from_chart(w::AbstractVector, n::Int64)
    μ₀ = w[n+1]
    (isfinite(μ₀) && μ₀ != 0) || return nothing
    return ComplexF64.(vcat(w[1:n], w[n+2:end] ./ μ₀))
end
