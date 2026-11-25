# second-order information about r on X = V(G): the hessian of r restricted to X,
# and from it the morse index of a routing point. derivation in PIPELINE.md §1.5.
#
# with V an orthonormal basis of T_P X and λ the least-squares multipliers
# (JGᵀλ ≈ ∇r, exact at a critical point), the hessian of r|X is
#
#     H = Vᵀ (∇²r - ∑ⱼ λⱼ ∇²gⱼ) V
#
# at every point of X: the second derivative of r along the geodesics of X. the
# -∑ⱼ λⱼ ∇²gⱼ term accounts for the curvature of X, which makes r curve along X even
# in directions where it is flat in ℝⁿ.
#
# the morse index of a routing point P is the number of eigenvalues of H with the
# sign of r(P): the number of directions in which |r| increases.

# the projected lagrangian hessian Vᵀ(∇²φ - ∑ⱼ λⱼ ∇²gⱼ)V with JGᵀλ = ∇φ in the
# least-squares sense, for any function φ whose gradient and hessian at P are given.
# `JG` is k × n and `HG` stacks the n × n hessians of the gⱼ row-block by row-block.
#
# JG has rank c on X, the codimension, which is below k when G has more equations than
# that. the tangent space and the multipliers come from the rank-c truncated SVD: at a
# point a little off X the singular values after the c-th are small but not zero, and
# a rank test near machine precision would count them, losing tangent directions. the
# multipliers are then not unique, but Vᵀ(∑ⱼ λⱼ ∇²gⱼ)V is the same for all of them.
function _projected_hessian(
    JG::AbstractMatrix{Float64},
    HG::AbstractMatrix{Float64},
    ∇φ::AbstractVector{Float64},
    ∇²φ::AbstractMatrix{Float64},
    c::Integer = size(JG, 1)
    )

    k, n = size(JG)
    F = LA.svd(JG; full = true)
    c = min(c, length(F.S))
    V = F.V[:, c+1:n]                    # orthonormal basis of T_P X
    # least squares JGᵀλ = ∇φ, truncated at rank c; λ has length k
    λ = F.U[:, 1:c] * ((transpose(F.V[:, 1:c]) * ∇φ) ./ F.S[1:c])
    L = copy(∇²φ)
    @inbounds for j = 1:k
        L .-= λ[j] .* view(HG, (j-1)*n+1:j*n, :)
    end
    H = transpose(V) * L * V
    return (H + transpose(H)) ./ 2, V    # symmetric in exact arithmetic
end

"""
    hessian_and_tangent(cache, P) -> (H, V)

The hessian `H` of `r` restricted to `V(G)` at `P`, expressed in the orthonormal
basis `V` of the tangent space `T_P V(G)` that it is returned with:
`H = Vᵀ(∇²r - ∑ⱼ λⱼ ∇²gⱼ)V`, where `JGᵀλ = ∇r` in the least-squares sense (the
Lagrange multipliers, at a critical point). Valid at every point of `V(G)`.
"""
function hessian_and_tangent(cache::RoutingCache, point::AbstractVector{<:Real})
    x = Vector{Float64}(point)
    JG = copy(jacobian_G!(cache, x))
    HC.jacobian!(cache.HG_val, cache.∇G_sys, x)
    ∇r, ∇²r = evaluate_grad_hessian_r(cache.r, x)
    return _projected_hessian(JG, cache.HG_val, ∇r, ∇²r, cache.c)
end

"""
    hessian(cache, P)
    hessian(r, G, P)
    hessian(φ::Expression, G, vars, P; dim = nothing)

The hessian of `r` (or of any expression `φ`) restricted to `V(G)` at `P`, in the
orthonormal tangent basis returned by `hessian_and_tangent`. Valid at every point.
For an expression, `dim` is the dimension of `V(G)` at `P`; by default it is read off
the numerical rank of the jacobian of `G` at `P`.
"""
hessian(cache::RoutingCache, point::AbstractVector{<:Real}) =
    first(hessian_and_tangent(cache, point))

hessian(r::RoutingFunction, G, point::AbstractVector{<:Real}) =
    hessian(RoutingCache(r, G), point)

function hessian(
    φ::Expression,
    G,
    vars::Vector{Variable},
    point::AbstractVector{<:Real};
    dim::Union{Nothing,Integer} = nothing
    )

    G = _as_expressions(G)
    JG, HG = _G_derivatives(G, vars, point)
    ∇φ, Hφ = ambient_gradient_hessian(φ, vars, point)
    c = dim !== nothing ? length(vars) - dim :
        _numerical_rank(LA.svdvals(JG ./ _row_scales(G, vars)))
    return first(_projected_hessian(JG, HG, ∇φ, Hφ, c))
end

# tangential part of ∇r relative to its size: 0 at a critical point of r|X. the
# |r|/(1 + ‖P‖) term keeps the ratio meaningful in the rare case ∇r itself vanishes.
function _critical_residual(
    ∇r::AbstractVector{Float64},
    JG::AbstractMatrix{Float64},
    r_val::Float64,
    point::AbstractVector{Float64},
    c::Integer = size(JG, 1)
    )::Float64

    # minus the projection onto the normal space, the span of the top c right singular vectors
    F = LA.svd(JG)
    Vc = F.V[:, 1:min(c, length(F.S))]
    t = ∇r - Vc * (transpose(Vc) * ∇r)
    scale = LA.norm(∇r) + abs(r_val) / (1 + LA.norm(point))
    return scale == 0 ? 0.0 : LA.norm(t) / scale
end

# how close x is to a critical point of r|X, measured two ways from one evaluation:
#
#   distance: the length of the Newton step to the nearest critical point,
#             ‖H⁻¹ Vᵀ∇r‖ with H the hessian above -- an honest length, independent
#             of the scaling of f and of how thin x's region is. Inf if H is singular.
#   residual: ‖tangential part of ∇r‖ relative to ‖∇r‖ (see _critical_residual).
#             it fails in thin regions: at an exact critical point the rounding of
#             x alone leaves a residual of about eps·(scale/width)², which passes
#             1e-6 once a region is ~1e-5 wide. kept as the fallback for singular H.
function _criticality(cache::RoutingCache, x::Vector{Float64})
    JG = copy(jacobian_G!(cache, x))
    HC.jacobian!(cache.HG_val, cache.∇G_sys, x)
    ∇r, ∇²r = evaluate_grad_hessian_r(cache.r, x)
    H, V = _projected_hessian(JG, cache.HG_val, ∇r, ∇²r, cache.c)
    gT = transpose(V) * ∇r
    E = LA.eigen(LA.Symmetric(H))
    μmax = maximum(abs, E.values; init = 0.0)
    dist = if isempty(E.values)
        0.0
    elseif μmax == 0 || minimum(abs, E.values) <= 1e-12 * μmax
        Inf
    else
        LA.norm(E.vectors * ((transpose(E.vectors) * gT) ./ E.values))
    end
    return dist, _critical_residual(∇r, JG, evaluate_r(cache.r, x), x, cache.c)
end

"""
    critical_distance(cache, P)

How far `P` is from a critical point of `r` on `V(G)`: the length of the Newton
step from `P` to the nearest critical point, `‖H⁻¹ Vᵀ∇r(P)‖`, where `H` is the
hessian of `r` on `V(G)` and `V` a basis of the tangent space. `0` at a critical
point; `Inf` where the hessian is singular.
"""
critical_distance(cache::RoutingCache, point::AbstractVector{<:Real})::Float64 =
    first(_criticality(cache, Vector{Float64}(point)))

# whether x passes the criticality test: a Newton step shorter than
# crit_tol·(1 + ‖x‖), or -- where H is (nearly) singular -- a tangential gradient
# below crit_tol relative to ‖∇r‖
function _is_critical(cache::RoutingCache, x::Vector{Float64}; crit_tol::Float64 = 1e-6)::Bool
    dist, res = _criticality(cache, x)
    return dist <= crit_tol * (1 + LA.norm(x)) || res <= crit_tol
end

"""
    ambient_gradient_hessian(φ::Expression, vars, P)

The gradient and hessian of the expression `φ` in the ambient space `ℝⁿ` (not
restricted to `V(G)`) at `P`. For the routing function itself, use
[`evaluate_grad_hessian_r`](@ref).
"""
function ambient_gradient_hessian(
    g::Expression,
    vars::Vector{Variable},
    point::AbstractVector{<:Real}
    )

    ∇g = HC.differentiate(g, vars)
    grad_sys = HC.InterpretedSystem(System(∇g; variables = vars))
    val = zeros(Float64, length(vars))
    H = zeros(Float64, length(vars), length(vars))
    HC.evaluate_and_jacobian!(val, H, grad_sys, Vector{Float64}(point))
    return val, H
end

# JG and the stacked hessians of the gᵢ at a point, from scratch
function _G_derivatives(G::Vector{Expression}, vars::Vector{Variable}, point::AbstractVector{<:Real})
    n, k = length(vars), length(G)
    x = Vector{Float64}(point)
    G_sys = HC.InterpretedSystem(System(G; variables = vars))
    ∇G_sys = HC.InterpretedSystem(
        System(reduce(vcat, HC.differentiate(g, vars) for g in G); variables = vars),
    )
    JG = zeros(Float64, k, n)
    HG = zeros(Float64, k * n, n)
    HC.jacobian!(JG, G_sys, x)
    HC.jacobian!(HG, ∇G_sys, x)
    return JG, HG
end

"""
    morse_index(cache, P)
    idx(r, H, P)

The Morse index of the routing point `P`: the number of eigenvalues of the hessian
`H` of `r` on `V(G)` whose sign is the sign of `r(P)` -- the number of directions in
which `|r|` increases (the Morse index of `-|r|`). Index 0 is a local maximum of
`|r|`. A component's Euler characteristic is the sum of `(-1)^index` over its routing
points. `idx` takes a hessian already computed by [`hessian`](@ref).
"""
function idx(
    r::RoutingFunction,
    H::AbstractMatrix{Float64},
    critical_point::AbstractVector{<:Real}
    )::Int64

    σ = sign(evaluate_r(r, critical_point))
    i = 0
    for eval in LA.eigvals(LA.Symmetric(H))
        if sign(eval) == σ
            i += 1
        end
    end

    return i
end

morse_index(cache::RoutingCache, point::AbstractVector{<:Real})::Int64 =
    idx(cache.r, hessian(cache, point), point)

# a hessian eigenvalue this small relative to |r|/(1 + ‖P‖)², the natural scale of
# the hessian, means the critical point is degenerate: r is not Morse there
_degenerate(H, rP, P) = minimum(abs, LA.eigvals(LA.Symmetric(Matrix(H))); init = Inf) <=
                        1e-10 * abs(rP) / (1 + LA.norm(P))^2

"""
    routing_point_indices(cache, points) -> Vector{Int}

The [`morse_index`](@ref) of every routing point, in the order given. Warns when some
of them are degenerate critical points (a hessian eigenvalue numerically zero): `r`
is then not a Morse function, and indices, flows and Euler characteristics near those
points cannot be trusted. The usual cause is a non-generic centre `c`, such as a
centre of symmetry of `V(G)`.
"""
function routing_point_indices(
    cache::RoutingCache,
    critical_points::AbstractVector{<:AbstractVector{<:Real}}
    )::Vector{Int64}

    indices = Int64[]
    ndegenerate = 0
    for P in critical_points
        H = hessian(cache, P)
        rP = evaluate_r(cache.r, P)
        ndegenerate += _degenerate(H, rP, P)
        push!(indices, idx(cache.r, H, P))
    end
    ndegenerate > 0 && @warn(
        "$ndegenerate of the $(length(critical_points)) routing points are degenerate " *
        "critical points of r on V(G) (a hessian eigenvalue is numerically zero), so r is " *
        "not a Morse function and the indices, paths and Euler characteristics near them " *
        "are unreliable. The usual cause is a centre c that is not generic -- a centre of " *
        "symmetry of V(G), say; build the RoutingFunction with a random c.",
    )
    return indices
end

routing_point_indices(
    r::RoutingFunction,
    G,
    critical_points::AbstractVector{<:AbstractVector{<:Real}},
) = routing_point_indices(RoutingCache(r, G), critical_points)

"""
    sort_routing_points_by_index(cache, points) -> Dict{Int, Vector{Vector{Float64}}}

The routing points grouped by Morse index: `d[k]` holds those of index `k`. Only the
indices that occur are keys, so use `get(d, k, [])` for an index that may not.
"""
function sort_routing_points_by_index(
    cache::RoutingCache,
    critical_points::AbstractVector{<:AbstractVector{<:Real}}
    )

    sorter = Dict{Int,Vector{Vector{Float64}}}()

    for (P, ind) in zip(critical_points, routing_point_indices(cache, critical_points))
        push!(get!(() -> Vector{Float64}[], sorter, ind), Vector{Float64}(P))
    end

    return sorter
end

sort_routing_points_by_index(
    r::RoutingFunction,
    G,
    critical_points::AbstractVector{<:AbstractVector{<:Real}},
) = sort_routing_points_by_index(RoutingCache(r, G), critical_points)
