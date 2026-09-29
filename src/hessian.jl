# second-order information about r on X = V(G): the hessian of r restricted to X,
# and from it the morse index of a routing point. derivations in PIPELINE.md.
#
# with V an orthonormal basis of T_P X and λ the least-squares multipliers
# (JGᵀλ ≈ ∇r, exact at a critical point), the hessian of r|X is
#
#     H = Vᵀ (∇²r - ∑ⱼ λⱼ ∇²gⱼ) V                                          (1)
#
# at every point of X (the -∑ⱼ λⱼ ∇²gⱼ term is the second fundamental form). it
# equals the paper's Vᵀ∇²r V + ∑ᵢ ∂ᵢr·Wᵢ without building the W matrices. at a
# critical point two more forms agree with it:
#
#   (2) H = Vᵀ(∇²h - ∑ⱼ νⱼ∇²gⱼ)V / gᵈ   with h = f - r(P)·gᵈ, JGᵀν = ∇h;
#   (3) H = Vᵀ ∂ₓF₁(P, μ) V / gᵈ⁺¹      with F₁ = grad_num - JGᵀμ the first
#       block of the routing system, i.e. the tangential block of its jacobian.
#
# the package uses (1) (no case split, and (2), (3) are no more accurate); (3) is
# `_hessian_from_routing_system`, used by the tests as a cross-check.
#
# the morse index of a routing point P is the number of eigenvalues of H with the
# sign of r(P): the number of directions in which |r| increases.

# the projected lagrangian hessian Vᵀ(∇²φ - ∑ⱼ λⱼ ∇²gⱼ)V with JGᵀλ = ∇φ in the
# least-squares sense, for any function φ whose gradient and hessian at P are given.
# `JG` is k × n and `HG` stacks the n × n hessians of the gⱼ row-block by row-block.
function _projected_hessian(
    JG::AbstractMatrix{Float64},
    HG::AbstractMatrix{Float64},
    ∇φ::AbstractVector{Float64},
    ∇²φ::AbstractMatrix{Float64}
    )

    k, n = size(JG)
    V = LA.nullspace(JG)                 # orthonormal basis of T_P X; SVD, so rank revealing
    λ = transpose(JG) \ ∇φ               # least squares, n × k
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
    return _projected_hessian(JG, cache.HG_val, ∇r, ∇²r)
end

# form (3) of the header, from the jacobian of the routing system at (P, μ). only
# valid at a routing point; used to cross-check (1).
function _hessian_from_routing_system(cache::RoutingCache, point::AbstractVector{<:Real})
    r = cache.r
    n = cache.n
    x = Vector{Float64}(point)
    JG = copy(jacobian_G!(cache, x))
    μ = transpose(JG) \ _grad_num(r, x)
    J = zeros(ComplexF64, cache.m, cache.N)
    HC.jacobian!(J, cache.sys_interp, ComplexF64.(vcat(x, μ)))
    V = LA.nullspace(JG)
    H = transpose(V) * real.(J[1:n, 1:n]) * V ./ _g(r, x)^(r.d + 1)
    return (H + transpose(H)) ./ 2, V
end

"""
    hessian(cache, P)
    hessian(r, G, P)
    hessian(φ::Expression, G, vars, P)

The hessian of `r` (or of any expression `φ`) restricted to `V(G)` at `P`, in the
orthonormal tangent basis returned by `hessian_and_tangent`. Valid at every point.
"""
hessian(cache::RoutingCache, point::AbstractVector{<:Real}) =
    first(hessian_and_tangent(cache, point))

hessian(r::RoutingFunction, G, point::AbstractVector{<:Real}) =
    hessian(RoutingCache(r, G), point)

function hessian(
    φ::Expression,
    G,
    vars::Vector{Variable},
    point::AbstractVector{<:Real}
    )

    JG, HG = _G_derivatives(_as_expressions(G), vars, point)
    ∇φ, Hφ = ambient_gradient_hessian(φ, vars, point)
    return first(_projected_hessian(JG, HG, ∇φ, Hφ))
end

# tangential part of ∇r relative to its size: 0 at a critical point of r|X. the
# |r|/(1 + ‖P‖) term keeps the ratio meaningful in the rare case ∇r itself vanishes.
function _critical_residual(
    ∇r::AbstractVector{Float64},
    JG::AbstractMatrix{Float64},
    r_val::Float64,
    point::AbstractVector{Float64}
    )::Float64

    t = ∇r - transpose(JG) * (transpose(JG) \ ∇r)
    scale = LA.norm(∇r) + abs(r_val) / (1 + LA.norm(point))
    return scale == 0 ? 0.0 : LA.norm(t) / scale
end

# how close x is to a critical point of r|X, measured two ways from one evaluation:
#
#   distance: the length of the Newton step to the nearest critical point,
#             ‖H⁻¹ Vᵀ∇r‖ with H from (1) -- an honest length, independent of the
#             scaling of f and of how thin x's region is. Inf if H is singular.
#   residual: ‖tangential part of ∇r‖ relative to ‖∇r‖ (see _critical_residual).
#             it fails in thin regions: at an exact critical point the rounding of
#             x alone leaves a residual of about eps·(scale/width)², which passes
#             1e-6 once a region is ~1e-5 wide. kept as the fallback for singular H.
function _criticality(cache::RoutingCache, x::Vector{Float64})
    JG = copy(jacobian_G!(cache, x))
    HC.jacobian!(cache.HG_val, cache.∇G_sys, x)
    ∇r, ∇²r = evaluate_grad_hessian_r(cache.r, x)
    H, V = _projected_hessian(JG, cache.HG_val, ∇r, ∇²r)
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
    return dist, _critical_residual(∇r, JG, evaluate_r(cache.r, x), x)
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
    ambient_gradient_hessian(r, P)
    ambient_gradient_hessian(φ::Expression, vars, P)

The gradient and hessian in the ambient space `ℝⁿ` (not restricted to `V(G)`) of the
routing function, or of any expression `φ`, at `P`.
"""
ambient_gradient_hessian(r::RoutingFunction, point::AbstractVector{<:Real}) =
    evaluate_grad_hessian_r(r, point)

ambient_gradient_hessian(cache::RoutingCache, point::AbstractVector{<:Real}) =
    evaluate_grad_hessian_r(cache.r, point)

# gradient and hessian of any expression in ℝⁿ
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

# the W matrices of "Smooth Connectivity in Real Algebraic Varieties". not used by
# the pipeline (see (1) above); kept to check the paper's worked examples.
#
# `JG` is k × n, `HG` stacks the n × n hessians of the gᵢ row-block by row-block,
# and both are read, never written, so callers may hand in cache buffers.
function _compute_matrices(
    JG::AbstractMatrix{Float64},
    HG::AbstractMatrix{Float64},
    k::Int64
    )

    # columns form an orthonormal basis of T_x(V(G))
    V = LA.nullspace(JG)
    n, d = size(V)

    # well-constrained linear system for the stacked W's: the first k·d rows say
    # ∑ᵢ JG[j,i]·Wᵢ = -Vᵀ ∇²gⱼ V, the last d² rows say ∑ᵢ V[i,j]·Wᵢ = 0.
    A = zeros(Float64, k * d + d * d, n * d)
    @inbounds for j = 1:k, i = 1:n
        Jji = JG[j, i]
        for t = 1:d
            A[(j-1)*d+t, (i-1)*d+t] = Jji
        end
    end
    @inbounds for j = 1:d, i = 1:n
        Vij = V[i, j]
        for t = 1:d
            A[k*d+(j-1)*d+t, (i-1)*d+t] = Vij
        end
    end

    B = zeros(Float64, k * d + d * d, d)
    VtHV = zeros(Float64, d, d)
    tmp = zeros(Float64, n, d)
    @inbounds for j = 1:k
        Hj = view(HG, (j-1)*n+1:j*n, :)
        LA.mul!(tmp, Hj, V)
        LA.mul!(VtHV, transpose(V), tmp)
        B[(j-1)*d+1:j*d, :] .= .-VtHV
    end

    W = A \ B

    return [W[(i-1)*d+1:i*d, :] for i = 1:n], V
end

"""
    compute_matrices(cache, P) -> (W, V)
    compute_matrices(G, vars, P) -> (W, V)

The matrices `W₁, …, Wₙ` of "Smooth Connectivity in Real Algebraic Varieties" at `P`,
with the orthonormal tangent basis `V` they are written in: the hessian of `φ` on
`V(G)` is `Vᵀ∇²φ V + ∑ᵢ ∂ᵢφ Wᵢ`. The package computes hessians without them (see
[`hessian_and_tangent`](@ref)); they are kept to check the paper's worked examples.
"""
function compute_matrices(cache::RoutingCache, point::AbstractVector{<:Real})
    x = Vector{Float64}(point)
    jacobian_G!(cache, x)
    HC.jacobian!(cache.HG_val, cache.∇G_sys, x)
    return _compute_matrices(cache.JG_val, cache.HG_val, cache.k)
end

function compute_matrices(
    G,
    vars::Vector{Variable},
    point::AbstractVector{<:Real}
    )

    G = _as_expressions(G)
    JG, HG = _G_derivatives(G, vars, point)
    return _compute_matrices(JG, HG, length(G))
end

# gradient of the routing function at a point of X = V(G), in the tangent basis
function gradient(cache::RoutingCache, point::AbstractVector{<:Real})
    ∇r = reshape(evaluate_grad_r(cache.r, point), 1, :)
    V = LA.nullspace(jacobian_G!(cache, Vector{Float64}(point)))
    return ∇r * V
end

gradient(r::RoutingFunction, G, point::AbstractVector{<:Real}) =
    gradient(RoutingCache(r, G), point)

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
