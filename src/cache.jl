"""
    RoutingCache(r, G; dim = nothing, reg = 1e-8)

Everything the pipeline precomputes from a routing function `r` and the equations `G`
of `X = V(G)`: compiled systems for `G` and its derivatives, the routing system, the
two monodromy families, the two gradient fields, and scratch buffers. The `(r, G, ...)`
convenience methods build a fresh cache on every call.

`G` (a vector of expressions, an expression, or a `System`) must use only the
variables of `r`. It may have more equations `k` than the codimension `c` of `X`, but
its jacobian must have rank `c` on `X` off `V(f)`. `c = n - dim` if `dim` is given,
else `1` for one equation and [`codimension`](@ref)`(G, r.vars)` for several.

If `c < k`, homotopy continuation still needs a square system, so the routing system
is built from `H = B G`: `c` random combinations of the row-normalised equations.
Solutions on the extra components of `V(H)` are discarded. Everything defined on all
of `X` (flows, projection, hessians) uses `G`, since the jacobian of `H` drops rank on
a hypersurface of `X`.

`reg` damps `J Jᵀ` where the jacobian is nearly rank deficient. A cache holds mutable
buffers: use one per thread.
"""
struct RoutingCache{F,Fu}
    r::RoutingFunction
    G::Vector{Expression}
    vars::Vector{Variable}
    n::Int64                        # ambient dimension
    k::Int64                        # number of defining equations of V(G)
    c::Int64                        # codimension of V(G): the rank of JG on V(G)
    H::Vector{Expression}           # the c equations of the routing system: G, or B G
    B::Union{Nothing,Matrix{Float64}}   # c × k: H = B G; nothing when H = G
    reg::Float64                    # regularisation of J Jᵀ in the flow fields and projections

    G_sys::HC.InterpretedSystem     # G; its jacobian is JG
    ∇G_sys::HC.InterpretedSystem    # [∇g₁; …; ∇gₖ]; its jacobian stacks the hessians of the gᵢ

    sys::System                     # routing system of H in (x, μ)
    sys_interp::HC.InterpretedSystem
    sys_vars::Vector{Variable}      # vcat(vars, μ)
    centre_sys::System              # sys with the centre and constant of g as parameters,
                                    # in (x, μ₀, μ̂) on the chart below
    chart::Vector{ComplexF64}       # the random chart ℓ₀μ₀ + ℓ·μ̂ = 1 of centre_sys
    param_sys::System               # sys minus generic affine-linear forms
    m::Int64                        # number of equations of sys
    N::Int64                        # n + c, the (x, μ) count

    flow!::F                        # projected gradient field of r on V(G)
    flow_unit!::Fu                  # same field, rescaled so that time is arc length

    # scratch, sized once (the flow fields keep their own)
    G_val::Vector{Float64}          # k
    JG_val::Matrix{Float64}         # k × n
    HG_val::Matrix{Float64}         # (k·n) × n; rows (i-1)n+1:in are the hessian of gᵢ
    M::Matrix{Float64}              # k × k
    wk::Vector{Float64}             # k
    wn::Vector{Float64}             # n
    wn2::Vector{Float64}            # n
end

function RoutingCache(
    r::RoutingFunction,
    G;
    dim::Union{Nothing,Integer} = nothing,
    reg::Real = 1e-8,
    )

    reg > 0 || throw(ArgumentError("reg must be positive, got $reg"))
    reg = Float64(reg)
    G = _as_expressions(G)
    isempty(G) && throw(ArgumentError("G is empty: give at least one equation"))
    vars = r.vars
    n = length(vars)
    k = length(G)

    # G has to live in r's variables
    stray = setdiff(variables(G), vars)
    isempty(stray) || throw(ArgumentError(
        "G involves variables that r does not: $(join(stray, ", ")). r is a routing " *
        "function in $(join(vars, ", ")) -- rebuild it over all the variables, e.g. " *
        "RoutingFunction(f, [$(join(union(vars, stray), ", "))], c)",
    ))

    G_sys = HC.InterpretedSystem(System(G; variables = vars))
    # stacked row-wise so that the jacobian's i-th n × n block is exactly ∇²gᵢ
    ∇G_sys = HC.InterpretedSystem(
        System(reduce(vcat, HC.differentiate(g, vars) for g in G); variables = vars),
    )

    c = _cache_codimension(G, vars, n, k, dim)
    H, B = _square_up(G, vars, c)

    sys = routing_system(r, H)
    sys_vars = variables(sys)
    m = length(expressions(sys))
    N = length(sys_vars)

    # the monodromy families (PIPELINE.md §3.2). the affine family sys(z) - (Q z + q),
    # z = (x, μ), is sys at parameter 0; much larger than the centre family, but
    # transitive even when X is reducible.
    chart = randn(ComplexF64, c + 1)
    centre_sys = _centre_family(r, H, chart)
    @unique_var q[1:m, 1:(N + 1)]   # unique names: a user variable q must not collide
    shifts = [sum(q[i, j] * sys_vars[j] for j = 1:N) + q[i, N+1] for i = 1:m]
    param_sys = System(
        [expressions(sys)[i] - shifts[i] for i = 1:m];
        variables = sys_vars,
        parameters = vec(q),
    )

    flow! = _projected_gradient_field(r, G_sys, n, k, reg, false)
    flow_unit! = _projected_gradient_field(r, G_sys, n, k, reg, true)

    return RoutingCache(
        r, G, vars, n, k, c, H, B, reg,
        G_sys, ∇G_sys,
        sys, HC.InterpretedSystem(sys), sys_vars, centre_sys, chart, param_sys, m, N,
        flow!, flow_unit!,
        zeros(Float64, k),          # G_val
        zeros(Float64, k, n),       # JG_val
        zeros(Float64, k * n, n),   # HG_val
        zeros(Float64, k, k),       # M
        zeros(Float64, k),          # wk
        zeros(Float64, n),          # wn
        zeros(Float64, n),          # wn2
    )
end

# c from dim, or estimated. an estimate below k leads to squaring up, which would hide
# a rank that is low because G is not reduced: check for that.
function _cache_codimension(G::Vector{Expression}, vars::Vector{Variable}, n::Int64, k::Int64,
                            dim::Union{Nothing,Integer})::Int64
    if dim !== nothing
        c = n - dim
        1 <= c <= min(k, n) || throw(ArgumentError(
            "dim = $dim: the dimension of V(G) must be between $(n - min(k, n)) and " *
            "$(n - 1), since V(G) ⊂ ℝ^$n is cut out by $k equation$(k == 1 ? "" : "s")"))
        return c
    end
    k == 1 && return 1
    c = codimension(G, vars)
    c < k && !_looks_reduced(G, vars, c) && @warn(
        "RoutingCache: the jacobian of G has rank $c at points of V(G), below the $k " *
        "equations, but G does not look reduced there (a repeated factor?). Then the " *
        "codimension of V(G) is larger than $c and the results are wrong. Rewrite G, or " *
        "pass the dimension: RoutingCache(r, G; dim = ...).")
    return c
end

# `_reduced_at` at a few complex points of V(G); fixed seed
function _looks_reduced(G::Vector{Expression}, vars::Vector{Variable}, c::Int64; samples::Int64 = 3)::Bool
    n, k = length(vars), length(G)
    G_sys = HC.InterpretedSystem(System(G ./ _row_scales(G, vars); variables = vars))
    rng = Random.Xoshiro(0x72656475)
    tested = 0
    for _ = 1:4samples
        x = _complex_newton_to_variety(G_sys, randn(rng, ComplexF64, n), k)
        x === nothing && continue
        _reduced_at(G_sys, x, k, c, rng) || return false
        tested += 1
        tested == samples && break
    end
    return true
end

# H and B: G and nothing when c = k; otherwise H = B G with B = Qᵀ D⁻¹, Q a random k × c
# matrix with orthonormal columns and D the coefficient norms of G (so that no equation
# dominates). JH = B JG. B uses the global random stream, like the centre of r.
function _square_up(G::Vector{Expression}, vars::Vector{Variable}, c::Int64)
    k = length(G)
    c == k && return G, nothing
    Q = Matrix(LA.qr(randn(k, c)).Q)        # k × c, orthonormal columns
    B = transpose(Q) ./ transpose(_row_scales(G, vars))
    H = Expression[sum(B[i, j] * G[j] for j = 1:k) for i = 1:c]
    return H, Matrix{Float64}(B)
end

function Base.show(io::IO, cache::RoutingCache)
    print(io, "RoutingCache: ", cache.r, " on V(G) ⊂ ℝ^", cache.n,
          " cut out by ", cache.k, " equation", cache.k == 1 ? "" : "s")
    cache.c == cache.k || print(io, " (codimension ", cache.c, ", squared up)")
end

routing_system(cache::RoutingCache)::System = cache.sys

# the jacobian of H, from the jacobian J of G at the same point
_jacobian_H(cache::RoutingCache, J::AbstractMatrix) = cache.B === nothing ? J : cache.B * J

# how far x is from V(G): the length of the rank-c Gauss-Newton step to it
function _distance_to_variety(cache::RoutingCache, x::Vector{Float64})::Float64
    G, J = evaluate_G!(cache, x)
    return LA.norm(_truncated_solve(J, G, cache.c))
end

"""
    evaluate_G!(cache, P) -> (G(P), JG(P))
    jacobian_G!(cache, P) -> JG(P)

`G` and its `k × n` jacobian at `P`. They are written into, and returned as, the
cache's own buffers, which the next call overwrites: `copy` them to keep them.
"""
function evaluate_G!(cache::RoutingCache, point::AbstractVector{Float64})
    HC.evaluate_and_jacobian!(cache.G_val, cache.JG_val, cache.G_sys, point)
    return cache.G_val, cache.JG_val
end

function jacobian_G!(cache::RoutingCache, point::AbstractVector{Float64})
    HC.jacobian!(cache.JG_val, cache.G_sys, point)
    return cache.JG_val
end

# whether JG has rank below c at the point P of V(G). a threshold on σ_c(JG) would
# depend on the scale of G, so compare σ_c at P with σ_c a short step away: it grows
# with the distance from the rank-drop locus. (σ_c rather than σ_min, which vanishes
# on all of V(G) when k > c.) the projection stops slightly short of the locus, so
# `step` must be well above that gap. stepping along P - r.c, generic because the
# centre is, keeps seeded runs reproducible.
function _rank_drops_at(
    cache::RoutingCache,
    P::AbstractVector{Float64};
    step::Float64 = 1e-3,
    ratio::Float64 = 0.1
    )

    c = cache.c
    σ = LA.svdvals(jacobian_G!(cache, P))
    length(σ) < c && return true
    # rank below c at P already; also covers G rank deficient everywhere, where the
    # ratio below would compare two zeros
    σ[c] <= c * eps(Float64) * σ[1] && return true

    u = P .- cache.r.c
    dist = LA.norm(u)
    dist == 0 && return false
    Q = P .+ (step * (1 + LA.norm(P)) / dist) .* u
    return σ[c] < ratio * LA.svdvals(jacobian_G!(cache, Q))[c]
end

"""
    project_to_variety!(P, cache)          # overwrites P
    project_to_variety(P, cache)           # returns a new point
    project_to_variety_residual!(P, cache) -> (P, ‖G(P)‖)

Moves `P` onto `V(G)` by damped Gauss-Newton steps, halving each until `‖G‖` decreases.
The residual version also returns the `‖G‖` reached, to tell failures apart.
"""
project_to_variety!(
    point::AbstractVector{Float64},
    cache::RoutingCache;
    maxiter::Int64 = 50,
    tol::Float64 = 1e-15,
    reg::Float64 = cache.reg,
    maxbacktrack::Int64 = 20,
) = first(_project_to_variety!(point, cache.G_sys, cache.G_val, cache.JG_val,
                               cache.M, cache.wk, cache.wn, cache.wn2,
                               maxiter, tol, reg, maxbacktrack))

project_to_variety(point::AbstractVector{<:Real}, cache::RoutingCache; kwargs...) =
    project_to_variety!(Vector{Float64}(point), cache; kwargs...)

project_to_variety_residual!(
    point::AbstractVector{Float64},
    cache::RoutingCache;
    maxiter::Int64 = 50,
    tol::Float64 = 1e-15,
    reg::Float64 = cache.reg,
    maxbacktrack::Int64 = 20,
) = _project_to_variety!(point, cache.G_sys, cache.G_val, cache.JG_val, cache.M,
                         cache.wk, cache.wn, cache.wn2, maxiter, tol, reg, maxbacktrack)

"""
    projected_gradient_field(cache; unit_speed = false) -> (field!, f_sys)

The in-place ODE right-hand side `field!(du, u, p, t)` of the projected gradient flow
of `r` on `V(G)` (PIPELINE.md §1.4). The sign of `p` ascends (`+1`) or descends (`-1`)
`|r|`, and its size rescales time. With `unit_speed` time is arc length. Also returns
the compiled numerator `f`, for callbacks on `V(f)`.
"""
projected_gradient_field(cache::RoutingCache; unit_speed::Bool = false) =
    (unit_speed ? cache.flow_unit! : cache.flow!), cache.r.f_sys

projected_gradient_field(
    r::RoutingFunction,
    G;
    reg::Real = 1e-8,
    unit_speed::Bool = false,
) = projected_gradient_field(RoutingCache(r, G; reg = reg); unit_speed = unit_speed)
