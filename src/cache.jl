# everything precomputed from (r, G): compiled systems, the routing system and its
# monodromy family, the gradient fields, scratch buffers. all symbolic work happens
# here. the buffers make a cache stateful: build one per thread.
"""
    RoutingCache(r, G; reg = 1e-8)

Everything the pipeline precomputes from a routing function `r` and the equations
`G` of the variety `X = V(G)`: the compiled systems for `G` and its derivatives, the
routing system whose solutions are the routing points, the two families monodromy
can run over, the two gradient fields, and scratch buffers. Build it once and pass it to
every routine; the `(r, G, ...)` convenience methods build a fresh one on each call.

`G` may be a vector of expressions, a single expression, or a `System`; it must use
only the variables of `r`, and must cut out `X` with a jacobian of full rank
`length(G)` at the points of `X` off the removed locus (reduced equations, as many
as the codimension). `reg` damps the normal equations `J Jᵀ` where the jacobian is
nearly rank deficient.

A cache holds mutable buffers: do not share one between threads.
"""
struct RoutingCache{F,Fu}
    r::RoutingFunction
    G::Vector{Expression}
    vars::Vector{Variable}
    n::Int64                        # ambient dimension
    k::Int64                        # number of defining equations of V(G)
    reg::Float64                    # regularisation of J Jᵀ in the flow fields and projections

    G_sys::HC.InterpretedSystem     # G; its jacobian is JG
    ∇G_sys::HC.InterpretedSystem    # [∇g₁; …; ∇gₖ]; its jacobian stacks the hessians of the gᵢ

    sys::System                     # routing system in (x, μ)
    sys_interp::HC.InterpretedSystem
    sys_vars::Vector{Variable}      # vcat(vars, μ)
    centre_sys::System              # sys with the centre and constant of g as parameters,
                                    # in (x, μ₀, μ̂) on the chart below
    chart::Vector{ComplexF64}       # the random chart ℓ₀μ₀ + ℓ·μ̂ = 1 of centre_sys
    param_sys::System               # sys minus generic affine-linear forms
    m::Int64                        # number of equations of sys
    N::Int64                        # n + k, the (x, μ) count

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
    reg::Real = 1e-8,
    )

    reg > 0 || throw(ArgumentError("reg must be positive, got $reg"))
    reg = Float64(reg)
    G = _as_expressions(G)
    isempty(G) && throw(ArgumentError("G is empty: give at least one equation"))
    vars = r.vars
    n = length(vars)
    k = length(G)

    # G has to live in r's variables (HC's own error for this is cryptic)
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

    sys = routing_system(r, G)
    sys_vars = variables(sys)
    m = length(expressions(sys))
    N = length(sys_vars)

    # the two families monodromy can run over (PIPELINE.md §3.2). the centre family
    # is the default; the affine family sys(z) - (Q z + q), z = (x, μ), with the
    # entries of Q and q as parameters and parameter 0 the routing system itself, is
    # much larger but transitive even when X is reducible.
    chart = randn(ComplexF64, k + 1)
    centre_sys = _centre_family(r, G, chart)
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
        r, G, vars, n, k, reg,
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


function Base.show(io::IO, cache::RoutingCache)
    print(io, "RoutingCache: ", cache.r, " on V(G) ⊂ ℝ^", cache.n,
          " cut out by ", cache.k, " equation", cache.k == 1 ? "" : "s")
end

routing_system(cache::RoutingCache)::System = cache.sys

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

"""
    project_to_variety!(P, cache)          # overwrites P
    project_to_variety(P, cache)           # returns a new point
    project_to_variety_residual!(P, cache) -> (P, ‖G(P)‖)

Moves `P` onto `V(G)` by damped Newton steps along the normal directions, backtracking
so that `‖G‖` decreases at every step. The residual version also reports the `‖G‖`
reached, so a point that never got there can be told apart.
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
of `r` on `V(G)` (see PIPELINE.md): `p` is the direction, `+1` to
ascend `|r|`, `-1` to descend, and its size rescales time. With `unit_speed` the
tangential part is normalised, so that time is arc length. Also returns the compiled
numerator `f`, for callbacks on the removed locus.
"""
projected_gradient_field(cache::RoutingCache; unit_speed::Bool = false) =
    (unit_speed ? cache.flow_unit! : cache.flow!), cache.r.f_sys

projected_gradient_field(
    r::RoutingFunction,
    G;
    reg::Real = 1e-8,
    unit_speed::Bool = false,
) = projected_gradient_field(RoutingCache(r, G; reg = reg); unit_speed = unit_speed)
