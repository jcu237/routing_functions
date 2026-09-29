# connecting routing points: from a routing point of positive index, follow the
# gradient flow of |r| up to the maxima (index 0 routing points) it reaches. by the
# mountain pass theorem, two maxima in the same component are joined this way
# through the index 1 points between them.

# minimum distance from `point` to the index 0 routing points. runs inside the ODE
# callback, so it does not allocate.
function distance_to_endpoints(
    point::AbstractVector{Float64},
    index0_points::Vector{Vector{Float64}}
    )::Float64

    best = Inf
    @inbounds for Q in index0_points
        s = 0.0
        for i in eachindex(point, Q)
            δ = Q[i] - point[i]
            s += δ * δ
        end
        s < best && (best = s)
    end
    return sqrt(best)
end

# index of the point of `points` nearest to `point`
function nearest_index(
    point::AbstractVector{Float64},
    points::Vector{Vector{Float64}}
    )::Int64

    best = Inf
    ind = 0
    @inbounds for (j, Q) in enumerate(points)
        s = 0.0
        for i in eachindex(point, Q)
            δ = Q[i] - point[i]
            s += δ * δ
        end
        if s < best
            best = s
            ind = j
        end
    end
    return ind
end

"""
    gradient_flow!(cache, P, targets; tol = 1e-2, kwargs...) -> Bool

Follows the unit-speed gradient flow of `|r|` on `V(G)` from `P` until it comes within
`tol` of one of the points `targets` (the index 0 routing points). Overwrites `P` with
where the flow stopped and returns whether it arrived.

Keywords: `dtmax` (largest step, default `tol`), `tspan = (0.0, 1e3)` (arc length),
`maxiters = 10^6`, `origin` (the critical point the flow is leaving, see below),
`verbose`.
"""
function gradient_flow!(
    cache::RoutingCache,
    point::Vector{Float64},
    index0_points::Vector{Vector{Float64}};
    tol::Real = 1e-2,
    dtmax::Real = 0.0,
    tspan::Tuple{Real,Real} = (0.0, 1e3),
    maxiters::Integer = 10^6,
    origin::Union{Nothing,AbstractVector{Float64}} = nothing,
    verbose::Bool = false,
    Verbose::Union{Nothing,Bool} = nothing
    )::Bool

    verbose = _verbose(verbose, Verbose, :gradient_flow!)
    isempty(index0_points) && return false
    tol = Float64(tol)
    dtmax = dtmax > 0 ? Float64(dtmax) : tol

    # the arrival callback fires on a sign change, so a start already inside a ball
    # would never trigger it. such a start counts as arrived, unless the critical
    # point it is leaving (`origin`) is nearer still: then shrink the balls instead.
    d0 = distance_to_endpoints(point, index0_points)
    if d0 <= tol
        if origin === nothing || d0 < LA.norm(point - origin)
            verbose && println("gradient_flow!: start point is already within tol of an endpoint")
            return true
        end
        tol = d0 / 2
    end

    # time is arc length, so this fires when the path first comes within tol of a
    # target; the callback also checks interpolated points inside each step
    arrived(u, t, integrator) = distance_to_endpoints(u, index0_points) - tol
    callback = SciMLBase.ContinuousCallback(arrived, SciMLBase.terminate!)

    prob = SciMLBase.ODEProblem(cache.flow_unit!, copy(point), (Float64(tspan[1]), Float64(tspan[2])), 1.0)
    sol = _quiet() do
        SciMLBase.solve(prob, reltol = 1e-8, abstol = 1e-8, dtmax = dtmax,
                        maxiters = maxiters, callback = callback)
    end

    point .= last(sol.u)
    arrived_at_endpoint = sol.retcode == SciMLBase.ReturnCode.Terminated

    if verbose
        println(
            "gradient_flow!: ", sol.retcode, ", arc length ", round(last(sol.t); digits = 4),
            ", distance to nearest index 0 point ",
            round(distance_to_endpoints(point, index0_points); digits = 6),
        )
    end

    return arrived_at_endpoint
end

gradient_flow!(
    r::RoutingFunction,
    G,
    point::Vector{Float64},
    index0_points::Vector{Vector{Float64}};
    reg::Real = 1e-8,
    kwargs...,
) = gradient_flow!(RoutingCache(r, G; reg = reg), point, index0_points; kwargs...)

"""
    find_starting_points_for_flow(cache, P, H, V; step_size = 0.1, max_halvings = 30)

Points from which to leave the critical point `P`: `P ± εv`, projected onto `V(G)`, for
each unstable eigenvector `v` of the hessian `H` of `r|V(G)` (written in the tangent
basis `V`). Unstable means eigenvalue `μ` with `sign(μ) = sign(r(P))`, so that `|r|`
increases along `v`.

`ε` starts at `step_size` and is halved until the step can be trusted to stay in `P`'s
own region of `V(G) ∖ V(f)`: `r` keeps its sign, and `|r(Q)| − |r(P)|` is within a
factor 2 of the quadratic model `½|μ|ε²` at both `ε` and `ε/2`.
"""
function find_starting_points_for_flow(
    cache::RoutingCache,
    critical_point::Vector{Float64},
    H::AbstractMatrix{Float64}, # hessian of r|X at critical_point
    V::AbstractMatrix{Float64}; # orthonormal basis of the tangent space it is written in
    step_size::Real = 0.1,
    max_halvings::Integer = 30
    )

    # why the model test: a routing point can sit within 1e-4 of V(f) (the 27 lines
    # examples have them), and a fixed step of 0.1 then lands in the neighbouring
    # region, joining two regions that are not connected. the sign test catches a
    # step across a hypersurface; the model test also catches a step through an
    # isolated point of V(f) ∩ X (a removed singular point, as on the Chubs surface),
    # where r keeps its sign. one scale alone can agree by coincidence, hence two.
    r = cache.r
    P = critical_point
    r0 = evaluate_r(r, P)
    sgn = sign(r0)
    E = LA.eigen(LA.Symmetric(Matrix(H)))

    # the start point at ε, and whether it passes the sign and model tests
    function trial(v, ε, μ)
        Q = project_to_variety!(P + ε .* v, cache)
        all(isfinite, Q) || return Q, false, false
        rQ = evaluate_r(r, Q)
        same_side = sign(rQ) == sgn && abs(rQ) > abs(r0)
        ρ = (abs(rQ) - abs(r0)) / (abs(μ) * ε^2 / 2)
        return Q, same_side, same_side && 0.5 <= ρ <= 2
    end

    starts = Vector{Float64}[]
    for i in eachindex(E.values)
        μ = E.values[i]
        sign(μ) == sgn || continue
        v = V * view(E.vectors, :, i)
        for σ in (1.0, -1.0)
            ε = Float64(step_size)
            Q, side, model = trial(σ .* v, ε, μ)
            fallback = side ? Q : nothing
            chosen = nothing
            for _ = 1:max_halvings
                # below this scale ½|μ|ε² is lost in the rounding of |r| (a nearly
                # degenerate critical point): stop, and use the smallest step that
                # kept the sign of r and increased |r|
                abs(μ) * (ε / 2)^2 / 2 > 1e3 * eps() * abs(r0) || break
                Q2, side2, model2 = trial(σ .* v, ε / 2, μ)
                side2 && (fallback = Q2)
                if model && model2
                    chosen = Q
                    break
                end
                ε /= 2
                Q, side, model = Q2, side2, model2
            end
            chosen === nothing && (chosen = fallback)
            if chosen === nothing
                @warn(
                    "find_starting_points_for_flow: every step off the critical point along " *
                    "an unstable direction left its region; that path is skipped",
                    critical_point = P, eigenvalue = μ,
                )
            else
                push!(starts, chosen)
            end
        end
    end
    return starts
end

"""
    solve_ivp(cache, P, final_points; kwargs...) -> Vector{Vector{Float64}}

Leaves the routing point `P` along each unstable direction (two paths per unit of
index) and follows the gradient flow of `|r|` until it arrives near one of
`final_points` (the index 0 routing points). Returns where the paths that arrived
ended; paths that never arrive are dropped.

`|r|` increases along the flow, so a path never crosses `V(f)` and ends in `P`'s
region, at a point where `r` has the sign of `r(P)`: only those points of
`final_points` are offered as destinations.

Keywords: `start_step_size = 0.1` and `max_halvings = 30` (see
[`find_starting_points_for_flow`](@ref)); `tol = 1e-2` (arrival radius) and
`grad_step_size` (largest step, default `tol`), plus `tspan` and `maxiters` (see
[`gradient_flow!`](@ref)); `verbose`.
"""
function solve_ivp(
    cache::RoutingCache,
    initial_point::Vector{Float64},
    final_points::Vector{Vector{Float64}};
    grad_step_size::Real = 0.0,
    start_step_size::Real = 0.1,
    max_halvings::Integer = 30,
    tol::Real = 1e-2,
    tspan::Tuple{Real,Real} = (0.0, 1e3),
    maxiters::Integer = 10^6,
    verbose::Bool = false,
    Verbose::Union{Nothing,Bool} = nothing
    )::Vector{Vector{Float64}}

    verbose = _verbose(verbose, Verbose, :solve_ivp)

    sgn = sign(evaluate_r(cache.r, initial_point))
    targets = [Q for Q in final_points if sign(evaluate_r(cache.r, Q)) == sgn]

    H, V = hessian_and_tangent(cache, initial_point)
    starts = find_starting_points_for_flow(cache, initial_point, H, V; step_size = start_step_size,
                                           max_halvings = max_halvings)

    solns = Vector{Float64}[]
    for P in starts
        Q = copy(P)
        if gradient_flow!(cache, Q, targets; tol = tol, dtmax = grad_step_size, tspan = tspan,
                          maxiters = maxiters, origin = initial_point, verbose = verbose)
            push!(solns, Q)
        elseif verbose
            println("solve_ivp: flow from ", round.(P; digits = 6), " reached no index 0 routing point")
        end
    end

    return solns
end

solve_ivp(
    r::RoutingFunction,
    G,
    initial_point::Vector{Float64},
    final_points::Vector{Vector{Float64}};
    kwargs...,
) = solve_ivp(RoutingCache(r, G), initial_point, final_points; kwargs...)
