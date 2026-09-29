# finding the routing points -- the critical points of r on X = V(G) off V(f).
#
# two sources, combined in `routing_points`:
#   1. gradient flows (`flow_to_routing_points`): from starting points on X, follow
#      the gradient of |r| up (to maxima) and down (to minima of |r|, or into V(f))
#      and Newton-refine where each flow ends. they find real routing points
#      directly, but only those some flow happens to reach.
#   2. monodromy on the routing system: every routing point is an isolated solution
#      of the polynomial system `routing_system`, and HomotopyContinuation's
#      `monodromy_solve` collects the complex solutions of the routing system by
#      walking loops in the parameter space of a family containing it -- by default
#      the family obtained by moving the centre c and the constant of g.

# `stop_when` is a predicate on a real point of ℝⁿ; a solution z = (x, μ) of the
# routing system is cut down to x and checked for realness first.
function _satisfies(stop_when, z::AbstractVector{<:Number}, n::Int64)::Bool
    @inbounds for i = 1:n
        abs(imag(z[i])) < IMAG_TOL || return false
    end
    return stop_when(real.(view(z, 1:n)))::Bool
end

# whether a solution of the routing system (or just its x part) is a routing point:
# real, off V(f) (distance to V(f) against locus_tol, see `on_zero_locus`), and a
# critical point of r|V(G) (Newton step to the critical point against crit_tol, see
# `_criticality`). the last is a safety net: Newton's method and homotopy endgames
# can report success at a point merely close to a solution.
function _is_routing_point(
    cache::RoutingCache,
    z::AbstractVector{<:Number};
    locus_tol::Real = 1e-6,
    crit_tol::Real = 1e-6
    )::Bool

    n = cache.n
    @inbounds for i = 1:n
        (isfinite(z[i]) && abs(imag(z[i])) < IMAG_TOL) || return false
    end
    x = Float64[real(z[i]) for i = 1:n]
    on_zero_locus(cache.r, x; locus_tol = locus_tol) && return false
    return _is_critical(cache, x; crit_tol = Float64(crit_tol))
end

# the callback that ends a seeding flow once it comes within locus_tol·(1 + ‖u‖) of
# V(f), measured by the Newton distance |f|/‖∇f‖ (a test on |f| alone depends on
# the scale of f). r changes sign across V(f), so the field is discontinuous there
# and a flow heading into V(f) can only end on the removed locus.
function _near_zero_locus_callback(r::RoutingFunction, n::Int64, locus_tol::Float64)
    f_cb = zeros(Float64, 1)
    ∇f_cb = zeros(Float64, n)
    function near_zero_locus(u, t, integrator)
        # a path running off to infinity makes f NaN, and `HC.evaluate!` throws an
        # InexactError writing it to a Float64 buffer: stop the path instead
        all(isfinite, u) || return true
        try
            HC.evaluate!(f_cb, r.f_sys, u)
            HC.evaluate!(∇f_cb, r.∇f_sys, u)
        catch err
            err isa InexactError || rethrow()
            return true
        end
        return _newton_distance(@inbounds(f_cb[1]), ∇f_cb) <= locus_tol * (1 + LA.norm(u))
    end
    return SciMLBase.DiscreteCallback(near_zero_locus, SciMLBase.terminate!)
end

# one seeding flow from x0 in direction dir (+1 up, -1 down), and the Newton
# refinement of where it ends. returns the solution (x, μ) of the routing system, or
# nothing.
#
# the ODE parameter is the direction divided by |r(x0)|: this rescales time so that
# the flow's speed does not depend on the size of r (multiplying f by 1e-8 would
# otherwise slow it down by the same factor).
function _flow_and_refine(cache::RoutingCache, x0::Vector{Float64}, dir::Float64, callback,
                          tspan::Tuple{Float64,Float64}, maxiters::Int64)
    r = cache.r
    r0 = abs(evaluate_r(r, x0))
    scale = (isfinite(r0) && r0 > 0) ? inv(r0) : 1.0
    prob = SciMLBase.ODEProblem(cache.flow!, copy(x0), tspan, dir * scale)
    sol = _quiet() do
        SciMLBase.solve(prob, reltol = 1e-8, abstol = 1e-8, maxiters = maxiters,
                        callback = callback)
    end
    x = last(sol.u)
    all(isfinite, x) || return nothing

    # μ by least squares from grad_num = JGᵀμ, then Newton on the routing system
    jacobian_G!(cache, x)
    μ = transpose(cache.JG_val) \ _grad_num(r, x)
    newton_result = HC.newton(cache.sys_interp, ComplexF64.(vcat(x, μ)))
    return HC.is_success(newton_result) ? HC.solution(newton_result) : nothing
end

# grad_num(x) = g ∇f - 2d f (x - c) = g^(d+1) ∇r, without forming powers of g
function _grad_num(r::RoutingFunction, x::Vector{Float64})::Vector{Float64}
    fx = zeros(Float64, 1)
    ∇f = zeros(Float64, length(x))
    HC.evaluate!(fx, r.f_sys, x)
    HC.evaluate!(∇f, r.∇f_sys, x)
    return _g(r, x) .* ∇f .- (2 * r.d * fx[1]) .* (x .- r.c)
end

# the points of `pts` with distinct x parts. x determines the multipliers, and these
# can be many orders of magnitude larger than x, so comparing whole solutions (x, μ)
# would compare only μ.
function _unique_by_x(pts::AbstractVector{<:AbstractVector}, n::Int64)
    isempty(pts) && return pts
    U = HC.UniquePoints(ComplexF64.(pts[1][1:n]), 1)
    keep = eltype(pts)[]
    for (i, p) in enumerate(pts)
        _, added = HC.add!(U, ComplexF64.(p[1:n]), i; atol = 1e-14, rtol = 1e-8)
        added && push!(keep, p)
    end
    return keep
end

# whether the x part of a solution is real
_real_x(z::AbstractVector, n::Int64) = all(i -> abs(imag(z[i])) < IMAG_TOL, 1:n)

"""
    flow_to_routing_points(cache; kwargs...) -> Vector{Vector{Float64}}

Routing points found by gradient flow alone. Starting points are drawn uniformly from
`[-box, box]ⁿ` (or taken from `starts`) and projected onto `V(G)`; from each, the
gradient flow of `|r|` on `V(G)` is followed up and down, and where it ends is refined
by Newton's method on the routing system. Returns the routing points found, without
repeats. Much cheaper than [`routing_points`](@ref), which adds monodromy, but it only
finds the routing points some flow reaches -- mostly maxima and minima of `|r|`,
rarely saddles.

Keywords:
* `nstarts = 100`: starting points that must land on `V(G)`;
* `box = 3.0`: half-width of the sampling box. Too small misses a far-away `V(G)`;
* `max_attempts = 20`: sampling budget, as a multiple of `nstarts`;
* `starts`: your own starting points (vectors of length `n`), each tried once, instead
  of random ones -- points near `V(f)` find small regions best;
* `proj_tol = 1e-8`: how small `‖G‖` must get for a start to count as on `V(G)`;
* `tspan = (0.0, 200.0)`, `maxiters = 10_000`: limits of each ODE integration (time is
  rescaled so that it does not depend on the scale of `r`);
* `locus_tol = 1e-6`, `crit_tol = 1e-6`: see [`routing_points`](@ref);
* `stop_when`: a predicate on a routing point; the first one satisfying it is
  returned at once;
* `all_vars = false`: return the solutions `(x, μ)` of the routing system, with the
  multipliers (see [`routing_system`](@ref));
* `verbose = false`.
"""
function flow_to_routing_points(cache::RoutingCache; all_vars::Bool = false, kwargs...)::Vector{Vector{Float64}}
    n = cache.n
    return [all_vars ? real.(z) : real.(z[1:n]) for z in _flow_seeds(cache; kwargs...)]
end

# the flow stage proper. returns the solutions (x, μ) of the routing system as
# complex vectors, the form monodromy takes its start solutions in. the starts that
# landed on V(G) are pushed to `landed`, if given.
function _flow_seeds(
    cache::RoutingCache;
    landed::Union{Nothing,Vector{Vector{Float64}}} = nothing,
    nstarts::Integer = 100,
    box::Real = 3.0,
    starts::Union{Nothing,AbstractVector} = nothing,
    tspan::Tuple{Real,Real} = (0.0, 200.0),
    maxiters::Integer = 10_000,
    locus_tol::Real = 1e-6,
    crit_tol::Real = 1e-6,
    proj_tol::Real = 1e-8,
    max_attempts::Integer = 20,
    stop_when::Union{Nothing,Function} = nothing,
    verbose::Bool = false,
    f_tol::Union{Nothing,Real} = nothing
    )::Vector{Vector{ComplexF64}}

    r = cache.r
    n = cache.n
    f_tol === nothing || _ignored_keyword(:f_tol,
        "a flow now stops once it comes within `locus_tol` (relative to the size of " *
        "the point) of V(f), a distance that does not depend on how f is scaled")
    box, proj_tol, locus_tol = Float64(box), Float64(proj_tol), Float64(locus_tol)
    tspan = (Float64(tspan[1]), Float64(tspan[2]))
    callback = _near_zero_locus_callback(r, n, locus_tol)

    # where the flows begin: uniform samples of [-box, box]^n, or the caller's own
    # points, each tried once. uniform samples find a component in proportion to
    # its size, so small components are best found from supplied starts near V(f).
    starts = _as_points(starts)
    supplied = starts !== nothing
    if supplied
        all(p -> length(p) == n, starts) || throw(ArgumentError(
            "every supplied start must have length $n, the number of variables r is written in",
        ))
    end
    wanted = supplied ? length(starts) : nstarts
    budget = supplied ? length(starts) : max_attempts * nstarts

    pts = Vector{ComplexF64}[]
    start_pt = zeros(Float64, n)
    started = 0
    attempts = 0

    while started < wanted && attempts < budget
        attempts += 1
        if supplied
            copyto!(start_pt, starts[attempts])
        else
            @inbounds for i = 1:n
                start_pt[i] = box * (2 * rand() - 1)
            end
        end

        # skip starts that did not reach V(G); `nstarts` counts starts on V(G)
        _, residual = project_to_variety_residual!(start_pt, cache)
        (isfinite(residual) && residual < proj_tol && all(isfinite, start_pt)) || continue
        started += 1
        landed === nothing || push!(landed, copy(start_pt))

        # up to the attracting critical points of |r|, down to the repelling ones
        for dir in (1.0, -1.0)
            z = _flow_and_refine(cache, start_pt, dir, callback, tspan, Int64(maxiters))
            # a flow that ended on V(f) refines to a solution that is no routing point
            (z !== nothing && _is_routing_point(cache, z; locus_tol = locus_tol,
                                                 crit_tol = crit_tol)) || continue
            push!(pts, z)

            # early exit; `routing_points` then skips monodromy too
            if stop_when !== nothing && _satisfies(stop_when, z, n)
                verbose && println(
                    "flow_to_routing_points: stop_when satisfied after $attempts attempts",
                )
                return [z]
            end
        end
    end

    if started < wanted
        @warn(supplied ?
              "flow_to_routing_points: only $started of the $wanted supplied starts lie on V(G)" :
              "flow_to_routing_points: only $started of $nstarts starts landed on V(G) in " *
              "$attempts attempts (box = $box). If V(G) lies far from the origin, raise " *
              "`box` or `max_attempts`. If no start lands at all, check that G cuts out V(G) " *
              "with a full-rank jacobian: reduced equations (no repeated factors), as many " *
              "as the codimension, and V(G) real.")
    elseif verbose
        println(supplied ?
                "flow_to_routing_points: all $wanted supplied starts lie on V(G)" :
                "flow_to_routing_points: $started starts landed on V(G) in $attempts attempts")
    end
    pts = _unique_by_x(pts, n)
    verbose && println("flow_to_routing_points: $(2 * started) flows found $(length(pts)) ",
                       "distinct routing points")
    return pts
end

"""
    routing_points(cache; kwargs...) -> Vector{Vector{Float64}}

The routing points -- the critical points of `r` on `V(G) \\ V(f)` -- as points of `ℝⁿ`.

Two stages. Gradient flows from random (or supplied) starting points find routing
points directly ([`flow_to_routing_points`](@ref)). They then seed monodromy on the
routing system ([`routing_system`](@ref)), which collects its complex solutions by
walking loops in the parameters of a family of such systems. A solution is kept when
it is real, off `V(f)` and a genuine critical point.

Two families are available (`monodromy_family`):
* `:centre` (default): the routing systems of `f / (‖x − c‖² + a)ᵈ` for all centres `c`
  and constants `a`. Monodromy runs directly at the actual `(c, 1)`, and only has to
  find the complex critical points of `r` -- much fewer solutions than the affine
  family. Its loops are sampled at the scale of `V(G)`, estimated from the starting
  points that landed on it. It needs one routing point (or one generic complex point
  of `V(G)`, which it constructs) on each irreducible component of `V(G)`.
* `:affine`: the routing system minus generic affine-linear functions of `(x, μ)`.
  Many more solutions to find, but it reaches all of them even when `V(G)` is
  reducible and the flows found nothing on some of its components.

The monodromy stage stops heuristically (after a few loops without new solutions), so
the result is as complete as that stage manages to be -- it is not certified.

Keywords (besides those of [`flow_to_routing_points`](@ref): `nstarts`, `box`,
`starts`, `proj_tol`, `max_attempts`, `stop_when`):
* `locus_tol = 1e-6`: a point within `locus_tol * (1 + ‖P‖)` of `V(f)` lies on the
  removed locus and is not a routing point (see [`on_zero_locus`](@ref)). Independent of
  the scale of `f`; regions thinner than about twice this are not resolved;
* `crit_tol = 1e-6`: a candidate must be within `crit_tol * (1 + ‖P‖)` of a critical
  point (the length of a Newton step; see [`critical_distance`](@ref));
* `flow_options = (;)`: options for the flows, e.g. `(tspan = (0.0, 500.0), maxiters = 20_000)`;
* `monodromy_family = :centre`: `:centre` or `:affine`, see above;
* `monodromy_options = (;)`: passed to HomotopyContinuation's `monodromy_solve`, e.g.
  `(timeout = 60,)` to bound the time monodromy takes (possibly missing routing points),
  or `(max_loops_no_progress = 20,)` to make it more thorough;
* `all_vars = false`: return the solutions `(x, μ)` including the multipliers;
* `verbose = false`.

With `stop_when`, a flow seed satisfying the predicate is returned at once, skipping
monodromy.
"""
function routing_points(
    cache::RoutingCache;
    all_vars::Bool = false,
    locus_tol::Real = 1e-6,
    crit_tol::Real = 1e-6,
    nstarts::Integer = 100,
    box::Real = 3.0,
    starts::Union{Nothing,AbstractVector} = nothing,
    proj_tol::Real = 1e-8,
    max_attempts::Integer = 20,
    stop_when::Union{Nothing,Function} = nothing,
    flow_options::NamedTuple = (;),
    monodromy_family::Symbol = :centre,
    monodromy_options::NamedTuple = (;),
    verbose::Bool = false,
    zero_tol::Union{Nothing,Real} = nothing
    )::Vector{Vector{Float64}}

    n = cache.n                     # ambient dimension, i.e. where ∇r lives
    family = _monodromy_family(monodromy_family)
    zero_tol === nothing || _ignored_keyword(:zero_tol,
        "whether a point lies on the removed locus V(f) is now decided by its " *
        "distance to V(f) (see `locus_tol`), which does not depend on how f is scaled")

    # real routing points from the flows, and where their starts landed on V(G)
    landed = Vector{Float64}[]
    seed_points = _flow_seeds(
        cache;
        landed = landed,
        nstarts = nstarts,
        box = box,
        starts = starts,
        locus_tol = locus_tol,
        crit_tol = crit_tol,
        proj_tol = proj_tol,
        max_attempts = max_attempts,
        stop_when = stop_when,
        verbose = verbose,
        flow_options...,
    )

    # a seed satisfying `stop_when` is the answer: skip monodromy
    if stop_when !== nothing
        hits = [p for p in seed_points if _satisfies(stop_when, p, n)]
        if !isempty(hits)
            verbose && println("routing_points: stop_when met by a flow seed; skipping monodromy")
            return [all_vars ? real.(p) : real.(p[1:n]) for p in hits]
        end
    end

    options = _monodromy_options(monodromy_options)
    sols, nstart, code = family === :centre ?
        _monodromy_centre(cache, seed_points, landed, options) :
        _monodromy_affine(cache, seed_points, options)
    code == :timeout && @warn(
        "routing_points: monodromy stopped at its timeout after finding " *
        "$(length(sols)) solutions; routing points may be missing",
    )
    verbose && println(
        "routing_points: $(length(seed_points)) flow seeds, $nstart start solutions, ",
        "$(length(sols)) solutions from monodromy over the $family family ($code)",
    )

    # keep the (Newton-refined) seeds even if monodromy loses them
    candidates = [real.(z) for z in vcat(sols, seed_points) if _real_x(z, n)]
    isempty(candidates) && return Vector{Float64}[]

    # the routing system also has real solutions on V(f); they are recognised by
    # their distance to V(f), not by |r|, which is tiny in thin or far-away regions
    pts = [p for p in _unique_by_x(candidates, n) if
           _is_routing_point(cache, p; locus_tol = locus_tol, crit_tol = crit_tol)]
    verbose && println("routing_points: $(length(pts)) routing points")

    return all_vars ? pts : [p[1:n] for p in pts]
end

function _monodromy_family(family::Symbol)::Symbol
    family in (:centre, :center) && return :centre
    family === :affine && return :affine
    throw(ArgumentError("monodromy_family must be :centre or :affine, got :$family"))
end

# the centre of a set of points and their spread (root mean square distance from the
# centre), at least 1. the scale at which the centre family's loops are sampled.
function _extent(pts::Vector{Vector{Float64}}, n::Int64)
    isempty(pts) && return zeros(Float64, n), 1.0
    centre = sum(pts) ./ length(pts)
    spread = sqrt(sum(p -> sum(abs2, p .- centre), pts) / length(pts))
    return centre, max(1.0, spread)
end

# loop nodes for the centre family (PIPELINE.md §3.2). HC's default draws every
# parameter from a standard normal distribution, i.e. loops of size 1 around the
# origin, which never reach the parts of a large variety far from the origin. here
# the centre c is drawn around `centre` at a scale σ that is log-uniform in
# [scale, spread · scale], and a at the scale σ² of ‖x − c‖².
function _centre_sampler(centre::Vector{Float64}, scale::Float64, spread::Float64)
    n = length(centre)
    return function (p)
        σ = scale * spread^rand()
        return vcat(centre .+ σ .* randn(ComplexF64, n), σ^2 * randn(ComplexF64))
    end
end

# loop nodes for the affine family (PIPELINE.md §3.2): its parameters (the coefficients of the
# shifts) are of size 10-500 on the examples, so loops are sampled at the size of the
# base point rather than at size 1.
_affine_sampler(p) = (LA.norm(p) / sqrt(length(p))) .* randn(ComplexF64, length(p))

# a random complex point of X near `centre`: minimum-norm Gauss-Newton steps
# x ← x - JGᴴ(JG JGᴴ)⁻¹ G(x) from a random complex point. nothing if it fails.
function _complex_point_on_variety(cache::RoutingCache, centre::Vector{Float64}, scale::Float64)
    n, k = cache.n, cache.k
    x = ComplexF64.(centre) .+ scale .* randn(ComplexF64, n)
    Gx = zeros(ComplexF64, k)
    J = zeros(ComplexF64, k, n)
    for _ = 1:100
        try
            HC.evaluate_and_jacobian!(Gx, J, cache.G_sys, x)
        catch err
            err isa InexactError || rethrow()
            return nothing
        end
        (all(isfinite, Gx) && all(isfinite, J)) || return nothing
        δ = J' * ((J * J') \ Gx)
        all(isfinite, δ) || return nothing
        x .-= δ
        LA.norm(δ) <= 1e-13 * (1 + LA.norm(x)) && return x
    end
    return nothing
end

# a start pair for the centre family from a point x0 of X: parameters (c, a) that
# make x0 a critical point, with multipliers μ. given random (s, ν), the choice
# u = x0 - c = (s ∇f - JGᵀν) / (2d f), a = s - uᵀu gives g(x0) = s and
# grad_num(x0) = s ∇f - 2d f u = JGᵀν, so μ = ν. (s, ν) is scaled so that ‖u‖ = scale.
function _centre_start_pair(cache::RoutingCache, x0::Vector{ComplexF64}, scale::Float64)
    r = cache.r
    n, k = cache.n, cache.k
    fx = zeros(ComplexF64, 1)
    ∇f = zeros(ComplexF64, n)
    J = zeros(ComplexF64, k, n)
    HC.evaluate!(fx, r.f_sys, x0)
    HC.evaluate!(∇f, r.∇f_sys, x0)
    HC.jacobian!(J, cache.G_sys, x0)
    s = randn(ComplexF64)
    ν = randn(ComplexF64, k)
    u = (s .* ∇f .- transpose(J) * ν) ./ (2 * r.d * fx[1])
    t = scale / LA.norm(u)
    (isfinite(t) && t > 0) || return nothing
    s *= t
    ν .*= t
    u .*= t
    return vcat(x0, ν), vcat(x0 .- u, s - sum(u .^ 2))
end

# monodromy over the centre family, at the routing system itself. start solutions:
# the flow seeds, plus six constructed from random complex points of X and tracked to
# the routing system -- enough to start when the flows found nothing, and a chance to
# reach components of a reducible X that the seeds miss.
#
# the loops are sampled in phases of growing size (`CENTRE_PHASES`): loops at the
# scale of X first, until they stop finding solutions, then larger ones from all the
# solutions found so far. large loops find solutions far out (the last Clebsch region,
# 3RPR), but they rarely find new ones, and mixed in from the start they end the
# search before the smaller loops have found everything (one region of 14 missed in 3
# of 40 runs on a sphere; 0 of 40 with phases). a timeout applies to all phases together.
#
# works in the chart coordinates (x, μ₀, μ̂) of `_centre_family`; returns the solutions
# found as (x, μ), the number of start solutions, and HC's return code.
function _monodromy_centre(cache::RoutingCache, seeds, landed, options::NamedTuple)
    n = cache.n
    p0 = ComplexF64.(vcat(cache.r.c, 1.0))
    centre, scale = _extent(vcat(landed, [real.(z[1:n]) for z in seeds]), n)
    on_chart(zs) = [w for w in (_to_chart(z, n, cache.chart) for z in zs) if w !== nothing]

    starts = on_chart(seeds)
    # the constructed starts begin at growing distances from the centre, so that they
    # can land on components of X other than the nearest
    for spread in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
        x0 = _complex_point_on_variety(cache, centre, spread * scale)
        x0 === nothing && continue
        pair = _centre_start_pair(cache, x0, scale)
        pair === nothing && continue
        z, p1 = pair
        w = _to_chart(z, n, cache.chart)
        w === nothing && continue
        res = HC.solve(cache.centre_sys, [w]; start_parameters = p1, target_parameters = p0,
                       show_progress = false)
        append!(starts, solutions(res))
    end
    starts = _unique_by_x(starts, n)
    isempty(starts) && return Vector{ComplexF64}[], 0, :no_start_solutions

    nstart = length(starts)
    # a sampler of the caller's replaces the phases
    phases = haskey(options, :parameter_sampler) ? (nothing,) : CENTRE_PHASES
    deadline = haskey(options, :timeout) ? time() + options.timeout : Inf
    sols, code = starts, :heuristic_stop
    for spread in phases
        remaining = deadline - time()
        remaining > 0 || (code = :timeout; break)
        phase = (max_loops_no_progress = 10,)
        spread === nothing || (phase = merge(phase, (parameter_sampler =
                                   _centre_sampler(centre, scale, spread),)))
        # the caller's options win, except for the time left
        phase = merge(phase, options)
        isfinite(deadline) && (phase = merge(phase, (timeout = remaining,)))
        mon = monodromy_solve(cache.centre_sys, sols, p0; show_progress = false, phase...)
        isempty(solutions(mon)) || (sols = solutions(mon))
        code = mon.returncode
        code == :timeout && break
    end
    return [z for z in (_from_chart(w, n) for w in sols) if z !== nothing], nstart, code
end

# the loop sizes of the centre family's monodromy phases, as multiples of the extent
# of X: in phase i the scale of the loops is log-uniform in [1, CENTRE_PHASES[i]] times
# the extent (see `_centre_sampler`)
const CENTRE_PHASES = (1.0, 10.0, 100.0)

# monodromy over the affine family: seeds and a random start pair are moved to a
# generic member, monodromy runs there, and its solutions are tracked back.
function _monodromy_affine(cache::RoutingCache, seeds, options::NamedTuple)
    N, m = cache.N, cache.m
    # a random point S0 and random linear coefficients Q0, with the constant column
    # chosen so that S0 solves the family over p0 = vec(Q0)
    S0 = randn(ComplexF64, N)
    Q0 = randn(ComplexF64, m, N + 1)
    Q0[:, N+1] .= HC.evaluate(cache.sys, S0) .- Q0[:, 1:N] * S0
    p0 = vec(Q0)
    p_target = zeros(ComplexF64, length(p0))

    # the seeds solve the family at p_target; track them onto the generic fibre
    starts = Vector{ComplexF64}[S0]
    if !isempty(seeds)
        res_seeds = HC.solve(cache.param_sys, seeds; start_parameters = p_target,
                             target_parameters = p0, show_progress = false)
        append!(starts, solutions(res_seeds))
    end
    starts = _unique_by_x(starts, cache.n)

    mon = monodromy_solve(cache.param_sys, starts, p0; show_progress = false,
                          merge((parameter_sampler = _affine_sampler,), options)...)

    # and the whole fibre back to the routing system itself
    res = HC.solve(cache.param_sys, solutions(mon); start_parameters = p0,
                   target_parameters = p_target, show_progress = false)
    return solutions(res; only_nonsingular = false), length(starts), mon.returncode
end

# HC wants `timeout` as a Float64; accept any real number of seconds
_monodromy_options(o::NamedTuple) =
    haskey(o, :timeout) && o.timeout isa Real ? merge(o, (timeout = Float64(o.timeout),)) : o

# convenience wrappers that build a fresh cache on every call
flow_to_routing_points(r::RoutingFunction, G; kwargs...) =
    flow_to_routing_points(RoutingCache(r, G); kwargs...)

routing_points(r::RoutingFunction, G; kwargs...) =
    routing_points(RoutingCache(r, G); kwargs...)
