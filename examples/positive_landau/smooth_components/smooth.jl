# Smoothly connected components of the Landau incidence variety Y of a Feynman diagram:
# the connected components of Y ∖ Sing(Y) over ℝ, each with its routing points, their
# Morse indices and its Euler characteristic.
#
# The same Y and singular locus as ../positive.jl (read from ../results/<name>.txt, written
# by ../<name>.m2), but nothing about positivity: the routing numerator f is only the
# product of the factors that remove the singular pieces of Y, without α₁⋯α_{E-1}·M.
# Only stratum 1 (Y itself) is routed. An irreducible component of Y that lies entirely in
# the singular locus is removed by f, and the report says so; its own smooth components
# are not computed.
#
# The y equation y·α₁⋯α_{E-1}·U = 1 already keeps every αᵢ away from 0 on Y, so dropping
# the α's from f does not change the components; dropping M can merge some.
#
#   julia -t auto --project=. examples/positive_landau/smooth_components/smooth.jl A3
#   julia -t auto --project=. examples/positive_landau/smooth_components/smooth.jl A3 --monodromy-timeout=3600
#
# The report goes to the screen and to smooth_components/results/<name>.txt. A first
# version is written as soon as the routing points are found, so a run stopped during
# the connectivity step still leaves them.

include(joinpath(@__DIR__, "..", "positive.jl"))

const SMOOTH_RESULTS_DIR = joinpath(@__DIR__, "results")

"""
    smooth_numerator(L, stratum) -> f

The product of the factors of the singular pieces of `stratum`, each once (up to a
constant), every piece included. `1` if there are none.
"""
function smooth_numerator(L::LandauSystem, stratum::Stratum)
    factors = Expression[]
    x1, x2 = randn(length(L.vars)), randn(length(L.vars))
    value(h, x) = Float64(real(evaluate(h, L.vars => x)))
    proportional(h, q) = abs(value(h, x1) * value(q, x2) - value(h, x2) * value(q, x1)) <
                         1e-10 * (abs(value(h, x1) * value(q, x2)) + abs(value(h, x2) * value(q, x1)))
    for piece in stratum.pieces
        any(q -> proportional(piece.factor, q), factors) || push!(factors, piece.factor)
    end
    return isempty(factors) ? Expression(1) : prod(factors)
end

"What was found on Y ∖ Sing(Y)."
struct SmoothResult
    f::Union{Nothing,Expression}        # nothing if no routing function was needed
    note::String
    points::Vector{Vector{Float64}}     # all routing points (real points, in dimension 0)
    components::Vector{Component}       # nothing yet if the connectivity step has not run
    connected::Bool                     # whether `components` is the result of the connectivity step
    seconds::Vector{Float64}            # routing points, connectivity
end

on_components(L, stratum, q) = [c.number for c in stratum.components if vanishes_at(c.generators, L.vars, q)]

function point_line(L, stratum, q, index)
    on = on_components(L, stratum, q)
    where = isempty(stratum.components) ? "" :
            isempty(on) ? "   on no listed component of Y" : "   on component " * join(on, ", ") * " of Y"
    return "index $index:  " * point_string(L, q) * where
end

function smooth_report(io::IO, L::LandauSystem, S::SingularLocus, R::SmoothResult)
    stratum = S.strata[1]
    println(io, L)
    println(io)
    println(io, "Y ∖ Sing(Y), dimension ", stratum.dim)
    println(io, R.f === nothing ? R.note : "f = $(R.f)")
    @printf(io, "%d routing points (%.0f s)", length(R.points), R.seconds[1])
    R.connected && @printf(io, ", connectivity %.0f s", R.seconds[2])
    println(io)

    if !R.connected
        println(io)
        println(io, "The connectivity step has not finished: the routing points, not yet sorted into components.")
        for q in R.points
            println(io, "   ", point_string(L, q))
        end
    else
        println(io)
        println(io, length(R.components), " smoothly connected component",
                length(R.components) == 1 ? "" : "s", ", Euler characteristic ",
                euler_characteristic(R.components), " in all")
        for (k, C) in enumerate(R.components)
            on = unique(reduce(vcat, (on_components(L, stratum, q) for q in C.points); init = Int[]))
            println(io)
            println(io, "component $k: χ = $(C.euler_characteristic), $(length(C.points)) routing point",
                    length(C.points) == 1 ? "" : "s",
                    isempty(stratum.components) ? "" :
                    isempty(on) ? ", on no listed component of Y" : ", on component " * join(sort(on), ", ") * " of Y")
            any(==(0), C.indices) ||
                println(io, "   no index 0 routing point: a path did not arrive, and this component is",
                        " probably part of another one")
            for j in sortperm(C.indices)
                println(io, "   ", point_line(L, stratum, C.points[j], C.indices[j]))
            end
        end
    end

    # what f removes
    println(io)
    if isempty(stratum.components)
        println(io, "The singular locus of Y is not known and was not removed.")
    else
        println(io, "the irreducible components of Y:")
        for c in stratum.components
            verdict = c.number in S.unresolved ?
                "lies in the singular locus (removed; Macaulay2 found no equations for it); smooth components not computed" :
                c.degenerate ? "lies in the singular locus (removed); smooth components not computed" : "regular"
            println(io, "   ", c.number, ". dim ", c.dim, ", degree ", c.degree, ": ", verdict)
            println(io, "      ", generators_string(c.generators))
        end
        if !isempty(stratum.pieces)
            println(io, "the singular pieces removed by f:")
            for piece in stratum.pieces
                println(io, "   ", piece.number, ". dim ", piece.dim, ": ", generators_string(piece.generators))
            end
        end
    end
    println(io)
    println(io, "Components are complete only if every routing point was found, and the search for them")
    println(io, "stops heuristically. Each routing point has its Morse index, and χ = ∑ (-1)^index.")
    return nothing
end

function save_report(L, S, R)
    mkpath(SMOOTH_RESULTS_DIR)
    open(io -> smooth_report(io, L, S, R), joinpath(SMOOTH_RESULTS_DIR, "$(L.name).txt"), "w")
end

"""
    smooth_components(name; seed = 1, nstarts = 200, box = 10.0, max_attempts = 20,
                      save = true, kwargs...)

The smoothly connected components of `Y` for the diagram `name` of diagrams.jl, from
the singular locus in ../results/<name>.txt. The other keywords go to
`find_connectivity_matrix` (and through it to `routing_points`), e.g.
`monodromy_options = (timeout = 3600,)`.
"""
function smooth_components(name::Symbol; seed::Integer = 1, nstarts::Integer = 200, box::Real = 10.0,
                           max_attempts::Integer = 20, save::Bool = true, kwargs...)
    L = LandauSystem(name)
    S = read_singular_locus(L)
    stratum = S.strata[1]
    G = [stratum.equations; L.y_equation]
    Random.seed!(seed)

    if !isempty(stratum.components) && all(c -> c.degenerate, stratum.components)
        R = SmoothResult(nothing, "every component of Y lies in the singular locus, so nothing is left to route on",
                         Vector{Float64}[], Component[], true, [0.0, 0.0])
    elseif length(G) == length(L.vars)
        t = @elapsed result = solve(System(G; variables = L.vars); show_progress = false)
        nsingular(result) == 0 || @warn("$(L.name): $(nsingular(result)) singular solutions were left out")
        points = real_solutions(result)
        R = SmoothResult(nothing, "finitely many points, so the system is solved directly; each is a component",
                         points, [Component([q], [0]) for q in points], true, [t, 0.0])
    else
        f = smooth_numerator(L, stratum)
        cache = RoutingCache(RoutingFunction(f, L.vars), G)
        t1 = @elapsed begin
            starts = starting_points(L, cache; nstarts = nstarts, box = box, max_attempts = max_attempts)
            path_keys = (:grad_step_size, :start_step_size, :max_halvings, :tol, :path_options, :verbose)
            search = (; (k => v for (k, v) in pairs(kwargs) if !(k in path_keys))...)
            points = routing_points(cache; starts = starts, search...)
        end
        @printf("%d routing points, %.0f s\n", length(points), t1)
        R = SmoothResult(f, "", points, Component[], false, [t1, 0.0])
        save && save_report(L, S, R)

        paths = (; (k => v for (k, v) in pairs(kwargs) if k in path_keys)...)
        t2 = @elapsed components = connected_components(cache, points; paths...)
        @printf("%d components, %.0f s\n", length(components), t2)
        R = SmoothResult(f, "", points, components, true, [t1, t2])
    end

    println()
    smooth_report(stdout, L, S, R)
    save && save_report(L, S, R)
    return (system = L, singular_locus = S, result = R)
end

# run as a script: smooth.jl NAME [--monodromy-timeout=SECONDS]
if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: smooth.jl NAME [--monodromy-timeout=SECONDS]")
    options = script_options(ARGS[2:end])
    haskey(options, :singular_locus) && error("smooth.jl needs the singular locus: --no-singular-locus is not allowed")
    smooth_components(Symbol(ARGS[1]); options...)
end
