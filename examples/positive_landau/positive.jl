# Positive Landau discriminants: which irreducible components of the incidence variety Y
# of a Feynman diagram contain a point with α real and positive and a positive mass ratio
# M/m? This is the computation of examples/positive_landau.jl, for any diagram of
# diagrams.jl:
#
#   L = LandauSystem(:acn)              # the Landau equations, from Landau.jl
#   S = read_singular_locus(L)          # written by acn.m2: components and singular locus
#   positive_points(L, S.strata[1])     # routing points of r = f/gᵈ on Y, and the positive ones
#
# or all of it at once, which is what the <name>.jl scripts do:
#
#   positive_landau(:acn; nstarts = 200, box = 10.0)
#
# The routing numerator is f = α₁⋯α_{E-1}·M times one factor for each piece of the
# singular locus of Y. The first factors keep the signs of α and M constant on every
# connected component of V(G) ∖ V(f), and every such component contains a routing point,
# so every positive piece of Y off its singular locus gets one. The other factors are
# there because ConnectedComponents needs V(G) to be smooth of the right dimension off
# V(f); Macaulay2 chooses them (see singular_locus.m2).
#
# What the routing points cannot say anything about is the singular locus itself, which
# f removes. Macaulay2 sorts that out as far as it can: for each piece it reports whether
# it has no positive points (a generator of one sign), is a positive rational point, or
# is undecided; and a whole component of Y inside the singular locus that could have
# positive points becomes a stratum of its own, with its own equations and numerator.

using ConnectedComponents
using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "diagrams.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "results")

# ---------------------------------------------------------------------------------------
# what Macaulay2 found
# ---------------------------------------------------------------------------------------
"A piece of the singular locus of a stratum, and the factor of `f` that removes it."
struct Piece
    number::Int
    dim::Int
    positivity::Symbol                      # :none, :some or :unknown
    point::Union{Nothing,Vector{Float64}}   # the piece, when it is a positive rational point (in `vars`, y included)
    factor::Expression
    generators::Vector{Expression}
end

"An irreducible component of Y. `degenerate`: the jacobian of the Landau equations is rank deficient on all of it."
struct YComponent
    number::Int
    dim::Int
    degree::Int
    degenerate::Bool
    positivity::Symbol                      # :none (a generator has one sign on the positive points) or :unknown
    generators::Vector{Expression}
end

"""
A variety to run the routing function on: `equations` together with the y equation.
Stratum 1 is Y itself, with all its components. Any further one is a component of Y that
the jacobian of the Landau equations is rank deficient on, cut out (possibly with other
components, which are of no interest) by equations whose jacobian is not.
"""
struct Stratum
    number::Int
    label::String
    dim::Int
    equations::Vector{Expression}
    components::Vector{YComponent}
    pieces::Vector{Piece}
end

struct SingularLocus
    strata::Vector{Stratum}
    unresolved::Vector{Int}                 # degenerate components Macaulay2 found no equations for
end

# A polynomial printed by Macaulay2, e.g. "2*α_2*s-8*α_2+M-5" or "(1/2)*α_1^2", as an
# Expression. Numbers stay exact rationals until they meet a variable.
function m2_polynomial(str::AbstractString, env::Dict{Symbol,Expression})::Expression
    value = _m2_value(Meta.parse(str), env)
    return value isa Expression ? value : _m2_expression(value)
end

_m2_expression(q::Rational{BigInt}) =
    isone(denominator(q)) ? Expression(numerator(q)) : Expression(numerator(q)) / Expression(denominator(q))
_m2_expression(e::Expression) = e

_m2_value(x::Integer, _) = big(x) // 1
_m2_value(x::Symbol, env) = haskey(env, x) ? env[x] : error("unknown variable $x in a Macaulay2 polynomial")
function _m2_value(ex::Expr, env)
    # an integer too large for Int64 is parsed as a call of @int128_str or @big_str
    ex.head === :macrocall && return parse(BigInt, ex.args[end]) // 1
    ex.head === :call || error("cannot read $ex as a polynomial")
    op = ex.args[1]
    args = [_m2_value(a, env) for a in ex.args[2:end]]
    if op === :^
        return args[1]^Int(args[2])
    elseif all(a -> a isa Rational, args)
        op === :+ && return sum(args)
        op === :- && return length(args) == 1 ? -args[1] : args[1] - args[2]
        op === :* && return prod(args)
        op === :/ && return args[1] / args[2]
    else
        args = _m2_expression.(args)
        op === :+ && return sum(args)
        op === :- && return length(args) == 1 ? -args[1] : args[1] - args[2]
        op === :* && return prod(args)
        op === :/ && return args[1] / args[2]
    end
    error("cannot read $ex as a polynomial")
end

# The value of g at q, and the size its terms can have at a point of that size. The ratio
# is about 1e-16 where g vanishes. (The terms themselves would not do as a scale: a
# generator like `s` has a single term, which is as small as its value.)
function value_and_scale(g::Expression, vars::Vector{Variable}, q::AbstractVector{<:Real})
    E, C = exponents_coefficients(expand(g), vars)
    value, scale = 0.0, 0.0
    for j in eachindex(C)
        c = Float64(real(C[j]))
        value += c * prod(q[i]^E[i, j] for i in eachindex(vars))
        scale += abs(c) * prod(max(abs(q[i]), 1.0)^E[i, j] for i in eachindex(vars))
    end
    return value, scale
end

# whether q lies on the zero set of the generators `gens`
function vanishes_at(gens, vars::Vector{Variable}, q::AbstractVector{<:Real}; tol::Real = 1e-6)
    return all(gens) do g
        value, scale = value_and_scale(g, vars, q)
        abs(value) <= tol * scale
    end
end

"""
    read_singular_locus(L; file = results/<name>.txt)

The components of `Y` and its singular locus, as written by `<name>.m2`. Checks that
Macaulay2 worked with the equations of `L`.
"""
function read_singular_locus(L::LandauSystem; file::AbstractString = joinpath(RESULTS_DIR, "$(L.name).txt"))
    isfile(file) || error("$file not found. Run Macaulay2 first:\n" *
                          "    M2 --script examples/positive_landau/$(L.name).m2")
    env = Dict{Symbol,Expression}(Symbol(n) => Expression(v) for (n, v) in zip(m2_names(L), m2_vars(L)))
    poly(str) = m2_polynomial(str, env)
    polys(str) = [poly(strip(p)) for p in split(str, ";")]
    stale() = error("$file does not belong to the equations of diagrams.jl. Run " *
                    "generate_equations.jl and $(L.name).m2 again.")

    strata = Stratum[]
    unresolved = Int[]
    done = false
    for line in eachline(file)
        (isempty(strip(line)) || startswith(line, "#")) && continue
        fields = strip.(split(line, " | "))
        key = fields[1]
        if key == "diagram"
            fields[2] == string(L.name) || stale()
        elseif key == "ring"
            split(fields[2]) == m2_names(L) || stale()
        elseif key == "stratum"
            push!(strata, Stratum(parse(Int, fields[2]), fields[3], parse(Int, fields[4]),
                                  Expression[], YComponent[], Piece[]))
        elseif key == "equation"
            push!(strata[end].equations, poly(fields[2]))
        elseif key == "component"
            push!(strata[end].components,
                  YComponent(parse(Int, fields[2]), parse(Int, fields[3]), parse(Int, fields[4]),
                             fields[5] == "degenerate", Symbol(fields[6]), polys(fields[7])))
        elseif key == "piece"
            gens = polys(fields[7])
            positivity, point = _piece_positivity(L, fields[4])
            factor = fields[5] == "squares" ? sum(g^2 for g in gens) : poly(fields[6])
            push!(strata[end].pieces, Piece(parse(Int, fields[2]), parse(Int, fields[3]),
                                            positivity, point, factor, gens))
        elseif key == "unresolved"
            push!(unresolved, parse(Int, fields[2]))
        elseif key == "done"
            done = true
        else
            error("$file: cannot read the line\n    $line")
        end
    end
    done || error("$file is incomplete: $(L.name).m2 did not finish")

    # the first stratum is cut out by the Landau equations themselves
    x = randn(length(L.vars))
    same = length(strata[1].equations) == length(L.landau) &&
           all(zip(strata[1].equations, L.landau)) do (a, b)
               (va, scale), (vb, _) = value_and_scale(a, L.vars, x), value_and_scale(b, L.vars, x)
               abs(va - vb) <= 1e-10 * scale
           end
    same || stale()
    return SingularLocus(strata, unresolved)
end

"""
    unknown_singular_locus(L)

What to use when `<name>.m2` cannot finish: `Y` as the only stratum, with no components
and no singular pieces, so that `f = α₁⋯α_{E-1}·M`. The positive routing points found are
still positive points of `Y`. What is lost is the guarantee that every positive piece gets
one, since `V(G)` may be singular off `V(f)`, and the list of components.
`flow_to_routing_points` warns when starting points land where the jacobian is rank
deficient.
"""
unknown_singular_locus(L::LandauSystem) =
    SingularLocus([Stratum(1, "Y, singular locus not removed", expected_dimension(L),
                           copy(L.landau), YComponent[], Piece[])], Int[])

# "none", "unknown", or "some: 1/2 1/2 1 0 9": a positive rational point, in the variables of
# the Macaulay2 ring. y is put back in from the y equation.
function _piece_positivity(L::LandauSystem, field::AbstractString)
    startswith(field, "some:") || return Symbol(field), nothing
    rational(v) = (parts = parse.(BigInt, split(v, "/")); parts[1] // get(parts, 2, big(1)))
    values = [Float64(rational(v)) for v in split(field[6:end])]
    point = zeros(length(L.vars))
    point[indexin(m2_vars(L), L.vars)] .= values
    point[findfirst(==(L.y), L.vars)] = y_on_variety(L, point)
    return :some, point
end

# ---------------------------------------------------------------------------------------
# the routing function
# ---------------------------------------------------------------------------------------
is_positive(L::LandauSystem, q::AbstractVector{<:Real}) = all(>(0), q[L.positive])

# y = 1/(α₁⋯α_{E-1}·U) at the point q, as the y equation has it
function y_on_variety(L::LandauSystem, q::AbstractVector{<:Real})::Float64
    g = Float64(real(evaluate(prod(L.α) * L.U, L.vars => q)))
    return 1 / g
end

"""
    routing_numerator(L, stratum; remove_positive_free = true) -> (f, factors)

`f = α₁⋯α_{E-1}·M` times a factor for each piece of the singular locus of `stratum`. The
first factors keep the signs of `α` and `M` constant on each connected component of
`V(G) ∖ V(f)`; the others remove the points where the jacobian of the equations drops
rank. With `remove_positive_free = false` the pieces without positive points are left in.
No positive piece of the variety can reach them, so the positive routing points stay
meaningful and `f` has lower degree; but starting points can land there, where the
jacobian is rank deficient.
"""
function routing_numerator(L::LandauSystem, stratum::Stratum; remove_positive_free::Bool = true)
    factors = Expression[Expression(L.vars[i]) for i in L.positive]
    x1, x2 = randn(length(L.vars)), randn(length(L.vars))
    value(h, x) = Float64(real(evaluate(h, L.vars => x)))
    proportional(h, q) = abs(value(h, x1) * value(q, x2) - value(h, x2) * value(q, x1)) <
                         1e-10 * (abs(value(h, x1) * value(q, x2)) + abs(value(h, x2) * value(q, x1)))
    for piece in stratum.pieces
        piece.positivity === :none && !remove_positive_free && continue
        any(q -> proportional(piece.factor, q), factors) || push!(factors, piece.factor)
    end
    return prod(factors), factors
end

"""
    starting_points(L, cache; nstarts = 200, box = 10.0, max_attempts = 20)

`nstarts` points of the variety of `cache` reached from the positive part of the box
`[-box, box]ⁿ` (`α > 0` and `M > 0`), and `nstarts` more from the whole box. The first
half puts the gradient flows where the positive points are; the second gives monodromy
a start on components that have none. Each sample gets the `y` of the y equation and is
projected onto the variety. A sample that does not arrive is replaced, up to
`max_attempts * nstarts` samples for each half.
"""
function starting_points(L::LandauSystem, cache::RoutingCache; nstarts::Integer = 200,
                         box::Real = 10.0, max_attempts::Integer = 20)
    n = length(L.vars)
    iy = findfirst(==(L.y), L.vars)
    starts = Vector{Float64}[]
    for positive_part in (true, false)
        landed = 0
        for _ in 1:max_attempts*nstarts
            landed < nstarts || break
            x = box .* (2 .* rand(n) .- 1)
            positive_part && (x[L.positive] .= abs.(x[L.positive]))
            x[iy] = y_on_variety(L, x)
            isfinite(x[iy]) || continue
            _, residual = project_to_variety_residual!(x, cache)
            (isfinite(residual) && residual < 1e-8 && all(isfinite, x)) || continue
            push!(starts, x)
            landed += 1
        end
        landed < nstarts && @warn("$(L.name): only $landed of $nstarts starting points from the " *
                                  (positive_part ? "positive part of the box" : "whole box") *
                                  " landed on the variety (box = $box)")
    end
    return starts
end

"What `positive_points` found on one stratum."
struct StratumResult
    stratum::Stratum
    f::Union{Nothing,Expression}                # nothing if no routing function was needed
    note::String                                # why not
    points::Vector{Vector{Float64}}             # every routing point (every real point, in dimension 0)
    positive::Vector{Vector{Float64}}           # those with α > 0 and M > 0, on a component of the stratum
    components::Vector{Vector{Int}}             # for each positive point, the components of Y it lies on
end

"""
    positive_points(L, stratum; nstarts = 200, box = 10.0, remove_positive_free = true, kwargs...)

The routing points of `r = f/gᵈ` on `stratum` (`f` from [`routing_numerator`](@ref)), the
ones with `α > 0` and `M > 0`, and the components of `Y` each of those lies on. The other
keywords go to `routing_points`: `monodromy_options = (timeout = 3600,)`,
`monodromy_family = :affine`, `flow_options`, `locus_tol`, `crit_tol`, `verbose`.

Two cases need no routing function. A stratum of dimension 0 is a finite set of points:
the system is solved, and `points` are its real solutions. And if the jacobian is rank
deficient on every component of `Y`, then `f` removes all of `Y` and there is nothing
left on stratum 1; the components that matter are strata of their own.
"""
function positive_points(L::LandauSystem, stratum::Stratum; nstarts::Integer = 200, box::Real = 10.0,
                         max_attempts::Integer = 20, remove_positive_free::Bool = true, kwargs...)
    G = [stratum.equations; L.y_equation]
    note = ""
    if !isempty(stratum.components) && all(c -> c.degenerate, stratum.components)
        f, points = nothing, Vector{Float64}[]
        note = "every component lies in the singular locus, so nothing is left to route on"
    elseif length(G) == length(L.vars)
        result = solve(System(G; variables = L.vars); show_progress = false)
        nsingular(result) == 0 || @warn("$(L.name): $(nsingular(result)) singular solutions were left out")
        f, points = nothing, real_solutions(result)
        note = "finitely many points, so the system is solved directly"
    else
        f, _ = routing_numerator(L, stratum; remove_positive_free = remove_positive_free)
        cache = RoutingCache(RoutingFunction(f, L.vars), G)
        starts = starting_points(L, cache; nstarts = nstarts, box = box, max_attempts = max_attempts)
        points = routing_points(cache; starts = starts, kwargs...)
    end

    # the equations of a stratum can cut out more than its components: keep what lies on them
    # (everything, if the components are not known)
    on(q) = [c.number for c in stratum.components if vanishes_at(c.generators, L.vars, q)]
    positive = [q for q in points if is_positive(L, q) && (isempty(stratum.components) || !isempty(on(q)))]
    return StratumResult(stratum, f, note, points, positive, on.(positive))
end

# ---------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------
function point_string(L::LandauSystem, q::AbstractVector{<:Real})
    α = round.([q[1:length(L.α)]; 1]; sigdigits = 6)
    kin = [string(v, " = ", round(q[findfirst(==(v), L.vars)]; sigdigits = 6)) for v in L.kin]
    return "α = $α,  " * join(kin, ",  ")
end

generators_string(gens) = "{" * join(gens, ", ") * "}"

function report(io::IO, L::LandauSystem, S::SingularLocus, results::Vector{StratumResult})
    println(io, L)
    hit = Dict{Int,Int}()                   # component of Y => number of positive points found on it
    for R in results
        stratum = R.stratum
        println(io)
        println(io, "stratum ", stratum.number, " (", stratum.label, "), dimension ", stratum.dim)
        println(io, R.f === nothing ? R.note : "f = $(R.f)")
        println(io, length(R.positive), " of ", length(R.points),
                R.f === nothing ? " real points are positive" : " routing points are positive")
        for (j, q) in enumerate(R.positive)
            println(io, "   $j:  ", point_string(L, q),
                    isempty(R.components[j]) ? "" : "   on component " * join(R.components[j], ", "))
            foreach(c -> hit[c] = get(hit, c, 0) + 1, R.components[j])
        end
        # the singular pieces f removes that could have positive points, or do
        for piece in stratum.pieces
            piece.positivity === :none && continue
            print(io, "singular piece ", piece.number, " (dimension ", piece.dim, ")")
            whole = findfirst(c -> c.degenerate && string.(c.generators) == string.(piece.generators),
                              stratum.components)
            if piece.positivity === :some
                on = [c.number for c in stratum.components if vanishes_at(c.generators, L.vars, piece.point)]
                println(io, " is the positive point ", point_string(L, piece.point),
                        "   on component ", join(on, ", "))
                foreach(c -> hit[c] = get(hit, c, 0) + 1, on)
            elseif whole !== nothing
                println(io, " is component ", stratum.components[whole].number, " of Y, see below")
            else
                println(io, " could have positive points: ", generators_string(piece.generators))
            end
        end
    end

    println(io)
    if isempty(results[1].stratum.components)
        println(io, "The singular locus of Y was not removed and its components are not known: a positive piece")
        println(io, "of Y can have been missed, and `nid = true` tells which components the points lie on.")
        return nothing
    end
    println(io, "the irreducible components of Y:")
    resolved = Set(c.number for R in results[2:end] for c in R.stratum.components)
    for c in results[1].stratum.components
        verdict = if haskey(hit, c.number)
            "positive points found ($(hit[c.number]))"
        elseif c.positivity === :none
            "no positive points (a generator has one sign on them)"
        elseif c.number in S.unresolved
            "in the singular locus, and Macaulay2 found no equations for it: undecided"
        elseif c.degenerate && !(c.number in resolved)
            "in the singular locus: undecided"
        else
            "no positive routing point"
        end
        println(io, "   ", c.number, ". dim ", c.dim, ", degree ", c.degree, ": ", verdict)
        println(io, "      ", generators_string(c.generators))
    end
    println(io)
    println(io, "\"No positive routing point\" means no positive point off the singular pieces listed above,")
    println(io, "provided every routing point was found. The search for them stops heuristically.")
    return nothing
end

"""
    nid_membership(L, points)

The check of examples/positive_landau.jl: a numerical irreducible decomposition of the
whole system, and for each point the witness sets that contain it. Independent of
Macaulay2. It takes a minute on the parachute and has not been run on anything larger.
"""
function nid_membership(L::LandauSystem, points::Vector{Vector{Float64}})
    N = nid(System(L.equations; variables = L.vars))
    for (d, Ws) in witness_sets(N), (i, W) in enumerate(Ws)
        in_W = membership(points, W; show_progress = false)
        for j in findall(in_W)
            println("positive routing point $j lies on component $i of dimension $d (degree $(degree(W)))")
        end
    end
    return N
end

"""
    script_options(args = ARGS)

Keywords for [`positive_landau`](@ref) from the command line of a `<name>.jl` script. They
come after the script's own keywords and override them. run_all.sh uses them to bound a
run, and to go on when Macaulay2 gave no results:

    --monodromy-timeout=3600    monodromy stops after that many seconds, on each stratum
    --no-singular-locus         `singular_locus = false`
"""
function script_options(args = ARGS)
    options = Dict{Symbol,Any}()
    for arg in args
        if arg == "--no-singular-locus"
            options[:singular_locus] = false
        elseif startswith(arg, "--monodromy-timeout=")
            options[:monodromy_options] = (timeout = parse(Float64, last(split(arg, "="))),)
        else
            error("unknown option $arg: there are --monodromy-timeout=SECONDS and --no-singular-locus")
        end
    end
    return options
end

"""
    positive_landau(name; seed = 1, nstarts = 200, box = 10.0, nid = false, save = true,
                    singular_locus = true, kwargs...)

The whole computation for the diagram `name` of diagrams.jl: the Landau equations, the
singular locus found by `<name>.m2`, the routing points of every stratum and the positive
ones among them, and for each irreducible component of `Y` whether a positive point was
found on it. The report is printed, and with `save` written to
`results/<name>_positive.txt`. `nid = true` also runs [`nid_membership`](@ref) on the
positive routing points of `Y`. `singular_locus = false` does without `<name>.m2`, see
[`unknown_singular_locus`](@ref). The other keywords are those of [`positive_points`](@ref).

Returns `(system, singular_locus, results)`.
"""
function positive_landau(name::Symbol; seed::Integer = 1, nid::Bool = false, save::Bool = true,
                         singular_locus::Bool = true, kwargs...)
    L = LandauSystem(name)
    S = singular_locus ? read_singular_locus(L) : unknown_singular_locus(L)
    Random.seed!(seed)
    results = StratumResult[]
    for stratum in S.strata
        seconds = @elapsed push!(results, positive_points(L, stratum; kwargs...))
        @printf("stratum %d (%s): %d points, %d positive, %.0f s\n", stratum.number,
                stratum.label, length(results[end].points), length(results[end].positive), seconds)
    end
    println()
    report(stdout, L, S, results)
    if save
        mkpath(RESULTS_DIR)
        open(io -> report(io, L, S, results), joinpath(RESULTS_DIR, "$(name)_positive.txt"), "w")
    end
    nid && !isempty(results[1].positive) && nid_membership(L, results[1].positive)
    return (system = L, singular_locus = S, results = results)
end
