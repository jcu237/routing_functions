# Checks the diagrams of diagrams.jl against [LD]: the degree of the Landau discriminant
# on the equal-mass slice, computed as in Sec. 3.2 of [LD] (Landau.jl's degreeProjection),
# next to the degrees of Table 1. A wrong edge or a leg on the wrong vertex changes it.
#
#   julia -t auto --project=. examples/positive_landau/check_diagrams.jl            # all diagrams
#   julia -t auto --project=. examples/positive_landau/check_diagrams.jl env pentb  # some of them
#
# The incidence variety is cut with a random line in the space of the kinematic variables,
# and the distinct kinematic points among the solutions are counted. Solutions that are
# singular, or that the solver loses, are counted separately: a component of the
# discriminant whose fibres in Y have positive dimension (M = m for the parachute) only
# shows up there, if at all. So the count can fall short of Table 1 but must not exceed it.

include(joinpath(@__DIR__, "diagrams.jl"))

# The degrees of the components of codimension 1 of the discriminant in the chart m = 1:
# Table 1 of [LD], without the component m = 0 of npltrb; eq. (17) for the box; Prop. 2 for
# the bananas (s = k²m for k = E, E − 2, … > 0). A3 is not in [LD].
const DISCRIMINANT_DEGREES = Dict(
    :A4 => [2], :B2 => [1], :B3 => [1, 1], :B4 => [1, 1],
    :par => [1, 2], :acn => [4, 8], :env => [8, 8, 8, 9, 12], :npltrb => [1], :tdetri => [2],
    :debox => [3], :tdebox => [2], :pltrb => Int[], :dbox => [2, 4], :pentb => [12],
)

# the distinct points among `points`, up to a relative distance `tol`
function distinct(points; tol = 1e-6)
    kept = empty(points)
    for p in points
        any(q -> norm(p - q) <= tol * max(norm(p), 1), kept) || push!(kept, p)
    end
    return kept
end

"""
    discriminant_degree(L) -> (regular, singular)

The number of distinct kinematic points of `Y` over a random line of kinematic space, from
the nonsingular solutions and from the singular ones that give further points.
"""
function discriminant_degree(L::LandauSystem)
    nk = length(L.kin)
    A, b = randn(ComplexF64, nk - 1, nk), randn(ComplexF64, nk - 1)
    slice = nk > 1 ? A * L.kin + b : Expression[]
    result = solve(System([L.equations; slice]; variables = L.vars); show_progress = false)
    kin = indexin(L.kin, L.vars)
    regular = distinct([s[kin] for s in solutions(result)])
    singular_points = distinct([solution(r)[kin] for r in singular(result)])
    extra = [p for p in singular_points if !any(q -> norm(p - q) <= 1e-6 * max(norm(p), 1), regular)]
    return length(regular), length(extra)
end

function main(args)
    names = isempty(args) ? diagram_names() : Symbol.(args)
    for name in names
        L = LandauSystem(name)
        seconds = @elapsed regular, singular = discriminant_degree(L)
        expected = get(DISCRIMINANT_DEGREES, name, nothing)
        verdict = expected === nothing ? "not in [LD]" :
                  regular == sum(expected) ? "agrees with $expected" :
                  regular < sum(expected) ? "below $expected" : "DISAGREES with $expected"
        println(rpad(name, 8), "degree ", regular, singular > 0 ? " (+$singular singular)" : "",
                ": ", verdict, "   [", round(seconds; digits = 1), " s]")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
