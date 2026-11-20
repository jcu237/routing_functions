# Writes equations/<name>.m2 for the diagrams of diagrams.jl: the ring and the Landau
# equations of each one in Macaulay2 syntax, built with Landau.jl. The <name>.m2 scripts
# load these files, and positive.jl checks that the system Macaulay2 worked with is the
# one it builds itself.
#
#   julia --project=. examples/positive_landau/generate_equations.jl             # all diagrams
#   julia --project=. examples/positive_landau/generate_equations.jl acn env     # some of them

include(joinpath(@__DIR__, "diagrams.jl"))

const EQUATIONS_DIR = joinpath(@__DIR__, "equations")

m2_list(v) = "{" * join(v, ", ") * "}"

function write_equations(L::LandauSystem; dir::AbstractString = EQUATIONS_DIR)
    mkpath(dir)
    names = m2_names(L)
    E = nedges(L)
    path = joinpath(dir, "$(L.name).m2")
    open(path, "w") do io
        println(io, "-- Written by generate_equations.jl with Landau.jl. Do not edit: change diagrams.jl and")
        println(io, "-- run generate_equations.jl again.")
        println(io, "--")
        println(io, "-- The Landau equations of the ", L.title, " (", L.name, "), with equal masses, in the")
        println(io, "-- charts α_", E, " = 1 and m = 1.")
        println(io, "--   edges (the e-th one carries α_e): ", m2_list(m2_list.(L.edges)))
        println(io, "--   the leg p_i is attached to the vertex: ", m2_list(L.nodes))
        println(io)
        println(io, "diagramName = \"", L.name, "\";")
        println(io, "R = QQ[", join(names, ", "), "];")
        println(io)
        println(io, "-- the first Symanzik polynomial at α_", E, " = 1")
        println(io, "U = ", m2_string(L, L.U), ";")
        println(io)
        println(io, "-- the derivatives of F by α_1, ..., α_", E, " at α_", E, " = 1")
        println(io, "landauEquations = {")
        println(io, join(("    " * m2_string(L, g) for g in L.landau), ",\n"))
        println(io, "    };")
        println(io)
        println(io, "-- The system has one more equation, y*", join(names[1:E-1], "*"), "*U = 1, which says that no α_e")
        println(io, "-- vanishes and neither does U. Macaulay2 saturates by these instead (see singular_locus.m2).")
        println(io, "alphaVariables = ", m2_list(names[1:E-1]), ";")
        println(io)
        println(io, "-- the coordinates that are positive at a positive point: the α_e and the mass ratio M")
        println(io, "positiveVariables = ", m2_list(names[indexin(L.vars[L.positive], m2_vars(L))]), ";")
    end

    # the product written for Macaulay2 has to be the last equation Landau.jl built
    y_product = L.y * prod(L.α; init = Expression(1)) * L.U - 1
    @assert iszero(expand(y_product - L.y_equation)) "y equation of $(L.name) is not y·α₁⋯α_{E-1}·U − 1"
    return path
end

function main(args)
    names = isempty(args) ? diagram_names() : Symbol.(args)
    for name in names
        L = LandauSystem(name)
        path = write_equations(L)
        println(rpad(name, 8), length(L.equations), " equations in ", length(L.vars),
                " unknowns, expected dimension ", expected_dimension(L), "  ->  ", relpath(path))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
