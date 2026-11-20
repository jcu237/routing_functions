# The Feynman diagrams of Fig. 1 of
#   [LD] S. Mizera and S. Telen, "Landau Discriminants", arXiv:2109.08036,
# and the polynomial system that cuts out the incidence variety Y of each one.
#
# The equations come from Landau.jl, the package of [LD] (landau/julia_code/julia_code/src):
#   affineLandauEquations(edges, nodes)   ∂F/∂αₑ at α_E = 1 (e = 1, …, E), and y·α₁⋯α_{E-1}·U − 1
#   substitute4legs / substitute5legs      equal masses: Mᵢ = M, mₑ = m
# and then the kinematics are dehomogenized by m = 1, as in examples/positive_landau.jl.
#
# This file needs only HomotopyContinuation. It is included by positive.jl (the routing
# function computation), by generate_equations.jl (which writes the systems out for
# Macaulay2) and by check_diagrams.jl.

using HomotopyContinuation
using LinearAlgebra

# ---------------------------------------------------------------------------------------
# Landau.jl (Mizera and Telen, arXiv:2109.08036)
# ---------------------------------------------------------------------------------------
# Landau.jl as a whole needs Arpack, GenericSVD and Arblib, for sampling and interpolating
# discriminants. Building the equations needs none of that, so only these functions are
# taken from its source file, unchanged, and evaluated in the module below.
const LANDAU_JL = normpath(joinpath(@__DIR__, "..", "..", "landau", "julia_code", "julia_code",
                                    "src", "Landau.jl"))
const LANDAU_FUNCTIONS = (:getU, :getF, :affineLandauEquations, :substitute4legs, :substitute5legs)

module LandauJL
using HomotopyContinuation
using LinearAlgebra
end

function _load_landau!()
    isfile(LANDAU_JL) || error("Landau.jl not found at $LANDAU_JL")
    ast = Meta.parseall(read(LANDAU_JL, String); filename = LANDAU_JL)
    mod = only(ex for ex in ast.args if ex isa Expr && ex.head === :module)
    name(ex) = (sig = ex.args[1]; sig isa Expr && sig.head === :call ? sig.args[1] : nothing)
    for ex in mod.args[3].args
        ex isa Expr && ex.head === :function && name(ex) in LANDAU_FUNCTIONS &&
            Core.eval(LandauJL, ex)
    end
    missing_functions = [f for f in LANDAU_FUNCTIONS if !isdefined(LandauJL, f)]
    isempty(missing_functions) || error("not found in $LANDAU_JL: $missing_functions")
    return nothing
end
_load_landau!()

# ---------------------------------------------------------------------------------------
# the diagrams
# ---------------------------------------------------------------------------------------
# edges[e] is the pair of vertices joined by the internal edge with Schwinger parameter αₑ,
# and nodes[i] is the vertex that carries the external leg pᵢ, both as in Landau.jl. The
# labels follow Fig. 1 of [LD]. acn and dbox are given in [LD] itself (Sec. 3.2 and Sec. 5.2);
# the others are read off the figure. check_diagrams.jl compares the degree of the Landau
# discriminant of each one with Table 1 of [LD], and they all agree.
#
# The two families of Fig. 1 need a size. For the one-loop diagram Aₙ the vertex vᵢ carries
# pᵢ and joins αᵢ to αᵢ₊₁ (Sec. 2.5), so αᵢ is the edge vᵢ₋₁vᵢ. Landau.jl only has
# kinematics for 4 and 5 legs, so the triangle A3 gets p₁ and p₂ on one vertex. The banana
# B_E has p₁, p₂ on one vertex and p₃, p₄ on the other (Ex. 5); B2 is the bubble A2.
const DIAGRAMS = [
    (name = :A3, figure = "1a", title = "one-loop triangle, p₁ and p₂ on one vertex",
     edges = [[3, 1], [1, 2], [2, 3]], nodes = [1, 1, 2, 3]),
    (name = :A4, figure = "1a", title = "one-loop box",
     edges = [[4, 1], [1, 2], [2, 3], [3, 4]], nodes = [1, 2, 3, 4]),
    (name = :B2, figure = "1b", title = "bubble (banana with 2 edges)",
     edges = [[1, 2], [1, 2]], nodes = [1, 1, 2, 2]),
    (name = :B3, figure = "1b", title = "sunrise (banana with 3 edges)",
     edges = [[1, 2], [1, 2], [1, 2]], nodes = [1, 1, 2, 2]),
    (name = :B4, figure = "1b", title = "banana with 4 edges",
     edges = [[1, 2], [1, 2], [1, 2], [1, 2]], nodes = [1, 1, 2, 2]),
    # vertex 1 carries p₁ and p₂; α₁ = 12, α₂ = 13, and α₃, α₄ the doubled edge 23
    (name = :par, figure = "1c", title = "parachute",
     edges = [[1, 2], [1, 3], [2, 3], [2, 3]], nodes = [1, 1, 3, 2]),
    # the box with vertex i at pᵢ, and the diagonal α₅ = 13
    (name = :acn, figure = "1d", title = "acnode",
     edges = [[4, 1], [1, 2], [2, 3], [3, 4], [1, 3]], nodes = [1, 2, 3, 4]),
    # the acnode with the second diagonal α₆ = 24
    (name = :env, figure = "1e", title = "envelope",
     edges = [[4, 1], [1, 2], [2, 3], [3, 4], [1, 3], [2, 4]], nodes = [1, 2, 3, 4]),
    # vertex 1 carries p₁ and p₂, vertex 4 carries p₄ and vertex 5 carries p₃.
    # α₅ = 12, α₁ = 13, α₄ = 24, α₂ = 35, and the crossing edges α₆ = 25, α₃ = 34
    (name = :npltrb, figure = "1f", title = "non-planar triangle-box",
     edges = [[1, 3], [3, 5], [3, 4], [2, 4], [1, 2], [2, 5]], nodes = [1, 1, 5, 4]),
    # vertex 1 carries p₁ and p₂; α₁, α₂ the doubled edge 12 (p₄ at 2), α₃, α₄ the doubled
    # edge 13 (p₃ at 3), α₅ = 23
    (name = :tdetri, figure = "1g", title = "twice doubled-edge triangle",
     edges = [[1, 2], [1, 2], [1, 3], [1, 3], [2, 3]], nodes = [1, 1, 3, 2]),
    # the box with its top edge 34 doubled (α₄, α₅)
    (name = :debox, figure = "1h", title = "doubled-edge box",
     edges = [[4, 1], [1, 2], [2, 3], [3, 4], [3, 4]], nodes = [1, 2, 3, 4]),
    # the box with its bottom edge 12 (α₂, α₃) and its top edge 34 (α₅, α₆) doubled
    (name = :tdebox, figure = "1i", title = "twice doubled-edge box",
     edges = [[4, 1], [1, 2], [1, 2], [2, 3], [3, 4], [3, 4]], nodes = [1, 2, 3, 4]),
    # vertex 1 carries p₁ and p₂, vertex 4 carries p₄ and vertex 5 carries p₃.
    # triangle α₅ = 12, α₆ = 23, α₁ = 13 and box α₄ = 24, α₃ = 45, α₂ = 35, α₆
    (name = :pltrb, figure = "1j", title = "planar triangle-box",
     edges = [[1, 3], [3, 5], [4, 5], [2, 4], [1, 2], [2, 3]], nodes = [1, 1, 5, 4]),
    (name = :dbox, figure = "1k", title = "double-box",
     edges = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1], [3, 6]], nodes = [1, 2, 4, 5]),
    # pentagon 1 2 3 4 7 and box 4 5 6 7 sharing α₈ = 47; pᵢ at vertex i for i ≤ 3,
    # p₄ at 5 and p₅ at 6
    (name = :pentb, figure = "1l", title = "penta-box",
     edges = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 1], [4, 7]],
     nodes = [1, 2, 3, 5, 6]),
]

diagram_names() = [d.name for d in DIAGRAMS]

function diagram(name::Symbol)
    i = findfirst(d -> d.name === name, DIAGRAMS)
    i === nothing && error("no diagram named :$name; the diagrams are $(diagram_names())")
    return DIAGRAMS[i]
end

# ---------------------------------------------------------------------------------------
# the incidence variety
# ---------------------------------------------------------------------------------------
"""
    LandauSystem

The Landau equations of a diagram on the equal-mass slice, in the charts `α_E = 1` and
`m = 1`:

* `α`: the Schwinger parameters `α₁, …, α_{E-1}`;
* `y`: the variable of the last equation, which forces `α₁⋯α_{E-1}·U ≠ 0`;
* `kin`: the kinematic variables that occur, from `(s, t, M)` for 4 legs and from
  `(s12, s23, s34, s45, s51, M)` for 5 legs. A diagram with two legs on one vertex does
  not involve `t`, and the bananas involve only `s`;
* `vars = [α; y; kin]`, the order every point is written in;
* `U`: the first Symanzik polynomial at `α_E = 1`;
* `landau`: `∂F/∂αₑ` at `α_E = 1` and `m = 1`, for `e = 1, …, E`;
* `y_equation`: `y·α₁⋯α_{E-1}·U − 1`;
* `equations = [landau; y_equation]`, which cut out the incidence variety `Y`;
* `positive`: the positions in `vars` of the coordinates that have to be positive, i.e.
  `α` and the mass ratio `M`.

`Y` is cut out by `E + 1` equations in `E + length(kin)` unknowns, so it has dimension
`length(kin) − 1` where the jacobian has full rank.
"""
struct LandauSystem
    name::Symbol
    title::String
    edges::Vector{Vector{Int}}
    nodes::Vector{Int}
    α::Vector{Variable}
    y::Variable
    kin::Vector{Variable}
    vars::Vector{Variable}
    U::Expression
    landau::Vector{Expression}
    y_equation::Expression
    equations::Vector{Expression}
    positive::Vector{Int}
end

LandauSystem(name::Symbol) = (d = diagram(name); LandauSystem(d.name, d.title, d.edges, d.nodes))

function LandauSystem(name::Symbol, title::AbstractString, edges, nodes)
    nlegs = length(nodes)
    nlegs in (4, 5) || error("Landau.jl has kinematics for 4 and 5 legs only, got $nlegs")

    _, U, _, _, _ = LandauJL.getF(edges, nodes)
    ALE, y, α, p, mm = LandauJL.affineLandauEquations(edges, nodes)
    if nlegs == 4
        ALE, s, t, M, m = LandauJL.substitute4legs(ALE, p, mm; equalM = true, equalm = true)
        mandelstams = [s, t]
    else
        ALE, s12, s23, s34, s45, s51, M, m =
            LandauJL.substitute5legs(ALE, p, mm; equalM = true, equalm = true)
        mandelstams = [s12, s23, s34, s45, s51]
    end
    equations = expand.(subs(ALE, m => 1))

    occurring = variables(equations)
    kin = [v for v in [mandelstams; M] if v in occurring]
    αs = α[1:end-1]
    vars = [αs; y; kin]
    @assert Set(occurring) == Set(vars) "unexpected variables in the Landau equations of $name"
    positive = [i for (i, v) in enumerate(vars) if v in αs || v == M]

    return LandauSystem(name, String(title), [collect(Int, e) for e in edges], collect(Int, nodes),
                        αs, y, kin, vars, expand(subs(U, α[end] => 1)),
                        equations[1:end-1], equations[end], equations, positive)
end

nedges(L::LandauSystem) = length(L.edges)

# the dimension of Y where the jacobian of the equations has full rank
expected_dimension(L::LandauSystem) = length(L.vars) - length(L.equations)

function Base.show(io::IO, L::LandauSystem)
    print(io, "LandauSystem ", L.name, " (", L.title, "): ", length(L.equations),
          " equations in (", join(L.vars, ", "), "), expected dimension ", expected_dimension(L))
end

# ---------------------------------------------------------------------------------------
# text for Macaulay2
# ---------------------------------------------------------------------------------------
# Macaulay2 works without y (see singular_locus.m2): its ring has the variables
# α_1, …, α_{E-1} and the kinematic ones, under these names
m2_vars(L::LandauSystem) = [L.α; L.kin]
m2_names(L::LandauSystem) = [["α_$i" for i in eachindex(L.α)]; string.(L.kin)]

# Landau.jl works in floating point (it sets α_E = 1.0 and halves the Mandelstam
# invariants), but the coefficients of the equations are integers. Anything else is a bug.
function _exact_coefficient(c)::BigInt
    x = Float64(real(c))
    q = round(BigInt, x)
    (abs(imag(c)) < 1e-12 && abs(x - q) < 1e-9) ||
        error("coefficient $c of a Landau equation is not an integer")
    return q
end

# `poly` as a polynomial with integer coefficients in Macaulay2 syntax
function m2_string(poly::Expression, vars::Vector{Variable}, names::Vector{String})::String
    E, C = exponents_coefficients(expand(poly), vars)
    io = IOBuffer()
    first_term = true
    for j in eachindex(C)
        c = _exact_coefficient(C[j])
        c == 0 && continue
        monomial = join((E[i, j] == 1 ? names[i] : "$(names[i])^$(E[i, j])"
                         for i in eachindex(vars) if E[i, j] > 0), "*")
        print(io, c < 0 ? "-" : (first_term ? "" : "+"))
        if isempty(monomial)
            print(io, abs(c))
        else
            abs(c) == 1 || print(io, abs(c), "*")
            print(io, monomial)
        end
        first_term = false
    end
    return first_term ? "0" : String(take!(io))
end

function m2_string(L::LandauSystem, poly::Expression)
    L.y in variables(poly) && error("the Macaulay2 ring of $(L.name) has no y")
    return m2_string(poly, m2_vars(L), m2_names(L))
end
