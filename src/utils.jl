# small utilities shared by every stage: silencing the ODE solvers' warnings, and
# the handling of old keyword spellings.

# runs `f` with logging switched off.
#
# the ODE solvers warn when a path runs out of iterations or goes unstable, and on a
# badly scaled r there are thousands of such paths. the warning's advice -- raise
# `maxiters` -- does not apply here: the flows that run out of iterations are the
# ones that were never going to reach a critical point, and raising the cap buys no
# extra routing points at fifty times the running time. every caller checks the
# return value (`retcode`, `isfinite`, the residual) instead, so the message carries
# nothing the code does not already act on.
#
# `Base.CoreLogging` rather than the Logging stdlib so that this needs no new
# dependency, and a logger rather than the solvers' `verbose` keyword because that
# keyword's accepted type changes between OrdinaryDiffEq versions.
_quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

# a keyword that is still accepted but no longer does anything. these changed which
# points are kept, so the warning is shown once per session rather than hidden
# behind --depwarn.
_ignored_keyword(name, replacement) = @warn(
    "the keyword `$name` no longer has any effect and will be removed; $replacement",
    maxlog = 1, _id = Symbol(:ignored_keyword_, name),
)

# `Verbose` was the old spelling of `verbose` on the path tracking routines
function _verbose(verbose::Bool, Verbose::Union{Nothing,Bool}, fname::Symbol)
    Verbose === nothing && return verbose
    Base.depwarn("the `Verbose` keyword is deprecated, use `verbose`", fname)
    return Verbose
end

# a solution coordinate counts as real when its imaginary part is below this
const IMAG_TOL = 1e-8

# the equations of the variety, in whatever form the caller wrote them: a vector of
# expressions, variables or numbers, a single expression, or a HomotopyContinuation
# System. the package works with a Vector{Expression}.
_as_expressions(G::Vector{Expression}) = G
_as_expressions(G::AbstractVector) = Expression[Expression(g) for g in G]
_as_expressions(g::Union{Expression,Variable}) = Expression[Expression(g)]
_as_expressions(F::System) = Expression[e for e in expressions(F)]

# user-supplied points, e.g. `starts = [[1, 0], [0, 1]]`, as the Vector{Vector{Float64}}
# the numerical routines work with
_as_points(::Nothing) = nothing
_as_points(P::AbstractVector{<:AbstractVector{<:Real}}) = [Vector{Float64}(p) for p in P]
_as_points(P::AbstractVector) = [Float64[x for x in p] for p in P]   # tuples, Vector{Any}, ...
