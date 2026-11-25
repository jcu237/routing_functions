# helpers shared by every stage: silencing the ODE solvers, deprecated keywords,
# and conversion of user input.

# runs `f` with logging switched off.
#
# the ODE solvers warn when a path runs out of iterations or goes unstable, and on a
# badly scaled r there are thousands of such paths. the callers check `retcode`,
# `isfinite` or the residual themselves, and the flows that run out of iterations
# were never going to reach a critical point, so the advice to raise `maxiters`
# does not apply.
#
# uses `Base.CoreLogging` to avoid a Logging dependency, and a logger rather than
# the solvers' `verbose` keyword, whose accepted type varies across OrdinaryDiffEq
# versions.
_quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

# warns, once per session, that a keyword is accepted but ignored. shown even without
# --depwarn because dropping the keyword changed which points are kept.
_ignored_keyword(name, replacement) = @warn(
    "the keyword `$name` no longer has any effect and will be removed; $replacement",
    maxlog = 1, _id = Symbol(:ignored_keyword_, name),
)

# `Verbose` is the deprecated spelling of `verbose`
function _verbose(verbose::Bool, Verbose::Union{Nothing,Bool}, fname::Symbol)
    Verbose === nothing && return verbose
    Base.depwarn("the `Verbose` keyword is deprecated, use `verbose`", fname)
    return Verbose
end

# a solution coordinate counts as real when its imaginary part is below this
const IMAG_TOL = 1e-8

# the equations of the variety as a Vector{Expression}, from a vector of expressions,
# variables or numbers, a single expression or variable, or a System.
_as_expressions(G::Vector{Expression}) = G
_as_expressions(G::AbstractVector) = Expression[Expression(g) for g in G]
_as_expressions(g::Union{Expression,Variable}) = Expression[Expression(g)]
_as_expressions(F::System) = Expression[e for e in expressions(F)]

# user-supplied points, e.g. `starts = [[1, 0], [0, 1]]`, as a Vector{Vector{Float64}}
_as_points(::Nothing) = nothing
_as_points(P::AbstractVector{<:AbstractVector{<:Real}}) = [Vector{Float64}(p) for p in P]
_as_points(P::AbstractVector) = [Float64[x for x in p] for p in P]   # tuples, Vector{Any}, ...
