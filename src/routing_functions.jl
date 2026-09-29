# r = f / g^d, where g = ∑ᵢ (xᵢ - cᵢ)² + 1 is strictly positive on ℝⁿ and
# d = deg(f) ÷ 2 + 1 is large enough that r → 0 at infinity.
#
# only f and ∇f are compiled. r and its derivatives follow from them, because g's
# derivatives are known in closed form (∇g = 2(x - c), ∇²g = 2I):
#
#   r   = f / gᵈ
#   ∇r  = (∇f - (d f / g) ∇g) / gᵈ
#   ∇²r = ∇²f / gᵈ - d (∇f ∇gᵀ + ∇g ∇fᵀ + f ∇²g) / gᵈ⁺¹ + d (d + 1) f ∇g ∇gᵀ / gᵈ⁺²
#
# the interpreted systems are compiled once, here. anything that also depends on
# the variety V(G) lives in `RoutingCache`.
"""
    RoutingFunction(f, vars, c)
    RoutingFunction(f, vars)        # c random
    RoutingFunction(f, c)           # vars = variables(f)
    RoutingFunction(f)

The routing function `r = f / gᵈ`, with `g = ‖x - c‖² + 1` and `d = deg(f) ÷ 2 + 1`.

`f` is the polynomial whose zero set is **removed**: the package computes the
connected components of `V(G) \\ V(f)`. Typical choices:

* `1` (or any nonzero constant) removes nothing;
* a product `h₁ h₂ ⋯` removes `V(h₁) ∪ V(h₂) ∪ ⋯`, e.g. `x*y` removes the axes, and a
  coordinate product sorts points by the signs of the coordinates;
* the singular points of `V(G)` **must** lie in `V(f)`: multiply `f` by
  [`singular_locus`](@ref)`(G, vars)` (for a hypersurface `g` that is `‖∇g‖²`).

`vars` are the ambient variables, in the order points are written in; pass them
whenever `f` does not involve all of them (a constant `f` involves none). `c` is the
centre of `g`. It must be generic -- a symmetric choice such as the centre of a circle
makes `r` degenerate there and the results meaningless -- and is drawn at random from
`[0, 1]ⁿ` when omitted. `d` makes `r → 0` at infinity. Scaling `f` by a constant
changes nothing.

`r` is callable: `r(P)` is `r` at the point `P`.
"""
struct RoutingFunction
    f::Expression                # numerator
    g::Expression                # base of the denominator, determined by c
    c::Vector{Float64}           # center of g; chosen randomly unless specified
    d::Int64                     # exponent of g
    vars::Vector{Variable}       # ambient variables
    ∇f::Vector{Expression}       # gradient of the numerator
    grad_num::Vector{Expression} # ∇f·g - d·f·∇g = gᵈ⁺¹·∇r, polynomial; the routing system is built from it

    f_sys::HC.InterpretedSystem  # [f]; g > 0, so sign(r) == sign(f)
    ∇f_sys::HC.InterpretedSystem # ∇f; its jacobian is ∇²f
    f_abs_sys::HC.InterpretedSystem # [∑ₐ |cₐ| xᵃ], f with its coefficients made positive; bounds rounding in f

    function RoutingFunction(f::Expression, vars::Vector{Variable}, c::AbstractVector{<:Real})
        isempty(vars) && throw(ArgumentError(
            "f = $f involves no variables, so the ambient space is unknown; pass the " *
            "variables explicitly: RoutingFunction(f, vars) or RoutingFunction(f, vars, c)",
        ))
        length(c) == length(vars) || throw(ArgumentError(
            "the center c has length $(length(c)) but r is to be a function of " *
            "$(length(vars)) variables ($(join(vars, ", "))); they must match. if the " *
            "ambient space has variables f does not use, pass them all: " *
            "RoutingFunction(f, vars, c)",
        ))

        # f may use fewer variables than `vars`, but not more
        stray = setdiff(variables(f), vars)
        isempty(stray) || throw(ArgumentError(
            "f involves variables not in vars: $(join(stray, ", ")) (vars = $(join(vars, ", ")))",
        ))

        c = Vector{Float64}(c)
        g = sum((vars[i] - c[i])^2 for i in eachindex(vars)) + 1
        d = degree(f) ÷ 2 + 1
        ∇f = HC.differentiate(f, vars)
        grad_num = ∇f * g - d * f * HC.differentiate(g, vars)

        new(
            f, g, c, d, vars, ∇f, grad_num,
            HC.InterpretedSystem(System([f]; variables = vars)),
            HC.InterpretedSystem(System(∇f; variables = vars)),
            HC.InterpretedSystem(System([_abs_coefficients(f, vars)]; variables = vars)),
        )
    end
end

# ∑ₐ |cₐ| xᵃ for f = ∑ₐ cₐ xᵃ. evaluated at |x| it bounds every partial sum formed
# while evaluating f(x), so f(x) is indistinguishable from 0 once |f(x)| is below a
# modest multiple of eps times it.
function _abs_coefficients(f::Expression, vars::Vector{Variable})::Expression
    E, C = exponents_coefficients(f, vars)
    isempty(C) && return Expression(0)
    return sum(abs(Float64(real(C[i]))) * prod(vars[j]^E[j, i] for j in eachindex(vars); init = Expression(1))
               for i in eachindex(C))
end

RoutingFunction(f::Expression, c::AbstractVector{<:Real}) = RoutingFunction(f, variables(f), c)

RoutingFunction(f::Expression, vars::Vector{Variable}) =
    RoutingFunction(f, vars, rand(length(vars)))

RoutingFunction(f::Expression) = RoutingFunction(f, variables(f))

# a bare variable (`RoutingFunction(x[1], vars)`) or a constant
# (`RoutingFunction(1, vars)`, which removes nothing) is a perfectly good numerator.
# (not `f::Number`: an Expression is a Number, and would land back here forever.)
RoutingFunction(f::Union{Variable,Real}, args...) = RoutingFunction(Expression(f), args...)

# anything else is a mistake in the arguments; say which forms exist
RoutingFunction(f::Expression, args...) = throw(ArgumentError(
    "RoutingFunction takes (f), (f, c), (f, vars) or (f, vars, c), where vars is a " *
    "Vector{Variable} and c a vector of reals of the same length; got (f, " *
    join(string.(typeof.(args)), ", ") * ")",
))

Base.length(r::RoutingFunction) = length(r.vars)

function Base.show(io::IO, r::RoutingFunction)
    print(io, "RoutingFunction in ", length(r.vars), " variables: (", r.f, ") / (", r.g, ")^", r.d)
end

function _check_length(r::RoutingFunction, point)
    length(point) == length(r.vars) || error(
        "the point has length ", length(point), " but r is a function of ",
        length(r.vars), " variables",
    )
end

# g(P). g is a shifted sum of squares, so evaluating it directly beats going
# through the interpreter. `_g` skips the length check, for the inner loops.
function _g(r::RoutingFunction, point::AbstractVector{<:Real})::Float64
    s = 1.0
    @inbounds for i in eachindex(r.c)
        s += (point[i] - r.c[i])^2
    end
    return s
end

evaluate_g(r::RoutingFunction, point::AbstractVector{<:Real})::Float64 =
    (_check_length(r, point); _g(r, point))

"""
    evaluate_f(r, P)
    evaluate_g(r, P)
    evaluate_r(r, P)    # also r(P)

The numerator `f`, the base `g = ‖P - c‖² + 1` of the denominator, and `r = f / gᵈ`
at the point `P`.
"""
function evaluate_f(r::RoutingFunction, point::AbstractVector{<:Real})::Float64
    _check_length(r, point)
    out = zeros(Float64, 1)
    HC.evaluate!(out, r.f_sys, point)
    return @inbounds out[1]
end

# r(P)
evaluate_r(r::RoutingFunction, point::AbstractVector{<:Real})::Float64 =
    evaluate_f(r, point) / _g(r, point)^r.d

(r::RoutingFunction)(point::AbstractVector{<:Real}) = evaluate_r(r, point)

# out ← ∇r(x), given f(x) and ∇f(x) already evaluated. allocation free, for the
# flow fields.
@inline function _grad_r!(
    out::AbstractVector{Float64},
    r::RoutingFunction,
    x::AbstractVector{Float64},
    fx::Float64,
    ∇fx::AbstractVector{Float64}
    )

    g = _g(r, x)
    s = 2 * r.d * fx / g            # ∇g = 2(x - c), so (d f / g) ∇g = s (x - c)
    w = inv(g^r.d)
    @inbounds for i in eachindex(out)
        out[i] = (∇fx[i] - s * (x[i] - r.c[i])) * w
    end
    return out
end

"""
    evaluate_grad_r(r, P)
    evaluate_grad_hessian_r(r, P) -> (∇r, ∇²r)

The gradient, and the gradient and hessian, of `r` in the ambient space at `P`,
computed in closed form from `f`, `∇f`, `∇²f` and `g` by the quotient rule.
"""
function evaluate_grad_r(r::RoutingFunction, point::AbstractVector{<:Real})::Vector{Float64}
    _check_length(r, point)
    x = Vector{Float64}(point)
    fx = zeros(Float64, 1)
    ∇fx = zeros(Float64, length(x))
    HC.evaluate!(fx, r.f_sys, x)
    HC.evaluate!(∇fx, r.∇f_sys, x)
    return _grad_r!(similar(∇fx), r, x, fx[1], ∇fx)
end

function evaluate_grad_hessian_r(r::RoutingFunction, point::AbstractVector{<:Real})
    _check_length(r, point)
    x = Vector{Float64}(point)
    n = length(x)
    fx = zeros(Float64, 1)
    ∇f = zeros(Float64, n)
    ∇²f = zeros(Float64, n, n)
    HC.evaluate!(fx, r.f_sys, x)
    HC.evaluate_and_jacobian!(∇f, ∇²f, r.∇f_sys, x)
    f = fx[1]
    d = r.d
    g = _g(r, x)
    ∇g = 2 .* (x .- r.c)

    ∇r = _grad_r!(zeros(n), r, x, f, ∇f)
    ∇²r = ∇²f ./ g^d .- d .* (∇f * ∇g' .+ ∇g * ∇f') ./ g^(d + 1)
    ∇²r .+= d * (d + 1) * f .* (∇g * ∇g') ./ g^(d + 2)
    @inbounds for i = 1:n
        ∇²r[i, i] -= 2 * d * f / g^(d + 1)         # the f ∇²g term, ∇²g = 2I
    end
    return ∇r, ∇²r
end

# how far P is from the removed locus V(f). this, not |r(P)|, decides whether a
# point lies on V(f): |r| depends on the scale of f and is tiny far from c or in
# thin regions, a distance does not.
#
# from the second-order Taylor model of f at P (V(f) is no closer than the
# positive root of |f| = ‖∇f‖δ + ½‖∇²f‖δ²):
#
#     δ = 2|f| / (‖∇f‖ + sqrt(‖∇f‖² + 2‖∇²f‖·|f|)).
#
# where ∇f ≠ 0 this is about the Newton step |f|/‖∇f‖; the ∇²f term matters at
# singular points of V(f) (crossings, removed nodes), where ∇f vanishes as well.
# (Frobenius norm of ∇²f: an upper bound, so δ errs towards "close".)
function _distance_to_zero_locus(fx::Float64, ∇f_norm::Float64, ∇²f_norm::Float64)::Float64
    fx == 0 && return 0.0
    a = abs(fx)
    den = ∇f_norm + sqrt(∇f_norm^2 + 2 * ∇²f_norm * a)
    return den == 0 ? Inf : 2a / den
end

"""
    distance_to_zero_locus(r, P)

A lower estimate of the distance from `P` to `V(f)`, the locus `r` removes: the
positive root `δ` of `|f(P)| = ‖∇f(P)‖ δ + ½‖∇²f(P)‖ δ²`. It is `|f|/‖∇f‖` where `∇f ≠ 0`,
stays meaningful where `V(f)` is singular, and does not change when `f` is rescaled.
"""
function distance_to_zero_locus(r::RoutingFunction, point::AbstractVector{<:Real})::Float64
    _check_length(r, point)
    all(isfinite, point) || return NaN
    x = Vector{Float64}(point)
    n = length(x)
    fx = zeros(Float64, 1)
    ∇f = zeros(Float64, n)
    ∇²f = zeros(Float64, n, n)
    HC.evaluate!(fx, r.f_sys, x)
    HC.evaluate_and_jacobian!(∇f, ∇²f, r.∇f_sys, x)
    return _distance_to_zero_locus(fx[1], LA.norm(∇f), LA.norm(∇²f))
end

# two tests: the distance above against locus_tol·(1 + ‖P‖), and a rounding floor
# |f(P)| ≤ 1000·eps·∑ₐ |cₐ| |P|ᵃ, below which f(P) cannot be told apart from 0
# (needed where ∇²f vanishes too, e.g. three components of V(f) through a point).
"""
    on_zero_locus(r, P; locus_tol = 1e-6)

Whether `P` lies on the removed locus `V(f)`: its [`distance_to_zero_locus`](@ref)
is at most `locus_tol * (1 + ‖P‖)`, or `f(P)` is within rounding error of zero. A
solution of the routing system for which this holds is not a routing point.
"""
function on_zero_locus(r::RoutingFunction, point::AbstractVector{<:Real}; locus_tol::Real = 1e-6)::Bool
    δ = distance_to_zero_locus(r, point)
    isnan(δ) && return false
    δ <= locus_tol * (1 + LA.norm(point)) && return true
    x = Vector{Float64}(point)
    fx = zeros(Float64, 1)
    fabs = zeros(Float64, 1)
    HC.evaluate!(fx, r.f_sys, x)
    HC.evaluate!(fabs, r.f_abs_sys, abs.(x))
    return abs(fx[1]) <= 1e3 * eps() * fabs[1]
end

# the cheap first-order version, for the flow callbacks: |f| / ‖∇f‖
function _newton_distance(fx::Float64, ∇fx::AbstractVector{Float64})::Float64
    fx == 0 && return 0.0
    nrm = LA.norm(∇fx)
    return nrm == 0 ? Inf : abs(fx) / nrm
end
