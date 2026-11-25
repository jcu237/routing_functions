# the variety X = V(G) itself: the normal equations J Jᵀ that the flow fields and
# the projection solve against, and the projection of a point onto X.

# M updates to the Cholesky factor of J Jᵀ + reg·I, in M's lower triangle. the
# regularisation damps the step where J is rank deficient (a singular point of V(G)).
#
# in exact arithmetic every pivot is ≥ reg; in floating point, when J is rank
# deficient with large entries, the rounding in J Jᵀ (≈ eps·‖J‖²) can make a pivot
# negative, so the pivots are clamped at reg. (a reg relative to ‖J‖² cost a fifth
# of the Clebsch seeds; scaling the rows of G is the better cure, see IDEAS.md.)
#
function normal_factor!(M::AbstractMatrix{Float64}, J::AbstractMatrix{Float64}, reg::Float64)
    LA.mul!(M, J, transpose(J))
    k = size(M, 1)
    @inbounds for j = 1:k
        s = M[j, j] + reg
        for p = 1:j-1
            s -= M[j, p]^2
        end
        Ljj = sqrt(max(s, reg))
        M[j, j] = Ljj
        for i = j+1:k
            s = M[i, j]
            for p = 1:j-1
                s -= M[i, p] * M[j, p]
            end
            M[i, j] = s / Ljj
        end
    end
    return M
end

# solves L Lᵀ y = y in place, L being the factor left by `normal_factor!`
function normal_solve!(y::AbstractVector{Float64}, L::AbstractMatrix{Float64})
    k = length(y)
    @inbounds for i = 1:k
        s = y[i]
        for p = 1:i-1
            s -= L[i, p] * y[p]
        end
        y[i] = s / L[i, i]
    end
    @inbounds for i = k:-1:1
        s = y[i]
        for p = i+1:k
            s -= L[p, i] * y[p]
        end
        y[i] = s / L[i, i]
    end
    return y
end

# Gauss-Newton projection onto V(G): steps Jᵀ(J Jᵀ + reg·I)⁻¹G along the normal
# directions, each backtracked until ‖G‖ decreases. stops when ‖G‖ < tol or stops
# decreasing, and returns the point with the residual ‖G(point)‖ reached.
#
# this needs no change when G has more equations than the codimension c of V(G). in
# the singular values σ of J the step is ∑ σ/(σ² + reg) (u·G) v: directions with
# σ² ≫ reg get the Gauss-Newton factor 1/σ, and those with σ² ≪ reg are dropped. the
# m - c extra singular values vanish on V(G), so near it the step is the rank-c
# truncated Gauss-Newton step; further away G has only a second-order component along
# them, so they do no harm either. reg = 0 would divide by rounding noise there.
function _project_to_variety!(
    point::AbstractVector{Float64},
    G_sys::HC.InterpretedSystem,
    G_val::Vector{Float64},
    JG_val::Matrix{Float64},
    M::Matrix{Float64},
    wk::Vector{Float64},
    step::Vector{Float64},
    base::Vector{Float64},
    maxiter::Int64,
    tol::Float64,
    reg::Float64,
    maxbacktrack::Int64
    )

    residual = Inf

    for _ = 1:maxiter
        # a step that overflowed cannot be evaluated (see the flow field)
        all(isfinite, point) || return point, Inf
        try
            HC.evaluate_and_jacobian!(G_val, JG_val, G_sys, point)
        catch err
            err isa InexactError || rethrow()
            return point, Inf
        end
        residual = LA.norm(G_val)
        residual < tol && return point, residual

        # step = Jᵀ (J Jᵀ + reg·I)⁻¹ G
        copyto!(wk, G_val)
        normal_factor!(M, JG_val, reg)
        normal_solve!(wk, M)
        LA.mul!(step, transpose(JG_val), wk)

        copyto!(base, point)
        α = 1.0
        accepted = false
        for _ = 1:maxbacktrack
            @inbounds for i in eachindex(point)
                point[i] = base[i] - α * step[i]
            end
            if !all(isfinite, point)
                α *= 0.5
                continue
            end
            try
                HC.evaluate!(G_val, G_sys, point)
            catch err
                err isa InexactError || rethrow()
                α *= 0.5
                continue
            end
            if LA.norm(G_val) < residual
                accepted = true
                break
            end
            α *= 0.5
        end

        # no step length improves the residual: this is as close as we get
        if !accepted
            copyto!(point, base)
            return point, residual
        end
    end

    return point, residual
end

function project_to_variety!(
    point::AbstractVector{Float64},
    G::Vector{Expression},
    vars::Vector{Variable};
    maxiter::Int64 = 50,
    tol::Float64 = 1e-15,
    reg::Float64 = 1e-8,
    maxbacktrack::Int64 = 20
    )

    n, k = length(vars), length(G)
    G_sys = HC.InterpretedSystem(System(G; variables = vars))
    return first(_project_to_variety!(point, G_sys, zeros(k), zeros(k, n), zeros(k, k),
                                      zeros(k), zeros(n), zeros(n),
                                      maxiter, tol, reg, maxbacktrack))
end

# --- the codimension of X = V(G) -------------------------------------------------
#
# G may have more equations k than the codimension c of X (X is not a complete
# intersection). c is the rank of JG at a generic point of X, provided G is reduced
# there: the pipeline works with G wherever it needs X itself, and squares the routing
# system up to c equations (see `RoutingCache`).

# singular values below this fraction of the largest count as zero
const _RANK_RTOL = 1e-8

# the number of singular values above rtol times the largest
function _numerical_rank(σ::AbstractVector{<:Real}; rtol::Real = _RANK_RTOL)::Int64
    (isempty(σ) || !(σ[1] > 0)) && return 0
    return count(s -> s > rtol * σ[1], σ)
end

# the minimum-norm least-squares solution of J x = b for the best rank-c approximation
# of J: the Gauss-Newton step, and the least-squares multipliers, where J has rank c
# on X but more rows than c
function _truncated_solve(J::AbstractMatrix, b::AbstractVector, c::Integer)
    F = LA.svd(J)
    c = min(c, length(F.S))
    c == 0 && return zeros(promote_type(eltype(J), eltype(b)), size(J, 2))
    return F.V[:, 1:c] * ((F.U[:, 1:c]' * b) ./ F.S[1:c])
end

# each gᵢ divided by the norm of its coefficient vector, so that the rows of JG are of
# comparable size: for the numerical rank, and before G is squared up
function _row_scales(G::Vector{Expression}, vars::Vector{Variable})::Vector{Float64}
    return map(G) do g
        _, coeffs = HC.ModelKit.exponents_coefficients(g, vars)
        s = LA.norm(Float64.(coeffs))
        (isfinite(s) && s > 0) ? s : 1.0
    end
end

# a point of V(G) over ℂ: Gauss-Newton steps from x, each the minimum-norm solution of
# J δ = G for J truncated at relative size rtol. nothing if it does not converge.
function _complex_newton_to_variety(G_sys::HC.InterpretedSystem, x::Vector{ComplexF64}, k::Integer;
                                   maxiter::Integer = 100, rtol::Real = 1e-10)
    n = length(x)
    Gx = zeros(ComplexF64, k)
    J = zeros(ComplexF64, k, n)
    for _ = 1:maxiter
        try
            HC.evaluate_and_jacobian!(Gx, J, G_sys, x)
        catch err
            err isa InexactError || rethrow()
            return nothing
        end
        (all(isfinite, Gx) && all(isfinite, J)) || return nothing
        δ = LA.pinv(J; rtol = rtol) * Gx
        all(isfinite, δ) || return nothing
        x .-= δ
        LA.norm(δ) <= 1e-13 * (1 + LA.norm(x)) && return x
    end
    return nothing
end

"""
    codimension(G, vars; samples = 10) -> Int

The codimension of `X = V(G)` in `ℂⁿ`, estimated numerically: the largest rank of the
jacobian of `G` at `samples` points of `X` over `ℂ`, each reached by Gauss-Newton
steps from a random complex point. The points are complex, so that a real part of `X`
whose points are all singular (the handle of the Whitney umbrella, a sphere touching
a plane) does not lower the estimate. The random draws come from a fixed seed, so the
estimate is reproducible and leaves the global random stream alone.

The rank is the codimension only where `G` is reduced along `X`. On a repeated factor
such as `(x² + y² - 1)²` it is too small; `RoutingCache` checks for this when it uses
the estimate. When `X` has components of different dimensions the largest rank is
the codimension of the smallest of those the samples reach: pass the dimension
yourself (`RoutingCache(r, G; dim = ...)`) to choose which components to keep.
Throws if no sample reaches `X`.
"""
function codimension(G, vars::Vector{Variable}; samples::Integer = 10)::Int64
    G = _as_expressions(G)
    n, k = length(vars), length(G)
    s = _row_scales(G, vars)
    G_sys = HC.InterpretedSystem(System(G ./ s; variables = vars))
    rng = Random.Xoshiro(0x636f6469)
    best = -1
    J = zeros(ComplexF64, k, n)
    for _ = 1:samples
        x = _complex_newton_to_variety(G_sys, randn(rng, ComplexF64, n), k)
        x === nothing && continue
        HC.jacobian!(J, G_sys, x)
        best = max(best, _numerical_rank(LA.svdvals(J)))
    end
    best >= 0 || throw(ArgumentError(
        "codimension: none of the $samples samples reached V(G). Is V(G) empty? " *
        "Pass the dimension of V(G) yourself."))
    return best
end

# whether G looks reduced along X at the complex point x of X, where JG has rank c:
# a short step along the tangent space of the rank-c jacobian must project back to
# within O(step²). on a repeated factor the "tangent space" contains normal
# directions, along which the step projects back by its whole length.
function _reduced_at(G_sys::HC.InterpretedSystem, x::Vector{ComplexF64}, k::Integer, c::Integer,
                     rng)::Bool
    n = length(x)
    J = zeros(ComplexF64, k, n)
    HC.jacobian!(J, G_sys, x)
    V = LA.svd(J; full = true).V[:, c+1:end]
    isempty(V) && return true
    v = V * randn(rng, ComplexF64, size(V, 2))
    ε = 1e-5 * (1 + LA.norm(x))
    q = x .+ (ε / LA.norm(v)) .* v
    p = _complex_newton_to_variety(G_sys, copy(q), k; maxiter = 60)
    # a repeated factor converges only linearly, so the projection may give up; it
    # has moved by about ε by then
    p === nothing && return false
    return LA.norm(p .- q) <= 1e-2 * ε
end

"""
    jacobian_rank_locus(G, vars; dim = nothing)

A polynomial that vanishes exactly on the points of `ℝⁿ` where the jacobian `J` of `G`
has rank below the codimension `c` of `X = V(G)`: the sum of the squares of the `c × c`
minors of `J`, which is the `c`-th elementary symmetric function of `J Jᵀ`. When `G`
has as many equations as the codimension it is `det(J Jᵀ)`; for a single equation `g`
it is `‖∇g‖²`. `c` is `n - dim` if the dimension `dim` of `X` is given, and otherwise
`length(G)` for a single equation and [`codimension`](@ref)`(G, vars)` for several.

This describes the equations `G`, not the set `X`. Only its zeros on `X` matter. They
include every singular point of `X`, but they can include nonsingular points too: a
nonsingular point `p` of `X` is among them exactly when `X` does not have codimension
`c` at `p`, or `G` does not generate the ideal of `X` near `p`.

| `G` | what goes wrong | nonsingular points that vanish |
|---|---|---|
| `(x² + y² - 1)²` | repeated factor | the whole circle |
| `x³ - y³` | the real points are the line `x = y`, whose ideal `G` does not generate | the origin |
| `[x^2, x*y]` | `G` is not reduced at the origin (an embedded point) | the origin of the line `x = 0` |
| `[x*z, y*z]` with `dim = 2` | the line `x = y = 0` has a larger codimension than the plane `z = 0` | the line, off the plane |
| `x² + y²` in `ℝ³` | real points of too low a dimension | the whole `z`-axis |

More equations than the codimension are fine: on the twisted cubic, cut out by three
quadrics, it vanishes nowhere. Do not square `G` up first (replace it by `c` random
combinations of its equations) to get a smaller polynomial: the jacobian of the
combinations drops rank on a hypersurface of `X` that depends on the combination,
and those zeros would cut the components of `X` apart.

For a single equation it is enough that `g` has no repeated factor and that each of
its irreducible factors changes sign on `ℝⁿ`.

Removing the singular points of `X` is the caller's job: multiply `f` by a polynomial
whose zeros on `X` are those points and nothing else. It usually has far lower degree
than this one, which has degree `2c(deg G - 1)` and is dense. On the cone
`x² + y² = z²`, `z` will do. Vanishing on the singular points is not enough, because
`x` also cuts the cone along two lines. When nothing better is to hand, this polynomial
is the fallback:

```julia
r = RoutingFunction(f * jacobian_rank_locus(G, vars), vars)
```

Where the rank drops on a whole piece of `X` that should be kept, no choice of `f`
helps, because the method needs `JG` to have rank `c` there: `G` must be rewritten.
`flow_to_routing_points` warns when its starts land on such a piece, whether or not
`V(f)` contains it. It cannot see a rank drop on a thinner set, such as the origin of
`x³ - y³`, which random starts on `X` do not reach.
"""
function jacobian_rank_locus(G, vars::Vector{Variable}; dim::Union{Nothing,Integer} = nothing)::Expression
    G = _as_expressions(G)
    k, n = length(G), length(vars)
    c = dim !== nothing ? n - dim : (k == 1 ? 1 : codimension(G, vars))
    1 <= c <= min(k, n) || throw(ArgumentError(
        "the codimension of V(G) must be between 1 and $(min(k, n)) (the number of " *
        "equations, and of variables), got $c"))
    J = HC.differentiate(G, vars)                 # k × n
    total = Expression(0)
    for rows in _combinations(k, c), cols in _combinations(n, c)
        total += _laplace_det(J[rows, cols])^2
    end
    return total
end

# determinant by cofactor expansion: division free, so it stays a polynomial
function _laplace_det(M::AbstractMatrix{Expression})::Expression
    m = size(M, 1)
    m == 1 && return M[1, 1]
    return sum((-1)^(j + 1) * M[1, j] * _laplace_det(M[2:end, [c for c = 1:m if c != j]]) for j = 1:m)
end

# all increasing k-element subsets of 1:n
function _combinations(n::Int, k::Int)
    k == 0 && return [Int[]]
    k > n && return Vector{Int}[]
    return vcat([[c..., n] for c in _combinations(n - 1, k - 1)], _combinations(n - 1, k))
end
