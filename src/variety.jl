# the variety X = V(G): the normal equations J Jᵀ, projection onto X, the codimension
# of X, and `jacobian_rank_locus`.

# the Cholesky factor of J Jᵀ + reg·I, written into M's lower triangle. reg damps the
# step where J is rank deficient. in exact arithmetic every pivot is ≥ reg, but when J
# is rank deficient with large entries the rounding in J Jᵀ (≈ eps·‖J‖²) can make one
# negative, so pivots are clamped at reg. (a reg relative to ‖J‖² lost a fifth of the
# Clebsch seeds.)
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

# solves L Lᵀ y = b in place (y holds b on entry), L the factor from `normal_factor!`
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

# Gauss-Newton projection onto V(G): steps Jᵀ(J Jᵀ + reg·I)⁻¹G, each halved until ‖G‖
# decreases. stops when ‖G‖ < tol or no step decreases it; returns the point and ‖G‖.
#
# also correct when G has more equations k than the codimension c. in terms of the
# singular values the step is ∑ σ/(σ² + reg) (u·G) v, so directions with σ² ≪ reg are
# dropped. the k - c extra singular values vanish on V(G), so near it this is the
# rank-c truncated step, and further away G has only a second-order component along
# them. with reg = 0 they would divide rounding noise.
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

# --- the codimension of X ------------------------------------------------------------
#
# G may have more equations k than the codimension c of X. if G is reduced along X, c
# is the rank of JG at a generic point of X.

# the number of singular values above rtol times the largest, and above atol
const _RANK_RTOL = 1e-8
function _numerical_rank(σ::AbstractVector{<:Real}; rtol::Real = _RANK_RTOL, atol::Real = 0)::Int64
    (isempty(σ) || !(σ[1] > 0)) && return 0
    return count(s -> s > max(rtol * σ[1], atol), σ)
end

# how many of the singular values S to keep when truncating at rank c: at most c, and
# none at rounding level (the tolerance of `LA.nullspace`), where J has rank below c
_truncation_rank(S::AbstractVector, c::Integer, J::AbstractMatrix) =
    isempty(S) ? 0 : min(c, count(s -> s > minimum(size(J)) * eps(Float64) * S[1], S))

# the minimum-norm least-squares solution of J x = b with J truncated to rank c
function _truncated_solve(J::AbstractMatrix, b::AbstractVector, c::Integer)
    F = LA.svd(J)
    c = _truncation_rank(F.S, c, J)
    c == 0 && return zeros(promote_type(eltype(J), eltype(b)), size(J, 2))
    return F.V[:, 1:c] * ((F.U[:, 1:c]' * b) ./ F.S[1:c])
end

# the coefficient norm of each gᵢ. dividing by it puts the rows of JG on a common
# scale, for numerical ranks and before squaring up
function _row_scales(G::Vector{Expression}, vars::Vector{Variable})::Vector{Float64}
    return map(G) do g
        _, coeffs = HC.ModelKit.exponents_coefficients(g, vars)
        s = LA.norm(Float64.(coeffs))
        (isfinite(s) && s > 0) ? s : 1.0
    end
end

# a point of V(G) over ℂ, by minimum-norm Gauss-Newton steps from x (J truncated at
# relative size rtol). overwrites x; nothing if it does not converge.
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

The codimension of `X = V(G)`, estimated as the largest numerical rank of the jacobian
of `G` at `samples` complex points of `X`, reached by Gauss-Newton from random starts.
Complex points, because real parts of `X` made of singular points (the handle of the
Whitney umbrella, a sphere touching a plane) would lower the estimate. The seed is
fixed: the result is reproducible and the global random stream is untouched.

The rank is the codimension only if `G` is reduced along `X`; a repeated factor such
as `(x² + y² - 1)²` makes it too small. If the components of `X` differ in dimension,
the result is the codimension of the smallest one the samples reach; pass `dim` to
`RoutingCache` to choose instead. Throws if no sample reaches a point of `X` where the
jacobian is nonzero: Newton can be drawn to an embedded point, as for `[x^2, x*y]`,
whose samples all converge to the origin.
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
        # the rows are normalised, so an absolute floor separates a vanishing jacobian
        # from a small one
        best = max(best, _numerical_rank(LA.svdvals(J); atol = _RANK_RTOL))
    end
    best >= 1 || throw(ArgumentError(
        "codimension: " * (best < 0 ? "no sample reached V(G); is it empty?" :
                           "the jacobian of G vanishes at every sample of V(G); is G " *
                           "reduced?") * " Pass the dimension of V(G): `dim = ...`."))
    return best
end

# whether G looks reduced at the complex point x of X: a step of length ε along the
# kernel of the rank-c jacobian must project back to within O(ε²). on a repeated factor
# that kernel contains normal directions, and the step projects back by about ε.
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
    # Newton converges only linearly on a repeated factor and may give up
    p === nothing && return false
    return LA.norm(p .- q) <= 1e-2 * ε
end

"""
    jacobian_rank_locus(G, vars; dim = nothing)

A polynomial vanishing exactly where the jacobian `J` of `G` has rank below the
codimension `c` of `X = V(G)`: the sum of the squares of the `c × c` minors of `J`,
i.e. the `c`-th elementary symmetric function of `J Jᵀ`. It is `det(J Jᵀ)` when `G` has
`c` equations, and `‖∇g‖²` for a single equation `g`. `c` is `n - dim` if `dim` is
given, else `1` for one equation and [`codimension`](@ref)`(G, vars)` for several.

It depends on the equations, not only on `X`, and only its zeros on `X` matter. They
contain every singular point of `X`, and a nonsingular point `p` exactly when `X`
does not have codimension `c` at `p` or `G` does not generate the ideal of `X` near `p`:

| `G` | why | nonsingular points where it vanishes |
|---|---|---|
| `(x² + y² - 1)²` | repeated factor | the whole circle |
| `x³ - y³` | `G` does not generate the ideal of the line `x = y` | the origin |
| `[x^2, x*y]` with `dim = 1` | embedded point at the origin | the origin of the line `x = 0` |
| `[x*z, y*z]` | `c = 2`, the codimension of the line `x = y = 0`, not of the plane `z = 0` | the whole plane |
| `x² + y²` in `ℝ³` | real points of too low a dimension | the whole `z`-axis |

More equations than the codimension are fine: for the twisted cubic cut out by three
quadrics it vanishes nowhere on the curve. Do not square `G` up first to get a
smaller polynomial: the jacobian of `c` random combinations of `G` drops rank on a
hypersurface of `X`, and those zeros would cut components apart.

For a single equation it suffices that `g` has no repeated factor and each irreducible
factor changes sign on `ℝⁿ`.

Removing the singular points of `X` is the caller's job: multiply `f` by a polynomial
whose zeros on `X` are those points and nothing else. That usually has far lower
degree than this one, which has degree `2c(d - 1)` for equations of degree `d` and is
dense. On the cone `x² + y² = z²`, `z` will do; `x` would not, since it also cuts the
cone along two lines. When nothing better is to hand:

```julia
r = RoutingFunction(f * jacobian_rank_locus(G, vars), vars)
```

Where the rank drops on a whole piece of `X` that should be kept, no `f` helps: `G`
must be rewritten. `flow_to_routing_points` warns when its starts land on such a
piece off `V(f)`. A rank drop on a thinner set, such as the origin of `x³ - y³`, is
not reached by random starts and goes unreported.
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
