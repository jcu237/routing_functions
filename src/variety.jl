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

"""
    singular_locus(G, vars)

A polynomial that vanishes exactly on the points of `ℝⁿ` where the jacobian of `G`
has rank below `length(G)` -- in particular on the singular points of `V(G)`: the sum
of the squares of the maximal minors of the jacobian. For a single equation `g` it is
`‖∇g‖²`.

The theory needs `V(G)` to be smooth off the removed locus, so when `V(G)` has
singular points they must be removed with it:

```julia
r = RoutingFunction(f * singular_locus(G, vars), vars)
```

(Its degree is `k · (deg G - 1) · 2`, which makes the routing system larger; when the
singular points are known, a smaller polynomial vanishing on them will do.)
"""
function singular_locus(G, vars::Vector{Variable})::Expression
    G = _as_expressions(G)
    k = length(G)
    J = HC.differentiate(G, vars)                 # k × n
    n = size(J, 2)
    k <= n || throw(ArgumentError("G has more equations ($k) than there are variables ($n)"))
    total = Expression(0)
    for cols in _combinations(n, k)
        total += _laplace_det(J[:, cols])^2
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
