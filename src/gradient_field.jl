# evaluates what the flow field needs (G, its jacobian, f, ∇f, ∇r). returns `false`
# if u or any value is non-finite, or the evaluation overflows (InexactError).
function _evaluate_field!(G_val, JG_val, grad_val, f_val, ∇f_val, G_sys, r, u)
    all(isfinite, u) || return false
    try
        HC.evaluate_and_jacobian!(G_val, JG_val, G_sys, u)
        HC.evaluate!(f_val, r.f_sys, u)
        HC.evaluate!(∇f_val, r.∇f_sys, u)
    catch err
        err isa InexactError || rethrow()
        return false
    end
    (all(isfinite, G_val) && isfinite(@inbounds f_val[1]) && all(isfinite, ∇f_val)) || return false
    _grad_r!(grad_val, r, u, @inbounds(f_val[1]), ∇f_val)
    return all(isfinite, grad_val)
end

# projected gradient field of r on X = V(G), with J = JG:
#
#   du = dir · sign(r) · (I - Jᵀ(JJᵀ + reg·I)⁻¹J) ∇r  -  Jᵀ(JJᵀ + reg·I)⁻¹ G
#
# Jᵀ(JJᵀ + reg·I)⁻¹ is a regularised pseudoinverse of J, so the first term is ∇r with
# its (approximate) normal component removed: it ascends |r| within X. the second is
# a Newton step holding the path on X. all k equations of G are used, even when k
# exceeds the codimension: reg acts as a soft cutoff that drops the vanishing extra
# singular values of J. `dir` is the ODE parameter: its sign picks the direction and
# its size rescales time. with `unit_speed` the tangential part is normalised, so
# time is arc length (needed when starting next to a critical point, where ∇r ≈ 0).
#
# the closure owns its buffers, so nothing else touching the cache can disturb an
# integration in progress.
function _projected_gradient_field(
    r::RoutingFunction,
    G_sys::HC.InterpretedSystem,
    n::Int64,
    k::Int64,
    reg::Float64,
    unit_speed::Bool
    )

    G_val = zeros(Float64, k)
    JG_val = zeros(Float64, k, n)
    grad_val = zeros(Float64, n)
    f_val = zeros(Float64, 1)
    ∇f_val = zeros(Float64, n)
    M = zeros(Float64, k, k)
    wk = zeros(Float64, k)
    wk2 = zeros(Float64, k)
    tangential = zeros(Float64, n)

    function flow!(du, u, dir, t)
        # a path running off to infinity (u non-finite, or so large that f overflows)
        # is dead: return a zero derivative and let the solver or a callback end it.
        # callers drop non-finite end points.
        if !_evaluate_field!(G_val, JG_val, grad_val, f_val, ∇f_val, G_sys, r, u)
            fill!(du, 0.0)
            return nothing
        end

        # both halves of the field solve against J Jᵀ + reg·I, so factor it once
        normal_factor!(M, JG_val, reg)

        # tangential part, oriented by sign(r) = sign(f) (g > 0)
        LA.mul!(wk, JG_val, grad_val)
        normal_solve!(wk, M)
        LA.mul!(tangential, transpose(JG_val), wk)
        s = dir * sign(@inbounds f_val[1])
        @inbounds for i = 1:n
            tangential[i] = s * (grad_val[i] - tangential[i])
        end

        # no absolute threshold here: ‖∇r‖ scales with f
        if unit_speed
            speed = LA.norm(tangential)
            (speed > 0 && isfinite(speed)) && (tangential ./= speed)
        end

        # normal part: pull back onto V(G)
        copyto!(wk2, G_val)
        normal_solve!(wk2, M)
        LA.mul!(du, transpose(JG_val), wk2)
        @inbounds for i = 1:n
            du[i] = tangential[i] - du[i]
        end
        return nothing
    end

    return flow!
end
