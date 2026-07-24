# Newton projection onto V(G). The differentiated system can be supplied by callers
# which invoke this repeatedly to avoid rebuilding the same symbolic Jacobian.
function project_to_variety!(
    point::Vector{Float64},
    G::Vector{Expression},
    vars::Vector{Variable};
    JG = HC.differentiate(G, vars),
    maxiter::Int64 = 5,
    tol::Float64 = 1e-15
    )

    for _ in 1:maxiter
        Gx = HC.evaluate(G, vars => point)
        if LA.norm(Gx) < tol
            return point
        end

        JGx = HC.evaluate(JG, vars => point)
        point .-= JGx \ Gx
    end

    return point
end

function tangent_gradient(
    r::routing_function,
    G::Vector{Expression},
    vars::Vector{Variable},
    point::Vector{Float64};
    JG = HC.differentiate(G, vars)
    )

    grad = Vector{Float64}(r.eval_grad(point))
    JGx = HC.evaluate(JG, vars => point)
    V = LA.nullspace(JGx)

    if size(V, 2) == 0
        return zeros(length(point))
    end

    return V * (transpose(V) * grad)
end

function projected_routing_vector_field(
    r::routing_function,
    G::Vector{Expression},
    vars::Vector{Variable},
    point::Vector{Float64},
    direction_sign::Float64;
    JG = HC.differentiate(G, vars)
    )

    v = direction_sign .* tangent_gradient(r, G, vars, point; JG = JG)
    nrm = LA.norm(v)
    if nrm == 0
        return v
    end

    return v ./ nrm
end

function rk4_projected_step!(
    r::routing_function,
    G::Vector{Expression},
    vars::Vector{Variable},
    point::Vector{Float64},
    step_size::Float64,
    direction_sign::Float64;
    JG = HC.differentiate(G, vars),
    projection_maxiter::Int64 = 5,
    projection_tol::Float64 = 1e-15
    )

    k1 = projected_routing_vector_field(r, G, vars, point, direction_sign; JG = JG)
    k2 = projected_routing_vector_field(r, G, vars, point .+ 0.5 .* step_size .* k1, direction_sign; JG = JG)
    k3 = projected_routing_vector_field(r, G, vars, point .+ 0.5 .* step_size .* k2, direction_sign; JG = JG)
    k4 = projected_routing_vector_field(r, G, vars, point .+ step_size .* k3, direction_sign; JG = JG)

    point .+= (step_size / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    project_to_variety!(point, G, vars; JG = JG, maxiter = projection_maxiter, tol = projection_tol)

    return point
end

# computes minimum distance to an index 0 saddle point
function distance_to_endpoints(
    point::Vector{Float64},
    index0_points::Vector{Vector{Float64}}
    )

    return minimum(LA.norm.([Q - point for Q in index0_points]))

end

# moves in direction of gradient, then projects back to variety
function gradient_flow!(
    r::routing_function,
    point::Vector{Float64},
    gradient::Vector{Float64},
    G::Vector{Expression},
    vars::Vector{Variable},
    step_size::Float64,
    index0_points::Vector{Vector{Float64}};
    tol::Float64 = 1e-2,
    max_steps::Int64 = 10_000,
    method::Symbol = :rk4,
    projection_maxiter::Int64 = 5,
    projection_tol::Float64 = 1e-15,
    Verbose::Bool = false
    )

    JG = HC.differentiate(G, vars)
    direction_sign = sign(r.eval(point))
    if direction_sign == 0
        direction_sign = sign(first(gradient))
    end

    count = 0
    while distance_to_endpoints(point, index0_points) > tol && count < max_steps
        if method == :euler
            gradient .= projected_routing_vector_field(r, G, vars, point, direction_sign; JG = JG)
            point .+= gradient .* step_size
            project_to_variety!(point, G, vars; JG = JG, maxiter = projection_maxiter, tol = projection_tol)
        elseif method == :rk4
            rk4_projected_step!(r, G, vars, point, step_size, direction_sign; JG = JG, projection_maxiter = projection_maxiter, projection_tol = projection_tol)
            gradient .= projected_routing_vector_field(r, G, vars, point, direction_sign; JG = JG)
        else
            throw(ArgumentError("unknown path tracking method $(method); use :rk4 or :euler"))
        end

        if (count % 100 == 0) && Verbose
            println(distance_to_endpoints(point, index0_points))
        end
        count += 1
    end

    return point
end

# finds P ± ϵv for unstable LA.eigenvectors v at a critical point P
function find_starting_points_for_flow(
    critical_point::Vector{Float64},
    H::Matrix{Float64}, # hessian of r at crit_point
    V::Matrix{Float64},  # basis of tangent space of V(G) at V
    G::Vector{Expression},
    r::routing_function;
    step_size::Float64 = 0.1
    )

    n, d = size(V)

    sgn = sign(r.eval(critical_point))
    E = LA.eigen(H)
    unstable_vecs = [
        V * E.vectors[:,i]
        for i in 1:d
            if sign(E.values[i]) == sgn
    ]

    JG = HC.differentiate(G, r.vars)
    pos_starting_points = [project_to_variety!(critical_point + v*step_size, G, r.vars; JG = JG) for v in unstable_vecs]
    neg_starting_points = [project_to_variety!(critical_point - v*step_size, G, r.vars; JG = JG) for v in unstable_vecs]

    return vcat(pos_starting_points, neg_starting_points)
end

# Track the projected routing gradient flow from a critical point.
function solve_ivp(
    r::routing_function,
    G::Vector{Expression},
    initial_point::Vector{Float64},
    final_points::Vector{Vector{Float64}};
    grad_step_size::Float64 = 0.05,
    start_step_size::Float64 = 0.1,
    tol::Float64 = 1e-2,
    max_steps::Int64 = 10_000,
    method::Symbol = :rk4,
    projection_maxiter::Int64 = 5,
    projection_tol::Float64 = 1e-15,
    Verbose::Bool = false
    )

    sgn = sign(r.eval(initial_point))
    V = LA.nullspace(HC.evaluate(HC.differentiate(G, r.vars), r.vars => initial_point))
    H = hessian(r, G, initial_point)
    starts = find_starting_points_for_flow(initial_point, H, V, G, r; step_size = start_step_size)

    solns = []

    for P in starts
        Q = copy(P)
        gradient_flow!(r, Q, r.eval_grad(Q) * sgn, G, r.vars, grad_step_size, final_points; tol = tol, max_steps = max_steps, method = method, projection_maxiter = projection_maxiter, projection_tol = projection_tol, Verbose = Verbose)
        push!(solns, Q)
    end

    return solns
end
