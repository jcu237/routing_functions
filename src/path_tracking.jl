# update to do an actual LA.normal projection later
function project_to_variety!(
    point::Vector{Float64},
    G::Vector{Expression},
    vars::Vector{Variable};
    maxiter::Int64 = 5,
    tol::Float64 = 1e-15
    )

    JG = HC.differentiate(G, vars)


    for _ in 1:maxiter
        Gx = HC.evaluate(G, vars => point)
        JGx = HC.evaluate(JG, vars => point)
        point .-= JGx \ Gx

        if LA.norm(Gx) < tol
            return point
        end
    end

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
    Verbose::Bool = false
    )
    count = 0
    while distance_to_endpoints(point, index0_points) > tol

        point .+= gradient * step_size ./ LA.norm(gradient)
        project_to_variety!(point, G, vars)
        gradient .= r.eval_grad(point) * sign(r.eval(point))
        
        if (count % 100 == 0) && Verbose
            println(distance_to_endpoints(point, index0_points))
        end
        count += 1
    end
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

    pos_starting_points = [project_to_variety!(critical_point + v*step_size, G, r.vars) for v in unstable_vecs]
    neg_starting_points = [project_to_variety!(critical_point - v*step_size, G, r.vars) for v in unstable_vecs]

    return vcat(pos_starting_points, neg_starting_points)
end


# 
function solve_ivp(
    r::routing_function,
    G::Vector{Expression},
    initial_point::Vector{Float64},
    final_points::Vector{Vector{Float64}};
    grad_step_size::Float64 = 0.05,
    start_step_size::Float64 = 0.1,
    tol::Float64 = 1e-2,
    Verbose::Bool = false
    )
    
    
    sgn = sign(r.eval(initial_point))
    V = LA.nullspace(HC.evaluate(HC.differentiate(G, r.vars), r.vars => initial_point))
    H = hessian(r, G, initial_point)
    starts = find_starting_points_for_flow(initial_point, H, V, G, r; step_size = start_step_size)

    solns = []

    for P in starts
        Q = copy(P)
        gradient_flow!(r, Q, r.eval_grad(Q) * sgn, G, r.vars, grad_step_size, final_points; tol = tol, Verbose = Verbose)
        push!(solns, Q)
    end

    return solns
end

