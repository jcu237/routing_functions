function boolean_power_sum(A::AbstractMatrix)
    # Validate shape
    size(A,1) == size(A,2) || throw(ArgumentError("A must be square"))
    n = size(A,1)

    # Normalize to Bool (accepts 0/1 integers too)
    Ab = Matrix{Bool}(A .!= 0)
    n = size(A,2)
    # Accumulator and current power
    S = falses(n, n)
    P = copy(Ab)  # P = A^k (Boolean)

    for k in 1:n
        # S ← S ∨ P
        @inbounds S .= S .| P
        # Next power if needed: P ← P ⊗ A  (Boolean product)
        if k < n
            P = boolean_matmul(P, Ab)
        end
    end

    return S
end

# Boolean matrix multiplication: (A ⊗ B)[i,j] = any_k (A[i,k] & B[k,j])
function boolean_matmul(A::AbstractMatrix{Bool}, B::AbstractMatrix{Bool})
    size(A,2) == size(B,1) || throw(ArgumentError("Inner dimensions must match"))
    n, m = size(A)
    _, p = size(B)
    C = falses(n, p)

    @inbounds for i in 1:n
        for j in 1:p
            val = false
            for k in 1:m
                if A[i,k] & B[k,j]
                    val = true
                    break  # short-circuit as soon as we find a witness
                end
            end
            C[i,j] = val
        end
    end
    return C
end


A = [1 0 1 0 0 0; 0 1 1 0 0 0; 1 1 1 0 0 0; 0 0 0 1 0 1; 0 0 0 0 1 1; 0 0 0 1 1 1]
boolean_power_sum(A)


function find_connectivity_matrix(
    r::routing_function,
    G::Vector{Expression};
    grad_step_size::Float64 = 0.05,
    start_step_size::Float64 = 0.1,
    tol::Float64 = 1e-2
    )

    routPoints = routing_points(r, G)
    index_dict = sort_routing_points_by_index(r, G, routPoints)
    final_points = index_dict[0]
    initial_points = vcat([v for (k,v) in index_dict if k != 0]...)

    A = Matrix{Int64}(LA.I, length(routPoints), length(routPoints))

    for P in initial_points
        solns = solve_ivp(r, G, P, final_points; grad_step_size = grad_step_size, start_step_size = start_step_size, tol = tol)
        P_ind = findfirst(==(P), routPoints)
        for x in solns
            d = distance_to_endpoints(x, final_points)
            x_ind = findfirst(==(d), [LA.norm(Q - x) for Q in routPoints])
            A[P_ind, x_ind] = 1
            A[x_ind, P_ind] = 1
        end
    end

    return boolean_power_sum(A), routPoints
end