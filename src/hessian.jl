# computes gradient and hessian of routing function in ℝ^n 
function ambient_gradient_hessian(
    r::routing_function,
    point::Vector{Float64}
    )

    ∇r = HC.differentiate(r.f/r.g^r.d, r.vars)
    Hr = HC.evaluate(HC.differentiate(∇r, r.vars), r.vars => point)
    return HC.evaluate(∇r, r.vars => point), Hr
end

# computes gradient and hessian of any expression in ℝ^n
function ambient_gradient_hessian(
    g::Expression, 
    vars::Vector{Variable},
    point::Vector{Float64}
    )

    ∇g = HC.evaluate(HC.differentiate(g, vars), vars => point)
    Hg = HC.evaluate(HC.differentiate(∇g, vars), vars => point)
    return ∇g, Hg
end

# computes gradient of routing function at a point on X = V(G)
function gradient(
    r::routing_function, 
    G::Vector{Expression},
    point::Vector{Float64},
    )

    ∇r = reshape(r.eval_grad(point), 1, :)
    JG = HC.evaluate(HC.differentiate(G, r.vars), r.vars => point)
    V = LA.nullspace(JG)
    
    return ∇r * V
end

# gets W matrices, really fucking annoying
function compute_matrices(
    G::Vector{Expression},
    vars::Vector{Variable},
    point::Vector{Float64}
    )

    
    # i^th row is ∇G[i]
    JG = HC.evaluate(HC.differentiate(G, vars), vars => point)

    # columns form an orthonomral basis of T_x(V(G))
    V = LA.nullspace(JG)

    n,d = size(V)

    # set up well-constrained linear system
    I_d = Matrix{Float64}(LA.I, d, d)
    A1s = [hcat((JG[j,i] * I_d for i in 1:n)...) for j in eachindex(G)]
    A1 = vcat(A1s...)

    A2s = [hcat((V[i,j] * I_d for i in 1:n)...) for j in 1:d]
    A2 = vcat(A2s...)

    HG = [-1 * V' * HC.evaluate(HC.differentiate(vec(HC.differentiate(g, vars)), vars), vars => point) * V for g in G]
    B = vcat(vcat(HG...), zeros(d^2,d))

    W = vcat(A1,A2) \ B

    return [W[(i-1)*d+1 : i*d, :] for i in 1:n], V

end


# computes hessian of routing function on a variety X = V(G) at a point
function hessian(
    r::routing_function,
    G::Vector{Expression},
    point::Vector{Float64}
    )

    W, V = compute_matrices(G, r.vars, point)
    ∇r, Hr = ambient_gradient_hessian(r, point)

    return sum([W[i] * ∇r[i] for i in eachindex(W)]) + transpose(V) * Hr * V
end

# computes hessian of any Expression, f, on V(G) at a specified point
function hessian(
    f::Expression,
    G::Vector{Expression},
    vars::Vector{Variable},
    point::Vector{Float64}
    )

    W, V = compute_matrices(G, vars, point)
    ∇f = HC.differentiate(f, vars)
    Hf = HC.evaluate(HC.differentiate(∇f, vars), vars => point)
    ∇f = HC.evaluate(∇f, vars => point)

    return sum([W[i] * ∇f[i] for i in eachindex(W)]) + transpose(V) * Hf * V
end

# computes the index of a critical point by counting the number of 
# LA.eigenvalues of Hr(crit) that match the sign of r(crit)
function idx(
    r::routing_function,
    H::Matrix{Float64},
    critical_point::Vector{Float64}
    )::Int64

    σ = sign(r.eval(critical_point))
    E = LA.eigen(H)
    i = 0
    for eval in E.values
        if sign(eval) == σ
            i += 1
        end
    end

    return i
end

function sort_routing_points_by_index(
    r::routing_function,
    G::Vector{Expression},
    critical_points::Vector{Vector{Float64}}
    )

    hessians = [hessian(r, G, P) for P in critical_points]

    sorter = Dict{Int, Vector{Vector{Float64}}}()

    for (H,P) in zip(hessians, critical_points)
        ind = idx(r, H, P)
        if haskey(sorter, ind)
            push!(sorter[ind], P)
        else
            sorter[ind] = [P]
        end
    end

    return sorter
end

