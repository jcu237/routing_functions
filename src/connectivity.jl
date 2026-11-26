# the graph whose vertices are the routing points and whose edges are the gradient
# paths between them, and its connected components.

# component label of every routing point, numbered in order of first appearance.
#
# `A` may be an adjacency matrix or the closure `find_connectivity_matrix` returns;
# only which entries are nonzero matters, and both have the same components.
function component_labels(A::AbstractMatrix)::Vector{Int64}
    size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
    n = size(A, 1)

    label = zeros(Int64, n)
    stack = Int64[]
    ncomp = 0

    for s = 1:n
        label[s] == 0 || continue
        ncomp += 1
        label[s] = ncomp
        push!(stack, s)

        while !isempty(stack)
            i = pop!(stack)
            # symmetric as built, but an entry in one direction should still join
            @inbounds for j = 1:n
                if label[j] == 0 && (A[i, j] != 0 || A[j, i] != 0)
                    label[j] = ncomp
                    push!(stack, j)
                end
            end
        end
    end

    return label
end

# the reflexive-transitive closure of the graph with adjacency matrix A: S[i,j] is
# true exactly when i and j lie in the same component. read off the labels in O(n²)
# instead of forming the Boolean power sum ⋁ₖ Aᵏ.
function reachability(A::AbstractMatrix)::BitMatrix
    labels = component_labels(A)
    return labels .== transpose(labels)
end

"""
    find_connectivity_matrix(cache; kwargs...) -> (A, routing_points)
    find_connectivity_matrix(cache, routing_points; kwargs...) -> (A, routing_points)

Finds the routing points (or takes the ones given) and joins each routing point of
positive index to the index 0 routing points that the gradient flow of `|r|` reaches
from it ([`solve_ivp`](@ref)). Returns the reachability matrix `A` of that graph, where
`A[i, j]` is true exactly when routing points `i` and `j` lie on the same connected
component of `V(G) ∖ V(f)`, together with the routing points in the order of `A`.

Keywords: `start_step_size = 0.1`, `max_halvings = 30`, `tol = 1e-2`,
`grad_step_size = 0.05` (see [`solve_ivp`](@ref)); `path_options = (;)`, extra keywords
for `solve_ivp`, e.g. `(tspan = (0.0, 1e4),)`; `verbose = false`. In the first form any
other keyword (`nstarts`, `box`, `starts`, `locus_tol`, ...) goes to
[`routing_points`](@ref).
"""
function find_connectivity_matrix(
    cache::RoutingCache,
    routPoints::AbstractVector{<:AbstractVector{<:Real}};
    grad_step_size::Real = 0.05,
    start_step_size::Real = 0.1,
    max_halvings::Integer = 30,
    tol::Real = 1e-2,
    path_options::NamedTuple = (;),
    verbose::Bool = false
    )

    routPoints = Vector{Float64}[Vector{Float64}(P) for P in routPoints]
    isempty(routPoints) && return falses(0, 0), routPoints

    indices = routing_point_indices(cache, routPoints)
    sgn = [sign(evaluate_r(cache.r, P)) for P in routPoints]
    index0 = findall(==(0), indices)

    A = Matrix{Int64}(LA.I, length(routPoints), length(routPoints))
    npaths = nlost = 0

    for i in eachindex(routPoints)
        indices[i] == 0 && continue
        # a path from P stays in P's region, where r has P's sign
        targets = [j for j in index0 if sgn[j] == sgn[i]]
        solns = isempty(targets) ? Vector{Float64}[] :
            solve_ivp(cache, routPoints[i], routPoints[targets]; grad_step_size = grad_step_size,
                      start_step_size = start_step_size, max_halvings = max_halvings,
                      tol = tol, path_options...)
        npaths += 2 * indices[i]
        nlost += 2 * indices[i] - length(solns)
        verbose && println("find_connectivity_matrix: $(length(solns)) of $(2 * indices[i]) ",
                           "paths from an index $(indices[i]) point arrived")
        for x in solns
            j = targets[nearest_index(x, routPoints[targets])]
            A[i, j] = 1
            A[j, i] = 1
        end
    end

    # every path ends at a maximum of |r|; one that reaches none means a maximum was
    # missed or the flow stopped early
    nlost > 0 && @warn(
        "find_connectivity_matrix: $nlost of $npaths paths reached no index 0 routing point, " *
        "so some components may be reported as split. either the search missed a maximum " *
        "of |r| (try more starts: `nstarts`, `box`, `starts`) or the flow stopped early " *
        "(raise `tspan` in `path_options`).",
    )

    return reachability(A), routPoints
end

function find_connectivity_matrix(
    cache::RoutingCache;
    grad_step_size::Real = 0.05,
    start_step_size::Real = 0.1,
    max_halvings::Integer = 30,
    tol::Real = 1e-2,
    path_options::NamedTuple = (;),
    verbose::Bool = false,
    search_options...
    )

    routPoints = routing_points(cache; verbose = verbose, search_options...)
    if isempty(routPoints)
        @warn(
            "find_connectivity_matrix: no routing points were found, so no components are " *
            "reported. either V(G)(ℝ) \\ V(f) is empty, or the search missed it: the flows " *
            "start from random points of [-box, box]^n projected onto V(G), so if V(G) lies " *
            "far from the origin raise `box`, or pass your own starting points with `starts`.",
        )
        return falses(0, 0), routPoints
    end
    return find_connectivity_matrix(cache, routPoints; grad_step_size = grad_step_size,
                                    start_step_size = start_step_size, max_halvings = max_halvings,
                                    tol = tol, path_options = path_options, verbose = verbose)
end

find_connectivity_matrix(r::RoutingFunction, G; kwargs...) =
    find_connectivity_matrix(RoutingCache(r, G); kwargs...)
