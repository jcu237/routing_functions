# the graph whose vertices are the routing points and whose edges are the gradient
# paths between them, and its connected components.

# component label of every routing point, numbered in order of first appearance.
#
# `A` may be the raw adjacency matrix or the reflexive-transitive closure
# `find_connectivity_matrix` returns -- the traversal only reads which entries are
# nonzero, and both matrices have the same connected components.
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
            # the matrix is symmetric as built, but a path recorded in one
            # direction only should still join the two points
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
# true exactly when i and j lie in the same connected component. read off the
# component labels, which costs O(n²) -- the closure as a Boolean power sum
# ⋁ₖ Aᵏ costs O(n⁴).
function reachability(A::AbstractMatrix)::BitMatrix
    labels = component_labels(A)
    return labels .== transpose(labels)
end

"""
    find_connectivity_matrix(cache; kwargs...) -> (A, routing_points)
    find_connectivity_matrix(cache, routing_points; kwargs...) -> (A, routing_points)

Finds the routing points (or takes the ones given), and joins each routing point of
positive index to the index 0 routing points that the gradient flow of `|r|` reaches
from it ([`solve_ivp`](@ref)). Returns the reachability matrix `A` of that graph --
`A[i, j]` is true exactly when routing points `i` and `j` lie on the same connected
component of `V(G) ∖ V(f)` -- together with the routing points, in the same order.

Keywords for the paths: `start_step_size = 0.1`, `max_halvings = 30`, `tol = 1e-2`,
`grad_step_size = 0.05` (see [`solve_ivp`](@ref)), and `path_options = (;)`, e.g.
`(tspan = (0.0, 1e4),)`, for [`gradient_flow!`](@ref). Any other keyword (`nstarts`,
`box`, `starts`, `locus_tol`, `monodromy_options`, ...) goes to
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
    any(==(0), indices) || error(
        "find_connectivity_matrix: none of the ", length(routPoints), " routing points has ",
        "index 0. every component of V(G) \\ V(f) carries a maximum of |r|, so the search ",
        "missed some; try more starts (`nstarts`, `box`, `starts`).",
    )
    sgn = [sign(evaluate_r(cache.r, P)) for P in routPoints]
    index0 = findall(==(0), indices)

    A = Matrix{Int64}(LA.I, length(routPoints), length(routPoints))

    for i in eachindex(routPoints)
        indices[i] == 0 && continue
        P = routPoints[i]
        # a path from P stays in P's region, where r has P's sign
        targets = [j for j in index0 if sgn[j] == sgn[i]]
        if isempty(targets)
            @warn("find_connectivity_matrix: a routing point of index $(indices[i]) has no " *
                  "index 0 routing point of its sign to flow to; the search missed at least " *
                  "one maximum of |r|", routing_point = P)
            continue
        end
        solns = solve_ivp(cache, P, routPoints[targets]; grad_step_size = grad_step_size,
                          start_step_size = start_step_size, max_halvings = max_halvings,
                          tol = tol, path_options...)
        verbose && println("find_connectivity_matrix: $(length(solns)) of $(2 * indices[i]) ",
                           "paths from an index $(indices[i]) point arrived")
        for x in solns
            j = targets[nearest_index(x, routPoints[targets])]
            A[i, j] = 1
            A[j, i] = 1
        end
    end

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
