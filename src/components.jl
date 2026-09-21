# a connected component of V(G), recorded through the routing points sitting on
# it: the points themselves, the index of each one, and the Euler characteristic
# they add up to.
#
# r|V(G) restricted to the component is a Morse function whose critical points are
# exactly those routing points, so χ = ∑ (-1)^index -- the alternating sum that
# `sort_routing_points_by_index` is used for by hand in the examples.
#
# `points[i]` has index `indices[i]`; the two are kept parallel rather than in a
# dict so that a component can be read off the connectivity matrix row by row.
struct Component
    points::Vector{Vector{Float64}}
    indices::Vector{Int64}
    euler_characteristic::Int64
end

function Component(points::Vector{Vector{Float64}}, indices::Vector{Int64})
    length(points) == length(indices) || throw(
        ArgumentError("got $(length(points)) points but $(length(indices)) indices"),
    )
    return Component(points, indices, euler_characteristic(indices))
end

# ∑ (-1)^index
euler_characteristic(indices::AbstractVector{<:Integer})::Int64 =
    sum(iseven(i) ? 1 : -1 for i in indices; init = 0)

euler_characteristic(C::Component)::Int64 = C.euler_characteristic

Base.length(C::Component) = length(C.points)

function Base.show(io::IO, C::Component)
    n = length(C.points)
    print(io, "Component: ", n, " routing point", n == 1 ? "" : "s",
          " of index ", isempty(C.indices) ? "--" : join(sort(C.indices), ", "),
          ", χ = ", C.euler_characteristic)
end

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

# groups routing points whose indices are already known
function connected_components(
    routing_pts::Vector{Vector{Float64}},
    indices::Vector{Int64},
    A::AbstractMatrix
    )::Vector{Component}

    length(routing_pts) == size(A, 1) || throw(ArgumentError(
        "got $(length(routing_pts)) routing points but a $(size(A, 1))×$(size(A, 2)) matrix",
    ))
    length(routing_pts) == length(indices) || throw(ArgumentError(
        "got $(length(routing_pts)) routing points but $(length(indices)) indices",
    ))

    labels = component_labels(A)
    ncomp = isempty(labels) ? 0 : maximum(labels)

    pts = [Vector{Float64}[] for _ = 1:ncomp]
    inds = [Int64[] for _ = 1:ncomp]
    for (j, l) in enumerate(labels)
        push!(pts[l], routing_pts[j])
        push!(inds[l], indices[j])
    end

    return [Component(pts[c], inds[c]) for c = 1:ncomp]
end

# the indices are what the Euler characteristic is made of, so they are computed
# here rather than asked for
connected_components(
    cache::RoutingCache,
    routing_pts::Vector{Vector{Float64}},
    A::AbstractMatrix,
) = connected_components(routing_pts, routing_point_indices(cache, routing_pts), A)

connected_components(
    r::RoutingFunction,
    G::Vector{Expression},
    routing_pts::Vector{Vector{Float64}},
    A::AbstractMatrix,
) = connected_components(RoutingCache(r, G), routing_pts, A)

# the whole pipeline: find the routing points, connect them, split them up. the
# search parameters are the ones `find_connectivity_matrix` takes.
function connected_components(cache::RoutingCache; kwargs...)::Vector{Component}
    A, routing_pts = find_connectivity_matrix(cache; kwargs...)
    return connected_components(cache, routing_pts, A)
end

connected_components(r::RoutingFunction, G::Vector{Expression}; kwargs...) =
    connected_components(RoutingCache(r, G); kwargs...)
