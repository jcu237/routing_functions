# the end result: the routing points grouped by connected component of V(G) ∖ V(f).

"""
    Component

A connected component of `V(G) ∖ V(f)`, recorded through the routing points on it:
`points[i]` is a routing point and `indices[i]` its Morse index. `r` restricted to the
component is a Morse function with exactly these critical points, so its Euler
characteristic is `euler_characteristic = ∑ (-1)^index`.
"""
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

"""
    euler_characteristic(indices)
    euler_characteristic(component)
    euler_characteristic(components)

`∑ (-1)^index`: the Euler characteristic of a component from the indices of its
routing points, or of all of `V(G) ∖ V(f)` from the list of its components.
"""
euler_characteristic(indices::AbstractVector{<:Integer})::Int64 =
    sum(iseven(i) ? 1 : -1 for i in indices; init = 0)

euler_characteristic(C::Component)::Int64 = C.euler_characteristic

euler_characteristic(Cs::AbstractVector{Component})::Int64 =
    sum(euler_characteristic, Cs; init = 0)

Base.length(C::Component) = length(C.points)

function Base.show(io::IO, C::Component)
    n = length(C.points)
    print(io, "Component: ", n, " routing point", n == 1 ? "" : "s",
          " of index ", isempty(C.indices) ? "--" : join(sort(C.indices), ", "),
          ", χ = ", C.euler_characteristic)
end

"""
    connected_components(cache; kwargs...) -> Vector{Component}
    connected_components(r, G; kwargs...)
    connected_components(cache, routing_points; kwargs...)
    connected_components(cache, routing_points, A)
    connected_components(routing_points, indices, A)

The connected components of `V(G) ∖ V(f)`, each with its routing points, their Morse
indices and its Euler characteristic.

The first form runs the whole pipeline: [`routing_points`](@ref), their indices, and
[`find_connectivity_matrix`](@ref); keywords go to the latter (and through it to
`routing_points`). Pass routing points you already have to skip the search, and a
connectivity matrix `A` as well to skip the paths.

The result is only as complete as the search for routing points: a component
without a routing point found on it is missing from the list.
"""
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

    # every component carries a maximum of |r|, i.e. an index 0 point. a component
    # made only of higher-index points means a path from one of them never arrived,
    # and the count is too high.
    orphans = [c for c = 1:ncomp if !any(j -> labels[j] == c && indices[j] == 0, eachindex(labels))]
    isempty(orphans) || @warn(
        "connected_components: $(length(orphans)) component(s) contain no index 0 routing " *
        "point, so a path from them failed to reach one; the number of components is " *
        "probably overcounted. A smaller `start_step_size` or `tol`, or a longer path " *
        "(`path_options = (tspan = (0.0, 1e4),)`), may help.",
        orphan_points = [routing_pts[j] for j in eachindex(labels) if labels[j] in orphans],
    )

    pts = [Vector{Float64}[] for _ = 1:ncomp]
    inds = [Int64[] for _ = 1:ncomp]
    for (j, l) in enumerate(labels)
        push!(pts[l], routing_pts[j])
        push!(inds[l], indices[j])
    end

    return [Component(pts[c], inds[c]) for c = 1:ncomp]
end

connected_components(
    cache::RoutingCache,
    routing_pts::AbstractVector{<:AbstractVector{<:Real}},
    A::AbstractMatrix,
) = (pts = Vector{Float64}[Vector{Float64}(P) for P in routing_pts];
     connected_components(pts, routing_point_indices(cache, pts), A))

connected_components(
    r::RoutingFunction,
    G,
    routing_pts::AbstractVector{<:AbstractVector{<:Real}},
    A::AbstractMatrix,
) = connected_components(RoutingCache(r, G), routing_pts, A)

function connected_components(
    cache::RoutingCache,
    routing_pts::AbstractVector{<:AbstractVector{<:Real}};
    kwargs...
    )::Vector{Component}
    A, pts = find_connectivity_matrix(cache, routing_pts; kwargs...)
    return connected_components(cache, pts, A)
end

function connected_components(cache::RoutingCache; kwargs...)::Vector{Component}
    A, pts = find_connectivity_matrix(cache; kwargs...)
    return connected_components(cache, pts, A)
end

connected_components(r::RoutingFunction, G; kwargs...) =
    connected_components(RoutingCache(r, G); kwargs...)
