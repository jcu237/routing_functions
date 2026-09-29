module ConnectedComponents

using HomotopyContinuation, LinearAlgebra
# imported, not used: SciMLBase and HomotopyContinuation both export `solve`, and
# bringing both into scope would make the `solve` this module re-exports ambiguous
# for everyone who loads it. every SciML call below is qualified.
import SciMLBase, OrdinaryDiffEq

const HC = HomotopyContinuation
const LA = LinearAlgebra

using Reexport: @reexport
@reexport using HomotopyContinuation

# the files follow the pipeline, and each only uses what the ones before it define
include("utils.jl")             # logging and old-keyword helpers
include("routing_functions.jl") # r = f / gᵈ and its derivatives; distance to V(f)
include("variety.jl")           # X = V(G): normal equations, projection onto X
include("gradient_field.jl")    # the projected gradient field of r on X
include("routing_system.jl")    # the Lagrange system whose solutions are the routing points
include("cache.jl")             # RoutingCache: everything precomputed from (r, G)
include("hessian.jl")           # hessian of r on X, criticality, morse indices
include("routing_points.jl")    # finding the routing points: flows, Newton, monodromy
include("mountain_pass.jl")     # flowing from a routing point up to the maxima it reaches
include("connectivity.jl")      # the graph of routing points joined by those flows
include("components.jl")        # grouping it into components with Euler characteristics

# the whole pipeline
export RoutingFunction, RoutingCache, connected_components, Component, euler_characteristic

# stage by stage: routing points, their indices, the paths between them
export routing_points, flow_to_routing_points, routing_system
export morse_index, routing_point_indices, sort_routing_points_by_index
export find_connectivity_matrix, solve_ivp, component_labels

# evaluating r = f / gᵈ and measuring distances
export evaluate_r, evaluate_f, evaluate_g, evaluate_grad_r, evaluate_grad_hessian_r
export distance_to_zero_locus, on_zero_locus, critical_distance

# the variety X = V(G)
export singular_locus, evaluate_G!, jacobian_G!
export project_to_variety!, project_to_variety, project_to_variety_residual!

# lower-level pieces: hessians on X, gradient fields and single paths
export hessian, hessian_and_tangent, ambient_gradient_hessian, compute_matrices, idx
export projected_gradient_field, gradient_flow!, find_starting_points_for_flow
export distance_to_endpoints, nearest_index

end
