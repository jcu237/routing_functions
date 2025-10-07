module ConnectedComponents

import LinearAlgebra
import HomotopyContinuation

const HC = HomotopyContinuation
const LA = LinearAlgebra

using Reexport: @reexport
@reexport using HomotopyContinuation

include("routing_functions.jl")
include("hessian.jl")
include("path_tracking.jl")

export routing_function, routing_system, routing_points, ambient_gradient_hessian, compute_matrices, hessian, idx, sort_routing_points_by_index, project_to_variety!, distance_to_endpoints, gradient_flow!, find_starting_points_for_flow, solve_ivp

end