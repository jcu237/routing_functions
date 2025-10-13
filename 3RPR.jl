using ConnectedComponents
using LinearAlgebra

a = [0, 14, 7] #.*0.1
b = [0, 0, 10] #.*0.1
A = [0, 16, 9] #.*0.1
B = [0, 0, 6] #.*0.1
c3 = 100 #0.1^2
P1 = [0,0] #.*0.1
P2 = [a[2],0] #.*0.1
P3 = [a[3],b[3]] #.*0.1



@var p[1:2], ϕ[1:2], c[1:2]

F = [
    ϕ[1]^2 + ϕ[2]^2 - 1,
    p[1]^2 + p[2]^2 - 2*(a[3]*p[1] + b[3]*p[2])*p[1] + 2*(b[3]*p[1] - a[3]*p[2])*ϕ[2] + a[3]^2 + b[3]^2 - c[1],
    p[1]^2 + p[2]^2 - 2*A[2]*p[1] + 2*((a[2]-a[3])*p[1] - b[3]*p[2] + A[2]*a[3] - A[2]*a[2])*ϕ[1] + 2*(b[3]*p[1]+(a[2]-a[3])*p[2] - A[2]*b[3])*ϕ[2] + (a[2]-a[3])^2 + b[3]^2 + A[2]^2 - c[2],
    p[1]^2 + p[2]^2 - 2*(A[3]*p[1] + B[3]*p[1]) + A[3]^2 + B[3]^2 - c3
]

vars = vcat(p[1:2], ϕ[1:2], c[1:2])

JF = differentiate(F, vars)

r = routing_function(det(JF[1:4,1:4]), vars)

routPoints = routing_points(r, F)

index_dict = sort_routing_points_by_index(r, F, routPoints)

initial_points = vcat([v for (k,v) in index_dict if k > 0]...)
final_points = index_dict[0]

solve_ivp(r, F, initial_points[1], final_points; Verbose = true, grad_step_size = 1.0, tol = 2.0, start_step_size = 1.1)
solve_ivp(r, F, initial_points[2], final_points; Verbose = true, grad_step_size = 1.0, tol = 2.0, start_step_size = 1.1)
solve_ivp(r, F, initial_points[3], final_points; Verbose = true, grad_step_size = 1.0, tol = 2.0, start_step_size = 1.1)
solve_ivp(r, F, initial_points[4], final_points; Verbose = true, grad_step_size = 1.0, tol = 2.0, start_step_size = 1.1)
solve_ivp(r, F, initial_points[5], final_points; Verbose = true, grad_step_size = 1.0, tol = 2.0, start_step_size = 1.1)
solve_ivp(r, F, initial_points[6], final_points; Verbose = true, grad_step_size = 1.0, tol = 2.0, start_step_size = 1.1)
solve_ivp(r, F, initial_points[7], final_points; Verbose = true, grad_step_size = 1.0, tol = 2.0, start_step_size = 1.1)
solve_ivp(r, F, initial_points[8], final_points; Verbose = true, grad_step_size = 1.0, tol = 2.0, start_step_size = 1.1)




M, routPoints = find_connectivity_matrix(r, F; grad_step_size = 1.0, tol = 2.0, start_step_size = 1.1)

