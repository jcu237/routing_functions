using ConnectedComponents


@var x[1:2]
r = routing_function(x[1]^2 + x[2]^2 - 9, x[1:2], [0.855, -1.632])
G = [x[2]^2 - x[1]*(x[1] - 1)*(x[1] + 1)]
routPoints = routing_points(r, G)
index_dict = sort_routing_points_by_index(r, G, routPoints)
final_points = index_dict[0]
initial_points = index_dict[1]

solns = solve_ivp(r, G, initial_points[1], final_points)


@var x[1:3]
r = routing_function(x[1]*x[2]*x[3], x[1:3])
G = [x[1]^3 - x[3], x[1]^2 - x[2]]
routPoints = routing_points(r, G)
index_dict = sort_routing_points_by_index(r, G, routPoints)
final_points = index_dict[0]


@var x[1:2]
r = routing_function(x[1]*x[2], [1/3,1/2])
G = [x[1]^4 + x[2]^4 - (x[1] - x[2])^2 * (x[1]+x[2])]
routPoints = routing_points(r, G)
index_dict = sort_routing_points_by_index(r, G, routPoints)


#ding dong
@var x[1:3]
g = x[1]^2 + x[2]^2 - x[3]^2 + x[3]^3
f = sum(differentiate(g, x[1:3]).^2)
r = routing_function(f, [0.7978234324, 0.6623073432, 0.2347907832])
G = [g]
routPoints = routing_points(r, G)
index_dict = sort_routing_points_by_index(r, G, routPoints)
final_points = index_dict[0]
initial_points = vcat(index_dict[1], index_dict[2])

# analyzing these gives 2 connected components. 
# i.e. the two index 0 routing points lie in distinct smoothly connected components
solns1 = solve_ivp(r, G, initial_points[1], final_points)
solns2 = solve_ivp(r, G, initial_points[2], final_points)
solns3 = solve_ivp(r, G, initial_points[3], final_points)








@var x[1:2]

G = [x[1]^2 + x[2]^2 - 1]
P = [2., 2.]

project_to_variety!(P, G, x[1:2])


@var x[1:2]
r = routing_function(one(Expression), x[1:2])
G = [x[2]^2 - x[1]*(x[1]-1)*(x[1]+1)]
critical_points = routing_points(r, G; zero_tol = 1e-15)
hessians = [hessian(r, G, p) for p in critical_points]
indices = [idx(r, hessians[i], critical_points[i]) for i in eachindex(hessians)]


r = routing_function(x[1]*x[2], x[1:2], [1/3, 1/2])
G = [x[1]^4 + x[2]^4 - (x[1] - x[2])^2*(x[1]+x[2])]
critical_points = routing_points(r, G; zero_tol = 1e-15)
hessians = [hessian(r, G, p) for p in critical_points]
indices = [idx(r, hessians[i], critical_points[i]) for i in eachindex(hessians)]


# example 2.5a, same as paper
G = [x[1]^2 - x[2]^2]
W, V = compute_matrices(G, x[1:2], [1.0,1.0])

# example 2.5b, different from paper, but its because it chooses a different basis for tangent space
# if you use the same V_x as paper, the W matrices are the same
@var x[1:3]
G = [x[1]^2 - x[2]^2*x[3]]
W, V = compute_matrices(G, x[1:3], [1.0,1.0,1.0])
H = hessian(4*x[1]^2 + 4*x[2]^2*x[3]^2 + x[2]^4, G, x[1:3], [1.,1.,1.])



# these are matrices from paper
P = hcat((1/sqrt(2) * [1 1/3; 1 -1/3; 0 4/3]), [2/3; -2/3; -1/3])
Q = hcat(V, [2/3; -2/3; -1/3])
# make right change of basis
R = (transpose(Q) * P)[1:2,1:2]

# do change of basis to get matrices from 2.5b
WW1 = transpose(R) * W[1] * R
WW2 = transpose(R) * W[2] * R
WW3 = transpose(R) * W[3] * R

# check these are the same from paper
round.(WW1; digits = 10) == round.([0 4/27; 4/27 -16/81]; digits = 10)
round.(WW2; digits = 10) == round.([0 -4/27; -4/27 16/81]; digits = 10)
round.(WW3; digits = 10) == round.([0 -2/27; -2/27 8/81]; digits = 10)

# do the same for the Hessian
HH = transpose(R) * H * R
round.(HH; digits = 10) == round.(2/81 * [567 303; 303 127]; digits = 10)



# from section 6 in paper, Chubs
@var x[1:3]
g = x[1]^4 + x[2]^4 + x[3]^4 - (x[1]^2 + x[2]^2 + x[3]^2) + 1/2
f = sum(differentiate(g, x[1:3]).^2)
c = [0.7978234324, 0.6623073432, 0.2347907832]
r = routing_function(f, x[1:3], c)
G = [g]
critical_points = routing_points(r, G; zero_tol = 1e-16)
hessians = [hessian(r, G, p) for p in critical_points];
indices = [idx(r, hessians[i], critical_points[i]) for i in eachindex(hessians)];

counter = Dict{Int, Int}();

for x in indices
    counter[x] = get(counter, x, 0) + 1
end

counter

euler = counter[0] - counter[1] + counter[2]


# F = routing_system(r,G)

# open("chubs_bertini_run/input", "w") do IO
#     for eq in F.expressions
#         println(IO, eq)
#     end
# end

# for chubs
bertini_solutions = read_real_parts("chubs_bertini_run/real_finite_solutions")

bertini_solutions[1:120, 1:3]

solns = [vec(bertini_solutions[i, 1:3]) for i in 1:120 if abs(evaluate(f, x[1:3] => vec(bertini_solutions[i,1:3]))) > 1e-15]

r.eval_grad.(solns)
