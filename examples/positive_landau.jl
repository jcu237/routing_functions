# Positive Landau discriminants.
#
# F is the second Symanzik polynomial of a two-loop graph with four propagators (the para):
# cubic and homogeneous in the Schwinger parameters x = (x1,...,x4), linear in the
# kinematic parameters p = (s,m,M). The incidence variety
#
#     V = {(x,p) ∈ (ℂ*)^4 × ℂ^3 : F = ∂F/∂x1 = ... = ∂F/∂x4 = 0}
#
# is the object whose projection to p-space is the Landau discriminant. The
# question here is which of its irreducible components carry a point whose x is
# real and strictly positive.

using ConnectedComponents
using LinearAlgebra
using Random

@var x1 x2 x3 x4 s m M
X = [x1, x2, x3, x4]
p = [s, m, M]
Z = [X; p]

F = m*x2*x3^2 + m*x2^2*x3 + m*x3*x1^2 + m*x3^2*x1 + m*x4*x1^2 + m*x4*x2^2 +
    m*x4*x3^2 + m*x4^2*x1 + m*x4^2*x2 + m*x4^2*x3 - M*x4*x2*x3 - M*x4*x3*x1 +
    2*m*x2*x3*x1 + 2*m*x4*x2*x1 + 3*m*x4*x2*x3 + 3*m*x4*x3*x1 -
    s*x2*x3*x1 - s*x4*x2*x1

dF = differentiate(F, X)

# since (ℂ*)^4 × {p = 0} is a component of V, and V is a cone in x and in p, we fix
# one affine chart in each. ℓ is generic, and ℓ·p = 1 removes p = 0, which is
# trivially positive and on which the jacobian of dF is rank deficient. δ > 0, so
# every positive x rescales onto δ·x = 1 and no positive point is lost.
δ = [0.31, 0.57, 0.83, 1.19]
ℓ = [0.37, -0.62, 0.91]
G = System([dF; sum(δ .* X) - 1; sum(ℓ .* p) - 1], Z)

# 6 equations in 7 unknowns. Y = V(G) is one surface and six curves -- the seven
# irreducible components of V that have p ≠ 0, sliced. Slicing does not break them
# up: each component is a cone in both x and p, so it maps onto its slice by
# rescaling, and the image of an irreducible variety is irreducible.


r = RoutingFunction(x1*x2*x3*x4, Z)
cache = RoutingCache(r, G)


is_positive(P; tol = 1e-4) = all(>(tol), view(P, 1:4))
F
Random.seed!(1)
hit = routing_points(cache; stop_when = is_positive, nstarts = 200,
                     box = 10.0, verbose = true)
if isempty(hit)
    println("no positive routing point found; raise nstarts")
else
    q = hit[1]
    println("positive routing point")
    println("   x = ", round.(q[1:4], sigdigits = 6))
    println("   p = ", round.(q[5:7], sigdigits = 6))
    println("   |G(q)| = ", norm(Float64.(evaluate(G, q))))
end


#Using the above we quickly get a positive routing point. Maybe we instead want to find all positive routing points.
#We can do this by dropping the stop_when predicate.

rp = routing_points(cache; nstarts = 1000, box = 100.0, verbose = true)

#with result of that computation in hand, we can filter the positive routing points from the result.
positive_rp = filter(p -> is_positive(p), rp)



#Lets see if these routing points belong to the same connected component.

components = connected_components(cache, positive_rp; grad_step_size = 1.0, tol = 2.0, start_step_size = 1.1,
                         path_options = (tspan = (0.0, 1e5),))

#We see that each of the positive routing points is put in a different connected component, but this is not right here.
#The line and the cubic found below cross at a point with x > 0, where V(G) is singular, and f = x1*x2*x3*x4 does not remove it.
#So instead we take a numerical irreducible decomposition to see which irreducible components the positive routing points lie on.
H = System(vcat(F,dF), Z)
irreducible_decomposition = nid(H)
witnessSets = witness_sets(irreducible_decomposition)

# witnessSets maps each dimension d to a vector of WitnessSets, one per irreducible
# component of that dimension, so there are two levels to loop over. membership
# takes a vector of points and sets up its homotopy once per witness set, so all of
# positive_rp is tested against a component in a single call.
for (d, Ws) in witnessSets, (i, W) in enumerate(Ws)
    in_W = membership(positive_rp, W; show_progress = false)
    for j in findall(in_W)
        println("positive routing point $j lies on component $i of dimension $d")
    end
end

#The positive routing points lie on 2 of the 7 components of V(G): the cubic x1 = x2, x3 = x4 and the line s = 0, M = 9m, x3 = x4 = x1 + x2.
#The third positive component of V is p = 0, which the chart ℓ·p = 1 removes.
#Over ℚ, V also has 7 components, since p = 0 is added and the two complex conjugate lines of V(G) count as one. So 3 of these 7 meet the positive orthant.
#positive_landau.m2 checks this exactly.

