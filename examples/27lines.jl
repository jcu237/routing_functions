
#Cubic surface with the 27 lines removed.
function TwentySevenLines()
    @var x,y,z
    @var a[1:4,1:4,1:4]
    terms = []
    for i in 0:3
        for j in 0:3
            for k in 0:3
                if i+j+k<=3
                    push!(terms,[i,j,k])
                end
            end
        end
    end
    f = sum([a[c[1]+1,c[2]+1,c[3]+1]*x^c[1]*y^c[2]*z^c[3] for c in terms])
    Params = [a[c[1]+1,c[2]+1,c[3]+1] for c in terms]

    @var t,b[1:2],c[1:2]

    lx = t
    ly = b[1]*t+b[2]
    lz = c[1]*t+c[2]

    g = subs(f,[x,y,z]=>[lx,ly,lz])
    Eqs = coefficients(g,[t])

    F = System(Eqs,variables=[b[1],b[2],c[1],c[2]],parameters=Params)
end

# the Clebsch cubic, in its usual affine form. it is smooth and all 27 of its
# lines are real -- the reason to use it here rather than a random cubic, whose
# lines come in complex conjugate pairs.
@var w x y z
@var a[1:4,1:4,1:4]

clebsch = 81*(x^3 + y^3 + z^3) -
          189*(x^2*y + x^2*z + x*y^2 + y^2*z + x*z^2 + y*z^2) +
          54*x*y*z + 126*(x*y + x*z + y*z) - 9*(x^2 + y^2 + z^2) - 9*(x + y + z) + 1

# TwentySevenLines writes a line as t -> (t, b[1]t + b[2], c[1]t + c[2]), so it
# only sees the lines that are graphs over x. in the standard chart 5 of the 27
# are not -- they are vertical or lie in the plane at infinity -- and the system
# has 22 solutions. homogenizing, moving the plane at infinity by a linear change
# of coordinates, and dehomogenizing again brings all 27 into view. it is the same
# surface, looked at through a different chart.
E, C = exponents_coefficients(clebsch, [x,y,z])
clebsch_h = sum(C[i] * w^(3 - sum(E[:,i])) * x^E[1,i] * y^E[2,i] * z^E[3,i] for i in axes(E,2))

A = [ 1.0  0.3 -0.2  0.1
      0.2  1.0  0.4 -0.3
     -0.1  0.5  1.0  0.2
      0.4 -0.2  0.3  1.0]
cubic = expand(subs(clebsch_h, [w,x,y,z] => A*[1,x,y,z]))

F = TwentySevenLines()

# the parameters of F are the coefficients of the cubic, in the order F lists
# them. `a` here is the same symbol TwentySevenLines uses, so the parameters can
# be matched by name instead of by position.
function cubic_parameters(cub)
    E, C = exponents_coefficients(cub, [x,y,z])
    coeff = Dict(a[E[1,i]+1, E[2,i]+1, E[3,i]+1] => Float64(C[i]) for i in axes(E,2))
    return [get(coeff, q, 0.0) for q in parameters(F)]
end

# plain `solve` is ambiguous once OrdinaryDiffEq is loaded, hence the qualification
res = HomotopyContinuation.solve(F; target_parameters = cubic_parameters(cubic))
lines = real_solutions(res)     # 27 of them, all real
length(lines) == 27

# line (b[1],b[2],c[1],c[2]) is t -> (t, b[1]t + b[2], c[1]t + c[2])
line_point(L) = [0.0, L[2], L[4]]
line_dir(L) = [1.0, L[1], L[3]] / sqrt(1 + L[1]^2 + L[3]^2)

# a plane n·X = d contains a line iff n·P = d and n·v = 0. stacking those two
# conditions for a set of lines gives a 2m x 4 system that drops rank exactly when
# the lines are coplanar, and the null vector is then (n, d).
function plane_of(Ls)
    B = zeros(2*length(Ls), 4)
    for (i, L) in enumerate(Ls)
        B[2i-1, 1:3] .= line_point(L); B[2i-1, 4] = -1.0
        B[2i,   1:3] .= line_dir(L)
        B[2i-1, :] ./= norm(B[2i-1, :])
    end
    S = svd(B)
    P = S.V[:,4]
    # scaling n to a unit vector makes the form the signed distance to the plane
    return P / norm(P[1:3]), S.S[4]     # (n, d), and how far the lines are from coplanar
end

# the 45 tritangent planes: each meets the cubic in three of the lines
triples = [(i,j,k) for i in 1:25 for j in i+1:26 for k in j+1:27
           if last(plane_of(lines[[i,j,k]])) < 1e-8]
length(triples) == 45

# nine of those planes, no two sharing a line, cover all 27 lines exactly once, so
# their product vanishes on the lines and on nothing else of the cubic: 3*9 = 27.
# that is the cheapest possible numerator -- any f cutting out the lines has
# degree at least 9 -- and degree matters a great deal here, since the routing
# system is built out of ∇f.
function cover(triples)
    used = falses(27)
    chosen = NTuple{3,Int}[]
    function search()
        i = findfirst(!, used)
        i === nothing && return true
        for T in triples
            (i in T && all(!used[t] for t in T)) || continue
            for t in T; used[t] = true end
            push!(chosen, T)
            search() && return true
            pop!(chosen)
            for t in T; used[t] = false end
        end
        return false
    end
    return search() ? chosen : nothing
end

planes = [first(plane_of(lines[collect(T)])) for T in cover(triples)]
f = expand(prod(P[1]*x + P[2]*y + P[3]*z - P[4] for P in planes))
degree(f) == 9

# deg f = 9 forces d = 5, and g^5 is of order 10^7 out where the surface is, so
# |r| on V(G) runs around 10^-6 -- under the 1e-5 at which routing_points decides
# a point lies on the zero locus of r and discards it. scaling f up fixes that and
# changes nothing else: a positive multiple of f has the same critical points with
# the same indices.
f = 1e6 * f

# and now the components of the cubic surface with its 27 lines removed
r = RoutingFunction(f, [x,y,z], [0.7978234324, 0.6623073432, 0.2347907832])
cache = RoutingCache(r, [cubic])

M, routPoints = find_connectivity_matrix(cache; nstarts = 200, box = 5.0,
                                         grad_step_size = 1e-1, tol = 2e-1,
                                         start_step_size = 5e-1, verbose = true)
components = connected_components(cache, routPoints, M)

# every routing point that turns up has index 0 and every component is a single
# point with χ = 1, which is the right shape for the answer: with the lines gone
# r goes to 0 along the boundary of every region and at infinity, so each region
# carries a maximum of |r| and is a disc. the reversed flow runs into the zero
# locus instead of into a critical point, which is why no index 2 points appear.
#
# the count is a lower bound, though. the routing system here is three equations
# of degree 15, monodromy stops well short of the full fibre (`heuristic_stop`,
# the same failure as Chubs above), so the answer is carried by the flow seeds,
# and a region is only seen if a random start in [-box, box]^3 happens to land in
# it. raising nstarts finds more, but slowly: 200 starts gives 60 components in
# about four minutes, 1000 starts gives 69 in about eleven.
#
# for comparison, in the projective picture the 27 lines meet in 115 points (10 of
# them Eckardt points, where three lines meet at once) and are cut into 240 arcs,
# so with χ(X) = -5 the arrangement has 240 - 115 - 5 = 120 faces. this affine
# chart cuts the surface along one more plane section, so the true count here is
# larger still.


# the same surface again, with Clebsch's own nonic as the numerator instead of a
# product of tritangent planes.
#
# in the P^4 model the Clebsch is {p1 = p3 = 0}; eliminating the fifth coordinate
# with x4 = -(x0+x1+x2+x3) puts it in P^3 as p3 = 0. Clebsch's covariant of order
# 9, whose intersection with the surface is exactly the 27 lines, is then
#
#     F9 = p5 * (3 p2^2 - 10 p4),      p_k = k-th power sum of the five coordinates
#
# with p2^2 and p4 of degree 4 and p5 of degree 5. nothing has to be solved for to
# write it down, which is the whole point of it: the construction above needs the
# 27 lines first, and then a choice of nine planes covering them.
@var v[1:4]
five = [v[1], v[2], v[3], v[4], -(v[1]+v[2]+v[3]+v[4])]
pk(k) = sum(c^k for c in five)

clebsch_cubic = pk(3)
clebsch_nonic = pk(5) * (3*pk(2)^2 - 10*pk(4))

# same chart move as before, so that all 27 lines stay in the affine picture
B = [ 1.0  0.3 -0.2  0.1
      0.2  1.0  0.4 -0.3
     -0.1  0.5  1.0  0.2
      0.4 -0.2  0.3  1.0]
cubic9 = expand(subs(clebsch_cubic, v => B*[1,x,y,z]))
nonic9 = expand(subs(clebsch_nonic, v => B*[1,x,y,z]))
degree(cubic9) == 3 && degree(nonic9) == 9

# this is the same surface as the one above: 27 real lines, 45 tritangent planes,
# and 10 Eckardt points, which pin it down as the Clebsch (it is the only cubic
# surface with 10 of them).
lines9 = real_solutions(HomotopyContinuation.solve(F; target_parameters = cubic_parameters(cubic9)))
length(lines9) == 27

triples9 = [(i,j,k) for i in 1:25 for j in i+1:26 for k in j+1:27
            if last(plane_of(lines9[[i,j,k]])) < 1e-8]
length(triples9) == 45

# three lines of a tritangent plane meet at a point at an Eckardt point
function concurrent(Ls)
    st = hcat(line_dir(Ls[1]), -line_dir(Ls[2])) \ (line_point(Ls[2]) - line_point(Ls[1]))
    P = line_point(Ls[1]) + st[1]*line_dir(Ls[1])
    norm(P - (line_point(Ls[2]) + st[2]*line_dir(Ls[2]))) > 1e-6 && return false
    Q, d = line_point(Ls[3]), line_dir(Ls[3])
    return norm((P - Q) - dot(P - Q, d)*d) < 1e-6
end
count(T -> concurrent(lines9[collect(T)]), triples9) == 10

# F9 and the product of nine tritangent planes are different degree 9 forms -- one
# splits into linear factors, the other does not -- but the nonics through the 27
# lines are unique modulo the cubic, so on the surface the two can only differ by a
# constant. building this model's nine planes and taking the ratio confirms it:
# `extrema(ratios)` agrees to four or five digits, which is all a degree 9
# evaluation in double precision is good for. the constant itself is of no
# interest and changes from run to run -- the cover search returns whichever nine
# planes it finds first, and each plane form carries an arbitrary sign.
prod9 = expand(prod(P[1]*x + P[2]*y + P[3]*z - P[4]
                    for P in [first(plane_of(lines9[collect(T)])) for T in cover(triples9)]))

# |r| = |F9|/g^5 has a median around 2e-3 on the surface, so scale it up past the
# 1e-5 that routing_points reads as "on the zero locus", exactly as above
r = RoutingFunction(1e3*nonic9, [x,y,z], [0.7978234324, 0.6623073432, 0.2347907832])
cache = RoutingCache(r, [cubic9])

ratios = Float64[]
for _ in 1:200
    P = 4*(2*rand(3) .- 1)
    project_to_variety!(P, cache)
    all(isfinite, P) || continue
    den = evaluate(nonic9, [x,y,z] => P)
    abs(den) > 1e-6 && push!(ratios, evaluate(prod9, [x,y,z] => P) / den)
end
extrema(ratios)

M, routPoints = find_connectivity_matrix(cache; nstarts = 200, box = 5.0,
                                         grad_step_size = 1e-1, tol = 2e-1,
                                         start_step_size = 5e-1, verbose = true)
components = connected_components(cache, routPoints, M)

# 86 components, again all of them a single index 0 routing point with χ = 1. the
# same run with the product of nine planes finds 60, so the nonic is the better
# numerator of the two: same degree, same zero locus on the surface, but its
# coefficients come out of power sums rather than out of nine planes one of which
# sits far from the origin, and the routing points are correspondingly easier to
# find. still a lower bound on the 120-odd regions, for the reasons above.
