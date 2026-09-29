# The Clebsch cubic surface with its 27 lines removed.
#
# The surface is cut out by `clebsch`; the numerator `f` is a nonic (computed in
# twentySeven.m2) whose intersection with the surface is exactly the 27 lines, so
# the components are the faces of the line arrangement. The count is derived below:
# 141, each a disc. See also clebsch_count.tex.
#
# Runtime: about two minutes, mostly the flows from ~1000 seeds.

using ConnectedComponents
using LinearAlgebra
using Random

Random.seed!(5)

@var x,y,z

clebsch = 81*(x^3+y^3+z^3) - 189*(x^2*(y+z) + y^2*(x+z) + z^2*(x+y)) + 54*x*y*z + 126*(x*y+y*z+x*z) - 9*(x^2+y^2+z^2) - 9*(x+y+z) + 1

f = 191231280*x^2*y^6*z+127720800*x*y^7*z-63510480*y^8*z+1035938160*x^2*y^5*z^2+628106400*x*y^6*z^2-153090000*y^7*z^2+2094026256*x^2*y^4*z^3+1240746336*x*y^5*z^3+365246496*y^6*z^3+2094026256*x^2*y^3*z^4+1425294144*x*y^4*z^4+1472043456*y^5*z^4+1035938160*x^2*y^2*z^5+1240746336*x*y^3*z^5+1472043456*y^4*z^5+191231280*x^2*y*z^6+628106400*x*y^2*z^6+365246496*y^3*z^6+127720800*x*y*z^7-153090000*y^2*z^7-63510480*y*z^8-276145200*x^2*y^5*z-269671680*x*y^6*z+91387440*y^7*z-1072353168*x^2*y^4*z^2-1146617856*x*y^5*z^2-3114288*y^6*z^2-1578244176*x^2*y^3*z^3-2131805952*x*y^4*z^3-949682880*y^5*z^3-1072353168*x^2*y^2*z^4-2131805952*x*y^3*z^4-1728838080*y^4*z^4-276145200*x^2*y*z^5-1146617856*x*y^2*z^5-949682880*y^3*z^5-269671680*x*y*z^6-3114288*y^2*z^6+91387440*y*z^7+113043600*x^2*y^4*z+195286464*x*y^5*z-29432160*y^6*z+328878144*x^2*y^3*z^2+737201736*x*y^4*z^2+149931000*y^5*z^2+328878144*x^2*y^2*z^3+1077252048*x*y^3*z^3+619663608*y^4*z^3+113043600*x^2*y*z^4+737201736*x*y^2*z^4+619663608*y^3*z^4+195286464*x*y*z^5+149931000*y^2*z^5-29432160*y*z^6-15895440*x^2*y^3*z-58278528*x*y^4*z-5212512*y^5*z-30585600*x^2*y^2*z^2-181630512*x*y^3*z^2-80366904*y^4*z^2-15895440*x^2*y*z^3-181630512*x*y^2*z^3-154076256*y^3*z^3-58278528*x*y*z^4-80366904*y^2*z^4-5212512*y*z^5+604125*x^2*y^2*z+7006122*x*y^3*z+3825549*y^4*z+604125*x^2*y*z^2+14260860*x*y^2*z^2+15283647*y^3*z^2+7006122*x*y*z^3+15283647*y^2*z^3+3825549*y*z^4+5985*x^2*y*z-305028*x*y^2*z-547461*y^3*z-305028*x*y*z^2-1145970*y^2*z^2-547461*y*z^3+1458*x*y*z+25419*y^2*z+25419*y*z^2-179*y*z

r = RoutingFunction(f, [x,y,z])

cache = RoutingCache(r, [clebsch])

# ---------------------------------------------------------------------------
# how many components should there be?
#
# V(f) ∩ V(clebsch) is the 27 lines, so this is the arrangement of the 27 lines on
# the real cubic surface. count its faces with Euler's formula, χ = V - E + F.
#
# X(ℝ) is ℝP² blown up at 6 real points, so χ(X(ℝ)) = 1 - 6 = -5.
#
# every line meets exactly 10 others, giving 27·10/2 = 135 intersecting pairs. on a
# general cubic surface those are 135 distinct points and each line carries 10 of
# them, so V = 135, E = 27·10 = 270 and F = -5 - 135 + 270 = 130. that is the
# number quoted for a general real cubic surface.
#
# the Clebsch is not general: it has 10 Eckardt points, where three lines meet at
# once. each one fuses three intersection points into one, so V = 135 - 20 = 115,
# and a line through e of them carries 10 - e marked points, so E = 270 - 30 = 240.
# that leaves F = -5 - 115 + 240 = 120 faces in ℝP³.
#
# this computation is affine, though, and the plane at infinity here is not
# tritangent -- X ∩ {w = 0} is a cubic curve C∞, not three of the 27 lines -- so
# C∞ cuts the surface further. the 27 lines meet {w = 0} in 21 distinct points, 3
# of which are Eckardt points already counted, and the 9 lines through those 3 gain
# nothing, so
#
#     V' = 115 + 21 - 3      = 133
#     E' = 240 + 18 + 21     = 279          (18 lines gain a mark, C∞ gains 21 arcs)
#     F' = -5 - 133 + 279    = 141
#
# so 141 components, and every one of them is a disc: |r| → 0 along the 27 lines
# and at infinity, so each face carries exactly one maximum of |r| and no saddles.
# every routing point therefore has index 0, the connectivity matrix is the
# identity, and the whole problem is to find all 141 critical points.
# ---------------------------------------------------------------------------

# the 27 lines. `TwentySevenLines` writes a line as t -> (t, b₁t + b₂, c₁t + c₂),
# so it only sees lines that are graphs over x; 5 of the 27 are not in this chart.
# homogenising, moving the plane at infinity and dehomogenising brings all 27 into
# view -- the same surface through a different chart -- and the lines are then
# mapped back to the original coordinates.
@var w
@var a[1:4,1:4,1:4]

function TwentySevenLines()
    @var xx, yy, zz
    terms = [[i,j,k] for i in 0:3 for j in 0:3 for k in 0:3 if i+j+k <= 3]
    f = sum([a[c[1]+1,c[2]+1,c[3]+1]*xx^c[1]*yy^c[2]*zz^c[3] for c in terms])
    Params = [a[c[1]+1,c[2]+1,c[3]+1] for c in terms]
    @var t, b[1:2], c[1:2]
    gg = subs(f, [xx,yy,zz] => [t, b[1]*t+b[2], c[1]*t+c[2]])
    System(coefficients(gg, [t]); variables = [b[1],b[2],c[1],c[2]], parameters = Params)
end

E3, C3 = exponents_coefficients(clebsch, [x,y,z])
clebsch_h = sum(C3[i] * w^(3 - sum(E3[:,i])) * x^E3[1,i] * y^E3[2,i] * z^E3[3,i]
                for i in axes(E3,2))
A = [ 1.0  0.3 -0.2  0.1
      0.2  1.0  0.4 -0.3
     -0.1  0.5  1.0  0.2
      0.4 -0.2  0.3  1.0]
cubicA = expand(subs(clebsch_h, [w,x,y,z] => A*[1,x,y,z]))

function cubic_parameters(cub)
    E, C = exponents_coefficients(cub, [x,y,z])
    coeff = Dict(a[E[1,i]+1, E[2,i]+1, E[3,i]+1] => Float64(C[i]) for i in axes(E,2))
    return [get(coeff, q, 0.0) for q in parameters(F27)]
end

F27 = TwentySevenLines()
L = real_solutions(HomotopyContinuation.solve(F27;
        target_parameters = cubic_parameters(cubicA), show_progress = false))
@assert length(L) == 27

# each line as two homogeneous points in the ORIGINAL coordinates
spans = [(A*[1.0, 0.0, l[2], l[4]], A*[0.0, 1.0, l[1], l[3]]) for l in L]

# ---------------------------------------------------------------------------
# seeds.
#
# a random point of the surface lands in a component in proportion to that
# component's size, so uniform sampling finds the large faces at once and the small
# ones -- the slivers around the Eckardt points and the line crossings -- almost
# never. it saturates around 136 of the 141 however large `nstarts` gets.
#
# every face touches one of the 27 lines, though, and more than that: every face is
# bounded by arcs of the arrangement. so cut each line at its marked points -- where
# it meets the other 26 lines, and where it crosses the plane at infinity -- and
# step off the middle of each arc, to both sides, inside the surface. a face
# adjacent to that arc is then hit whatever its size, which sampling t uniformly
# along the line does not guarantee: an arc shorter than the sample spacing is
# missed, and those are exactly the small faces.
∇c_sys = HomotopyContinuation.InterpretedSystem(
    System(differentiate(clebsch, [x,y,z]); variables = [x,y,z]))

# the parameters t at which the line S₁ + t S₂ meets the other lines, together with
# the one where it crosses {w = 0}
function marked_parameters(S, others)
    ts = Float64[]
    for T in others
        M = hcat(S[1], S[2], T[1], T[2])
        sv = svdvals(M)
        sv[4] / sv[1] < 1e-7 || continue                 # the two lines are skew
        N = nullspace(M; atol = 1e-7 * sv[1])
        size(N, 2) == 1 || continue
        # a S₁ + b S₂ + c T₁ + d T₂ = 0, so the meeting point is at t = b/a
        abs(N[1,1]) > 1e-12 || continue
        push!(ts, N[2,1] / N[1,1])
    end
    abs(S[2][1]) > 1e-12 && push!(ts, -S[1][1] / S[2][1])   # the point at infinity
    return sort!(ts)
end

function off_line_starts(spans, cache; step = 0.08, pad = 1.0, cap = 12.0)
    starts = Vector{Float64}[]
    grad = zeros(Float64, 3)
    for (i, S) in enumerate(spans)
        d = S[2][2:4]*S[1][1] - S[1][2:4]*S[2][1]   # constant affine direction
        norm(d) > 1e-12 || continue
        d = d / norm(d)

        ts = marked_parameters(S, spans[setdiff(1:length(spans), i)])
        isempty(ts) && continue
        # one sample per arc: the midpoints, plus one past each end for the arc that
        # closes the circle through t = ∞
        samples = vcat(first(ts) - pad, [(ts[k] + ts[k+1])/2 for k = 1:length(ts)-1],
                       last(ts) + pad)

        for t in samples
            u = S[1] + t*S[2]
            abs(u[1]) > 1e-8 || continue
            p = u[2:4] / u[1]
            maximum(abs, p) < cap || continue           # the arc runs off to infinity
            HomotopyContinuation.evaluate!(grad, ∇c_sys, p)
            norm(grad) > 1e-12 || continue
            # the line lies on X, so d is tangent to X and this crosses it inside X
            across = cross(grad/norm(grad), d)
            norm(across) > 1e-12 || continue
            across = across / norm(across)
            for σ in (1.0, -1.0)
                q = p + σ*step*across
                project_to_variety!(q, cache)
                (all(isfinite, q) && maximum(abs, q) < 20) || continue
                push!(starts, q)
            end
        end
    end
    return starts
end

seeds = off_line_starts(spans, cache)
append!(seeds, [6*(2*rand(3) .- 1) for _ in 1:400])   # plus the usual uniform ones
println(length(seeds), " seeds")

# about two minutes: one ODE integration per seed per time direction, then
# monodromy (a few seconds)
C = connected_components(cache; starts = seeds, grad_step_size = 1e-1, tol = 2e-1,
                         start_step_size = 5e-1, verbose = true)
println(length(C), " components (141 expected)")
