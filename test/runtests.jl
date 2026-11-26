using ConnectedComponents
using LinearAlgebra
using Random
using Test

const CC = ConnectedComponents

# the Euler characteristics of the components, sorted, so results can be compared
# without caring about the order components come out in
eulers(C) = sort([c.euler_characteristic for c in C])

@testset "ConnectedComponents.jl" begin

    @testset "normal_factor! survives a rank-deficient jacobian with large entries" begin
        # rounding in J Jᵀ is about eps·‖J‖² ≈ 1, far above reg = 1e-8, so an
        # unclamped Cholesky pivot can come out negative (a DomainError from sqrt)
        Random.seed!(0)
        for _ = 1:200
            a = 1e8 .* randn(3)
            J = vcat(a', (a .* (1 + 1e-12))', randn(1, 3))
            M = zeros(3, 3)
            CC.normal_factor!(M, J, 1e-8)
            y = randn(3)
            CC.normal_solve!(y, M)
            @test all(isfinite, M) && all(isfinite, y)
        end

    end

    @testset "closed-form derivatives of r agree with symbolic ones" begin
        @var x[1:3]
        f = x[1]^3 * x[2] - 2x[2] * x[3]^2 + x[1] - 0.5
        r = RoutingFunction(f, x[1:3], [0.3, -0.2, 0.7])
        rsym = f / r.g^r.d
        ∇rsym = differentiate(rsym, x[1:3])
        Hrsym = differentiate(∇rsym, x[1:3])
        P = [0.4, -1.1, 0.9]
        ∇r, Hr = evaluate_grad_hessian_r(r, P)
        @test evaluate_r(r, P) ≈ evaluate(rsym, x[1:3] => P)
        @test ∇r ≈ evaluate(∇rsym, x[1:3] => P)
        @test evaluate_grad_r(r, P) ≈ ∇r
        @test Hr ≈ evaluate(Hrsym, x[1:3] => P)
    end

    @testset "hessian on V(G)" begin
        # valid away from critical points too: on the unit sphere the hessian of a
        # linear function a·x is -(a·P)·I, in any orthonormal tangent basis
        @var z[1:3]
        sphere = [z[1]^2 + z[2]^2 + z[3]^2 - 1]
        a = [0.3, -1.2, 0.8]
        lin = sum(a[i] * z[i] for i = 1:3)
        for P in ([0.0, 0.0, 1.0], [0.6, 0.0, 0.8], [1.0, 2.0, -2.0] ./ 3)
            @test hessian(lin, sphere, z[1:3], P) ≈ -(a' * P) * Matrix(I, 2, 2) atol = 1e-12
        end

        # example 2.5b of "Smooth Connectivity in Real Algebraic Varieties". hessian
        # works in the basis V = nullspace(JG), with JG = [2 -2 -1] at (1, 1, 1); B is
        # the paper's orthonormal tangent basis, and R = VᵀB changes between the two
        @var y[1:3]
        G = [y[1]^2 - y[2]^2 * y[3]]
        φ = 4 * y[1]^2 + 4 * y[2]^2 * y[3]^2 + y[2]^4
        H = hessian(φ, G, y[1:3], [1.0, 1.0, 1.0])
        V = nullspace([2.0 -2.0 -1.0])
        B = [1 1/3; 1 -1/3; 0 4/3] ./ sqrt(2)
        R = transpose(V) * B
        @test transpose(R) * H * R ≈ 2 / 81 * [567 303; 303 127]

        # at routing points, the hessian built from the closed-form derivatives of r
        # matches the one built from the symbolic expression of r
        Random.seed!(3)
        @var x[1:2]
        r = RoutingFunction(x[1] * x[2], [1 / 3, 1 / 2])
        G2 = [x[1]^4 + x[2]^4 - (x[1] - x[2])^2 * (x[1] + x[2])]
        cache = RoutingCache(r, G2)
        pts = routing_points(cache)
        @test !isempty(pts)
        rexpr = r.f / r.g^r.d
        for P in pts
            @test critical_distance(cache, P) < 1e-8
            @test hessian(cache, P) ≈ hessian(rexpr, G2, x[1:2], P) rtol = 1e-6
        end

        # more equations than the codimension: the twisted cubic as a complete
        # intersection and cut out by three quadrics has the same hessian (up to the
        # tangent basis, so compare eigenvalues), also a little way off the curve, where
        # the third singular value of JG is small but not zero
        @var w[1:3]
        φ = w[1] + 2w[2] - w[3] + w[1] * w[3]
        T2 = [w[2] - w[1]^2, w[3] - w[1]^3]
        T3 = [w[2] - w[1]^2, w[3] - w[1] * w[2], w[1] * w[3] - w[2]^2]
        for t in (0.7, -1.2), off in (0.0, 1e-10)
            P = [t, t^2, t^3] .+ off .* [1.0, -1.0, 0.5]
            e2 = eigvals(Symmetric(hessian(φ, T2, w[1:3], P; dim = 1)))
            e3 = eigvals(Symmetric(hessian(φ, T3, w[1:3], P; dim = 1)))
            @test e2 ≈ e3 rtol = 1e-6
            @test hessian(φ, T3, w[1:3], P) ≈ hessian(φ, T3, w[1:3], P; dim = 1)
        end
    end

    @testset "the removed locus is recognised independently of the scale of f" begin
        @var x[1:2]
        f = x[1] * x[2] - 0.3
        P = [0.9, 0.6]
        d1 = distance_to_zero_locus(RoutingFunction(f, x[1:2], [0.1, 0.2]), P)
        d2 = distance_to_zero_locus(RoutingFunction(1e6 * f, x[1:2], [0.1, 0.2]), P)
        @test d1 ≈ d2
        # the second-order bound: the root of |f| = ‖∇f‖δ + ½‖∇²f‖δ²
        a, b, c = abs(0.9 * 0.6 - 0.3), norm([0.6, 0.9]), norm([0 1; 1 0])
        @test d1 ≈ 2a / (b + sqrt(b^2 + 2c * a))
        @test on_zero_locus(RoutingFunction(f, x[1:2]), [1.0, 0.3])

        # where two components of V(f) cross, ∇f vanishes too and |f|/‖∇f‖ is a ratio
        # of rounding errors; the second-order bound still sees the crossing, also
        # when f is expanded
        fx = expand((x[1] - 0.3) * (x[2] - 0.1) * (x[1] + x[2] - 1.7))
        rx = RoutingFunction(fx, x[1:2], [0.1, 0.2])
        @test on_zero_locus(rx, [0.3, 0.1])
        @test on_zero_locus(rx, [0.3 + 1e-9, 0.1 - 1e-9])
        @test !on_zero_locus(rx, [0.6, 0.5])
        # ... and three components through one point, where ∇²f is noise as well,
        # are caught by the rounding floor
        f3 = expand((x[1] - 0.3) * (x[2] - 0.1) * (x[1] + x[2] - 0.4))
        @test on_zero_locus(RoutingFunction(f3, x[1:2], [0.1, 0.2]), [0.3, 0.1])
    end

    @testset "inputs in any reasonable form, and clear errors" begin
        @var x[1:3]
        circle = x[1]^2 + x[2]^2 - 1

        # constants, bare variables and expressions are all numerators
        @test RoutingFunction(1, x[1:2]).d == 1
        @test RoutingFunction(x[1], x[1:2], [0.1, 0.2]).f == Expression(x[1])
        @test_throws ArgumentError RoutingFunction(one(Expression))        # no variables
        @test_throws ArgumentError RoutingFunction(x[1], x[1:2], [0.1])    # length of c
        @test_throws ArgumentError RoutingFunction(x[3], x[1:2])           # stray variable
        @test_throws ArgumentError RoutingFunction(x[1], "x")              # unknown form

        # G as a vector, a single expression, or a System
        r = RoutingFunction(x[1] * x[2], x[1:2], [0.3, 0.1])
        for G in ([circle], circle, System([circle]; variables = x[1:2]))
            @test RoutingCache(r, G).k == 1
        end
        @test_throws ArgumentError RoutingCache(r, [circle + x[3]])        # G uses x₃, r does not
        @test_throws ArgumentError RoutingCache(r, Expression[])

        # user variables named like the internal multipliers and parameters
        @var λ[1:2] q[1:2] μ[1:2] c[1:2]
        for v in (λ, q, μ, c)
            rv = RoutingFunction(v[1] * v[2], v, [0.3, 0.1])
            @test length(variables(routing_system(rv, [v[1]^2 + v[2]^2 - 1]))) == 3
        end
        # c and q are also the names of the monodromy families' parameters
        Random.seed!(13)
        rc = RoutingFunction(c[1] * q[1], [c[1], q[1]], [0.3, 0.1])
        @test eulers(connected_components(rc, [c[1]^2 + q[1]^2 - 1])) == [1, 1, 1, 1]

        # starts as tuples or untyped vectors
        cache = RoutingCache(r, [circle])
        @test !isempty(flow_to_routing_points(cache; starts = [(0.5, 0.5), Any[-0.5, 0.5]]))
        @test_throws ArgumentError flow_to_routing_points(cache; starts = [[1.0, 0.0, 0.0]])

        # reusing routing points: indices, components, and the total Euler characteristic
        Random.seed!(14)
        pts = routing_points(cache)
        @test [morse_index(cache, P) for P in pts] == routing_point_indices(cache, pts)
        C = connected_components(cache, pts)
        @test eulers(C) == [1, 1, 1, 1]
        @test euler_characteristic(C) == 4

        # a timeout in whole seconds (HC wants a Float64)
        @test !isempty(routing_points(cache; monodromy_options = (timeout = 30,)))
    end

    @testset "monodromy families" begin
        @var x[1:2]
        Random.seed!(15)
        r = RoutingFunction(x[1] * x[2], [1 / 3, 1 / 2])
        cache = RoutingCache(r, [x[1]^4 + x[2]^4 - (x[1] - x[2])^2 * (x[1] + x[2])])
        n = cache.n

        # the centre family at (c, 1), on its chart, is the routing system: its first
        # block is μ₀ times the routing system's, then G, then the chart
        z = randn(ComplexF64, cache.N)
        w = CC._to_chart(z, n, cache.chart)
        @test CC._from_chart(w, n) ≈ z
        Fw = evaluate(cache.centre_sys, w, vcat(r.c, 1.0))
        Fz = evaluate(routing_system(cache), z)
        @test Fw[1:n] ≈ w[n+1] .* Fz[1:n]
        @test Fw[n+1:cache.N] ≈ Fz[n+1:cache.N]
        @test abs(Fw[end]) < 1e-12

        # a constructed start pair solves the family at its parameters
        x0 = CC._complex_point_on_variety(cache, zeros(2), 1.0)
        @test x0 !== nothing
        z0, p1 = CC._centre_start_pair(cache, x0, 2.0)
        @test norm(evaluate(cache.centre_sys, CC._to_chart(z0, n, cache.chart), p1)) < 1e-10
        @test norm(z0[1:n] - p1[1:n]) ≈ 2.0            # ‖x0 - c‖ = scale

        # both families find the four routing points; all_vars gives solutions (x, μ)
        # of the routing system
        pc = routing_points(cache; all_vars = true)
        for zz in pc
            @test norm(evaluate(routing_system(cache), zz)) < 1e-8 * (1 + norm(zz))
        end
        pa = routing_points(cache; monodromy_family = :affine)
        @test length(pc) == length(pa) == 4
        @test_throws ArgumentError routing_points(cache; monodromy_family = :linear)

        # no flow seeds at all (the only start is far off the curve): monodromy starts
        # from constructed points of the complex curve
        @test length(routing_points(cache; starts = [[100.0, 100.0]])) == 4
    end

    @testset "jacobian_rank_locus" begin
        @var x[1:3]
        g = x[1]^2 + x[2]^2 - x[3]^2
        s = jacobian_rank_locus([g], x[1:3])
        P = [0.3, -0.7, 1.1]
        @test evaluate(s, x[1:3] => P) ≈ sum(abs2, evaluate(differentiate(g, x[1:3]), x[1:3] => P))
        # two equations: the jacobian drops rank where the sphere is tangent to the plane
        G = [x[1]^2 + x[2]^2 + x[3]^2 - 1, x[3] - 1]
        s2 = jacobian_rank_locus(G, x[1:3])
        @test abs(evaluate(s2, x[1:3] => [0.0, 0.0, 1.0])) < 1e-12
        @test evaluate(s2, x[1:3] => [0.6, 0.0, 0.8]) > 0.1
        # more equations than the codimension: the c × c minors, c the codimension
        T = [x[2] - x[1]^2, x[3] - x[1] * x[2], x[1] * x[3] - x[2]^2]   # twisted cubic, c = 2
        sT = jacobian_rank_locus(T, x[1:3])
        for t in (-1.3, 0.0, 0.7)
            P = [t, t^2, t^3]
            JP = evaluate(differentiate(T, x[1:3]), x[1:3] => P)
            σ = svdvals(JP)
            # e₂(J Jᵀ) = σ₁²σ₂² + σ₁²σ₃² + σ₂²σ₃², and σ₃ = 0 on the curve
            @test evaluate(sT, x[1:3] => P) ≈ σ[1]^2 * σ[2]^2 + σ[1]^2 * σ[3]^2 + σ[2]^2 * σ[3]^2
            @test evaluate(sT, x[1:3] => P) > 0.1
        end
        # det(J Jᵀ) vanishes on all of it
        @test abs(evaluate(jacobian_rank_locus(T, x[1:3]; dim = 0), x[1:3] => [0.7, 0.49, 0.343])) < 1e-12
        # more equations than variables: the origin, c = 3
        s4 = jacobian_rank_locus([x[1], x[2], x[3], x[1] + x[2]], x[1:3])
        @test evaluate(s4, x[1:3] => [0.0, 0.0, 0.0]) > 0.5
        @test_throws ArgumentError jacobian_rank_locus([x[1], x[2]], x[1:3]; dim = 3)
    end

    @testset "codimension" begin
        @var x y z
        @test codimension([y - x^2, z - x * y, x * z - y^2], [x, y, z]) == 2
        @test codimension([x, y, x + y], [x, y, z]) == 2
        @test codimension([x, 2x], [x, y]) == 1
        # complex samples: the real points (one point) are all singular
        @test codimension([x^2 + y^2 + z^2 - 1, z - 1], [x, y, z]) == 2
        # rows of very different sizes
        @test codimension([1e-8 * (x^2 + y^2 + z^2 - 1), z - 0.6], [x, y, z]) == 2
        # Newton converges to the embedded point at the origin, where JG = 0: an error,
        # not a wrong answer
        @test_throws ArgumentError codimension([x^2, x * y], [x, y])
        # reproducible, and the global random stream is left alone
        Random.seed!(3); a = rand()
        Random.seed!(3); codimension([x, y, x + y], [x, y, z]); @test rand() == a

        r = RoutingFunction(1, [x, y, z])
        cache = RoutingCache(r, [y - x^2, z - x * y, x * z - y^2])
        @test cache.c == 2 && cache.k == 3 && length(cache.H) == 2 && size(cache.B) == (2, 3)
        @test cache.N == 5 && length(variables(routing_system(cache))) == 5
        # a complete intersection is left alone
        cache = RoutingCache(r, [y - x^2, z - x^3])
        @test cache.c == 2 && cache.H == cache.G && cache.B === nothing
        # the dimension can be given, and is checked
        @test RoutingCache(r, [x, y, x + y]; dim = 1).c == 2
        @test_throws ArgumentError RoutingCache(r, [x, y]; dim = 0)
        @test_throws ArgumentError RoutingCache(r, [x, y]; dim = 3)
        # a repeated factor makes the rank too small: warned about, not squared up silently
        @test_logs (:warn, r"does not look reduced") match_mode = :any RoutingCache(r, [(x^2 + y^2 - 1)^2, z])
    end

    @testset "rank drops of G on V(G)" begin
        @var x y z
        Random.seed!(11)
        drops(G, vars, P; kw...) = CC._rank_drops_at(RoutingCache(RoutingFunction(1, vars), G; kw...), P)

        # full rank, at any row scale, and however close to tangent the equations are
        @test !drops([x^2 + y^2 - 1], [x, y], [0.6, 0.8])
        @test !drops([1e-8 * (x^2 + y^2 + z^2 - 1), z - 0.6], [x, y, z], [0.8, 0.0, 0.6])
        h = 1 - 1e-3
        @test !drops([x^2 + y^2 + z^2 - 1, z - h], [x, y, z], [sqrt(1 - h^2), 0.0, h])

        # a repeated factor, tested a little way off the circle, where the projection stalls
        @test drops([(x^2 + y^2 - 1)^2], [x, y], (1 + 1e-6) .* [0.6, 0.8])
        # plane ∪ line: the estimated codimension is the line's, so the plane drops rank
        @test drops([x * z, y * z], [x, y, z], [1.0, 2.0, 0.0])
        @test !drops([x * z, y * z], [x, y, z], [0.0, 0.0, 1.0])
        # with dim = 2 the line has rank above c, which is not a drop: it is lost unreported
        @test !drops([x * z, y * z], [x, y, z], [0.0, 0.0, 1.0]; dim = 2)
        # more equations than the codimension: no drop, unless dim is too small
        t = 0.7
        @test !drops([x * z - y^2, y - x^2, z - x * y], [x, y, z], [t, t^2, t^3])
        @test drops([x * z - y^2, y - x^2, z - x * y], [x, y, z], [t, t^2, t^3]; dim = 0)
        @test !drops([x, y, x + y], [x, y], [0.0, 0.0])
        # real points of too low a dimension
        @test drops([x^2 + y^2], [x, y, z], [1e-6, 0.0, 0.5])

        # the flows report it where f does not vanish, and stay quiet where V(f) removes it
        rank_warnings(thunk) =
            count(l -> occursin("rank below", string(l.message)),
                  first(Test.collect_test_logs(thunk; min_level = Base.CoreLogging.Warn)))
        G = [x * z, y * z]
        r = RoutingFunction(x - 3, [x, y, z])
        @test_logs (:warn, r"rank below 2") match_mode = :any flow_to_routing_points(r, G; nstarts = 20)
        r = RoutingFunction(jacobian_rank_locus(G, [x, y, z]), [x, y, z])
        @test rank_warnings(() -> flow_to_routing_points(r, G; nstarts = 20)) == 0
        # an isolated real point is reported as one point, not as a piece
        Random.seed!(11)
        g = x^2 + y^2 - x^3
        starts = [[1e-3, 2e-3], [-2e-3, 1e-3], [1e-3, -1e-3]]
        r = RoutingFunction(x - 1, [x, y])
        @test_logs (:warn, r"around 1 distinct point") match_mode = :any flow_to_routing_points(
            r, [g]; starts = starts)
        # and counts as in V(f) although the projection stalls a little way off it
        r = RoutingFunction(x^2 + y^2, [x, y])
        @test rank_warnings(() -> flow_to_routing_points(r, [g]; starts = starts)) == 0
    end

    @testset "examples with known answers" begin
        @var x[1:3]

        @testset "two circles" begin
            Random.seed!(1)
            r = RoutingFunction(-1 * one(Expression), x[1:2], [0.62, 0.35])
            G = [(x[1]^2 + x[2]^2 - 1) * (x[1]^2 + x[2]^2 - 9)]
            C = connected_components(RoutingCache(r, G); grad_step_size = 1e-1, tol = 2e-1,
                                     start_step_size = 5e-1)
            @test eulers(C) == [0, 0]
        end

        @testset "elliptic curve" begin
            Random.seed!(2)
            r = RoutingFunction(one(Expression), x[1:2], [0.855, -1.632])
            G = [x[2]^2 - x[1] * (x[1] - 1) * (x[1] + 1)]
            C = connected_components(RoutingCache(r, G); grad_step_size = 1e-1, tol = 2e-1,
                                     start_step_size = 5e-1)
            @test eulers(C) == [0, 1]         # the unbounded branch and the oval
        end

        @testset "twisted cubic with the origin removed" begin
            Random.seed!(4)
            r = RoutingFunction(x[1] * x[2] * x[3], x[1:3], [0.21, 0.83, 0.47])
            G = [x[1]^3 - x[3], x[1]^2 - x[2]]
            C = connected_components(RoutingCache(r, G))
            @test eulers(C) == [1, 1]
        end

        # not complete intersections: more equations than the codimension
        @testset "twisted cubic cut out by three quadrics, origin removed" begin
            Random.seed!(4)
            r = RoutingFunction(x[1] * x[2] * x[3], x[1:3], [0.21, 0.83, 0.47])
            G = [x[2] - x[1]^2, x[3] - x[1] * x[2], x[1] * x[3] - x[2]^2]
            cache = RoutingCache(r, G)
            C = connected_components(cache)
            @test eulers(C) == [1, 1]
        end

        @testset "a circle cut out by three equations" begin
            Random.seed!(2)
            G = [x[1]^2 + x[2]^2 + x[3]^2 - 1, x[3], 2x[3] + x[1]^2 + x[2]^2 + x[3]^2 - 1]
            r = RoutingFunction(x[1], x[1:3], [0.31, 0.12, 0.27])
            cache = RoutingCache(r, G)
            pts = routing_points(cache)
            # all on the circle, none on the rest of V(H)
            @test all(P -> abs(P[3]) < 1e-8 && abs(P[1]^2 + P[2]^2 - 1) < 1e-8, pts)
            @test eulers(connected_components(cache, pts)) == [1, 1]
        end

        @testset "quartic curve with the axes removed" begin
            Random.seed!(5)
            r = RoutingFunction(x[1] * x[2], [1 / 3, 1 / 2])
            G = [x[1]^4 + x[2]^4 - (x[1] - x[2])^2 * (x[1] + x[2])]
            C = connected_components(RoutingCache(r, G))
            @test eulers(C) == [1, 1, 1, 1]
        end

        @testset "a tiny numerator loses nothing" begin
            # |r| is about 1e-9 everywhere here. deciding that points lie on V(f) by
            # |r| < 1e-5, as the package used to, discards every routing point
            Random.seed!(6)
            r = RoutingFunction(1e-8 * x[1] * x[2], x[1:2], [0.3, 0.1])
            G = [x[1]^2 + x[2]^2 - 1]
            C = connected_components(RoutingCache(r, G))
            @test eulers(C) == [1, 1, 1, 1]   # the four open quarter-circles
        end

        @testset "ding dong with its singular point removed" begin
            Random.seed!(7)
            g = x[1]^2 + x[2]^2 - x[3]^2 + x[3]^3
            f = sum(differentiate(g, x[1:3]) .^ 2)
            r = RoutingFunction(f, x[1:3], [0.7978234324, 0.6623073432, 0.2347907832])
            C = connected_components(RoutingCache(r, [g]))
            @test eulers(C) == [0, 1]         # the cylinder below, the disc above
        end

        @testset "a thin region is kept" begin
            # the circle minus x = 0.3 and x = 0.3 + w: two thin arcs of width w and two
            # large ones. the old criticality test compared the tangential gradient with
            # |∇r|, whose rounding floor grows like eps/w², and dropped the thin arcs'
            # maxima once w was below ~1e-5
            Random.seed!(9)
            w = 6e-6
            r = RoutingFunction((x[1] - 0.3) * (x[1] - 0.3 - w), x[1:2], [0.1, -0.2])
            C = connected_components(RoutingCache(r, [x[1]^2 + x[2]^2 - 1]))
            @test eulers(C) == [1, 1, 1, 1]
        end

        @testset "a tiny numerator still connects through saddles" begin
            # the elliptic curve minus the circle x² + y² = 9, which cuts the unbounded
            # branch into three arcs, with f scaled by 1e-10: the flows out of the
            # oval's saddle must still arrive (the unit-speed field used to stop
            # normalising below an absolute 1e-8)
            Random.seed!(3)
            r = RoutingFunction(1e-10 * (x[1]^2 + x[2]^2 - 9), x[1:2], [0.855, -1.632])
            G = [x[2]^2 - x[1] * (x[1] - 1) * (x[1] + 1)]
            C = connected_components(RoutingCache(r, G); grad_step_size = 1e-1, tol = 2e-1,
                                     start_step_size = 5e-1)
            @test eulers(C) == [0, 1, 1, 1]
        end

        @testset "a nearly degenerate saddle" begin
            # r on the x-axis is 1 + τx² + (1 - 3τ)x⁴ + …: a saddle of |r| at 0 whose
            # eigenvalue 2τ is too small for the second-order step test to resolve.
            # the step falls back to the smallest one that increases |r|
            Random.seed!(10)
            @var y z
            τ = 1e-8
            r = RoutingFunction(1 + (3 + τ) * x[1]^2 + 4 * x[1]^4, [x[1], y, z], [0.0, 0.0, 0.0])
            C = connected_components(RoutingCache(r, [Expression(y), Expression(z)]))
            @test eulers(C) == [1]            # two maxima joined through the saddle
        end

        @testset "a sphere cut by four planes, f expanded" begin
            # 4 circles meeting pairwise in 12 points cut the sphere into
            # 2 - 12 + 24 = 14 discs. with f expanded, the 12 crossings are solutions
            # of the routing system on V(f) where ∇f = 0; none may survive as a routing point
            Random.seed!(11)
            f = expand((x[1] - 0.3) * (x[2] - 0.1) * (x[3] + 0.2) * (x[1] + x[2] + x[3] - 0.4))
            r = RoutingFunction(f, x[1:3], [0.31, 0.72, 0.18])
            C = connected_components(RoutingCache(r, [x[1]^2 + x[2]^2 + x[3]^2 - 1]))
            @test eulers(C) == fill(1, 14)
        end

        @testset "an empty variety has no components" begin
            Random.seed!(12)
            r = RoutingFunction(one(Expression), x[1:2], [0.2, 0.3])
            C = connected_components(RoutingCache(r, [x[1]^2 + x[2]^2 + 1]); nstarts = 5)
            @test isempty(C)
        end

        @testset "Chubs with its 12 nodes removed" begin
            # g = 0 is singular at (±1/√2, ±1/√2, 0) and permutations; f = |∇g|²
            # removes those points and leaves eight pieces, each with χ = -1. the
            # coarse connectivity settings are the ones testing.jl used, with
            # which a fixed 0.5 start step used to jump through a node and report 9
            Random.seed!(8)
            g = x[1]^4 + x[2]^4 + x[3]^4 - (x[1]^2 + x[2]^2 + x[3]^2) + 1 / 2
            f = sum(differentiate(g, x[1:3]) .^ 2)
            r = RoutingFunction(f, x[1:3], [0.7978234324, 0.6623073432, 0.2347907832])
            C = connected_components(RoutingCache(r, [g]); grad_step_size = 1e-1, tol = 2e-1,
                                     start_step_size = 5e-1)
            @test eulers(C) == fill(-1, 8)
            @test sum(length, C) == 80        # 32 + 44 + 4 routing points

            # a start step of 1.0 used to pass the second-order test by coincidence
            # and jump through a node; the model now has to hold at two scales
            A, pts = find_connectivity_matrix(RoutingCache(r, [g]); grad_step_size = 1e-1,
                                              tol = 2e-1, start_step_size = 1.0)
            @test eulers(connected_components(RoutingCache(r, [g]), pts, A)) == fill(-1, 8)
        end
    end
end
