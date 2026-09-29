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
        # the projected-lagrangian formula equals the W-matrix formula of the
        # paper at any point, critical or not
        @var y[1:3]
        G = [y[1]^2 - y[2]^2 * y[3]]
        φ = 4 * y[1]^2 + 4 * y[2]^2 * y[3]^2 + y[2]^4
        for P in ([1.0, 1.0, 1.0], [2.0, 1.0, 4.0], [0.5, -1.0, 0.25])
            W, V = compute_matrices(G, y[1:3], P)
            ∇φ, Hφ = ambient_gradient_hessian(φ, y[1:3], P)
            Hw = transpose(V) * Hφ * V + sum(W[i] .* ∇φ[i] for i = 1:3)
            @test hessian(φ, G, y[1:3], P) ≈ Hw atol = 1e-10
        end

        # example 2.5b of "Smooth Connectivity in Real Algebraic Varieties", after
        # changing to the paper's basis of the tangent space
        W, V = compute_matrices(G, y[1:3], [1.0, 1.0, 1.0])
        H = hessian(φ, G, y[1:3], [1.0, 1.0, 1.0])
        B = hcat(1 / sqrt(2) * [1 1/3; 1 -1/3; 0 4/3], [2 / 3; -2 / 3; -1 / 3])
        R = (transpose(hcat(V, [2 / 3; -2 / 3; -1 / 3])) * B)[1:2, 1:2]
        @test transpose(R) * H * R ≈ 2 / 81 * [567 303; 303 127]
        @test transpose(R) * W[1] * R ≈ [0 4/27; 4/27 -16/81] atol = 1e-12

        # at a routing point the hessian also equals the tangential block of the
        # jacobian of the routing system, divided by gᵈ⁺¹ (form (3) in hessian.jl),
        # and the symbolic hessian of r itself
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
            H = hessian(cache, P)
            @test H ≈ first(CC._hessian_from_routing_system(cache, P)) rtol = 1e-8
            @test H ≈ hessian(rexpr, G2, x[1:2], P) rtol = 1e-6
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
            sys = routing_system(rv, [v[1]^2 + v[2]^2 - 1])
            @test length(variables(sys)) == 3
            Random.seed!(13)
            @test eulers(connected_components(rv, [v[1]^2 + v[2]^2 - 1])) == [1, 1, 1, 1]
        end

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

        # both families find the four routing points
        pc = routing_points(cache)
        pa = routing_points(cache; monodromy_family = :affine)
        @test length(pc) == length(pa) == 4
        @test_throws ArgumentError routing_points(cache; monodromy_family = :linear)

        # no flow seeds at all (the only start is far off the curve): monodromy starts
        # from constructed points of the complex curve
        @test length(routing_points(cache; starts = [[100.0, 100.0]])) == 4

        # all_vars gives solutions of the routing system in (x, μ)
        for zz in routing_points(cache; all_vars = true)
            @test norm(evaluate(routing_system(cache), zz)) < 1e-8 * (1 + norm(zz))
        end
    end

    @testset "singular_locus" begin
        @var x[1:3]
        g = x[1]^2 + x[2]^2 - x[3]^2
        s = singular_locus([g], x[1:3])
        P = [0.3, -0.7, 1.1]
        @test evaluate(s, x[1:3] => P) ≈ sum(abs2, evaluate(differentiate(g, x[1:3]), x[1:3] => P))
        # two equations: the jacobian drops rank where the sphere is tangent to the plane
        G = [x[1]^2 + x[2]^2 + x[3]^2 - 1, x[3] - 1]
        s2 = singular_locus(G, x[1:3])
        @test abs(evaluate(s2, x[1:3] => [0.0, 0.0, 1.0])) < 1e-12
        @test evaluate(s2, x[1:3] => [0.6, 0.0, 0.8]) > 0.1
        @test_throws ArgumentError singular_locus([x[1], x[2], x[3], x[1] + x[2]], x[1:3])
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

        @testset "elliptic curve with two points removed" begin
            Random.seed!(3)
            r = RoutingFunction(x[1]^2 + x[2]^2 - 9, x[1:2], [0.855, -1.632])
            G = [x[2]^2 - x[1] * (x[1] - 1) * (x[1] + 1)]
            C = connected_components(RoutingCache(r, G); grad_step_size = 1e-1, tol = 2e-1,
                                     start_step_size = 5e-1)
            # the circle x² + y² = 9 meets only the unbounded branch, cutting it into
            # three arcs; the oval, a circle, is untouched
            @test eulers(C) == [0, 1, 1, 1]
        end

        @testset "twisted cubic with the origin removed" begin
            Random.seed!(4)
            r = RoutingFunction(x[1] * x[2] * x[3], x[1:3], [0.21, 0.83, 0.47])
            G = [x[1]^3 - x[3], x[1]^2 - x[2]]
            C = connected_components(RoutingCache(r, G))
            @test eulers(C) == [1, 1]
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
            # the elliptic curve minus two points again, with f scaled by 1e-10: the
            # flows out of the oval's saddle must still arrive (the unit-speed field
            # used to stop normalising below an absolute 1e-8)
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
