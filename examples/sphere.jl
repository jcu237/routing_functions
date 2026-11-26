# The unit sphere with the two circles z = ±1/2 removed: three components (two caps
# and a band). Plots the sphere coloured by r, the removed circles, the routing points
# (by Morse index) and the gradient paths from each saddle up to the maxima of |r|.
#
# Plotting needs CairoMakie in a reachable environment (e.g. the global one), not in
# the package's Project.toml:  julia --project=. -t auto examples/sphere.jl
using ConnectedComponents
using LinearAlgebra
using CairoMakie

const SciMLBase = ConnectedComponents.SciMLBase

@var x y z
g = x^2 + y^2 + z^2 - 1
f = (z-1/2)*(z+1/2)

# a fixed (generic) centre, so the picture is reproducible
r = RoutingFunction(f, [x, y, z], [0.07, 0.35, 0.70])
cache = RoutingCache(r, [g])

pts = routing_points(cache)
indices = routing_point_indices(cache, pts)
println(connected_components(cache, pts))

# ---------------------------------------------------------------------------------
# gradient paths: the same flow `solve_ivp` runs, but keeping the whole trajectory
# ---------------------------------------------------------------------------------
function gradient_path(cache, P, targets; tol = 1e-2, dtmax = 1e-2, tspan = (0.0, 50.0))
    H, V = hessian_and_tangent(cache, P)
    paths = Vector{Vector{Float64}}[]
    for Q in find_starting_points_for_flow(cache, P, H, V)
        arrived(u, _, _) = distance_to_endpoints(u, targets) - tol
        cb = SciMLBase.ContinuousCallback(arrived, SciMLBase.terminate!)
        prob = SciMLBase.ODEProblem(cache.flow_unit!, copy(Q), tspan, 1.0)
        sol = SciMLBase.solve(prob; reltol = 1e-8, abstol = 1e-8, dtmax = dtmax, callback = cb)
        path = [[copy(P)]; sol.u]
        if sol.retcode == SciMLBase.ReturnCode.Terminated
            push!(path, targets[nearest_index(last(path), targets)])
        end
        push!(paths, path)
    end
    return paths
end

sgn = [sign(r(P)) for P in pts]
paths = Vector{Vector{Float64}}[]
for i in eachindex(pts)
    indices[i] == 0 && continue
    targets = [pts[j] for j in eachindex(pts) if indices[j] == 0 && sgn[j] == sgn[i]]
    append!(paths, gradient_path(cache, pts[i], targets))
end

# ---------------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------------
azimuth, elevation = -0.2, 0.35                     # view direction (radians)
eye = [cos(elevation) * cos(azimuth), cos(elevation) * sin(azimuth), sin(elevation)]
visible(p) = dot(p, eye) > 0                        # orthographic view of the unit sphere

# split a curve on the sphere into its front and back halves, NaN-separated, lifted
# slightly off the surface. CairoMakie does not depth-test lines against surfaces, so
# the back half is drawn before the translucent sphere and the front half after it.
function split_curve(curve; lift = 1.005)
    gap = Point3f(NaN, NaN, NaN)
    vis = visible.(curve)
    # a point next to one on the other side goes in both halves, so they meet
    near(k, side) = any(vis[j] == side for j in max(k - 1, 1):min(k + 1, length(curve)))
    front = [near(k, true) ? Point3f(lift .* curve[k]) : gap for k in eachindex(curve)]
    back = [near(k, false) ? Point3f(lift .* curve[k]) : gap for k in eachindex(curve)]
    return front, back
end

circle(h; n = 400) = [[sqrt(1 - h^2) * cos(t), sqrt(1 - h^2) * sin(t), h] for t in range(0, 2π; length = n)]

# sphere, coloured by r
nθ, nφ = 200, 400
θs = range(0, π; length = nθ)
φs = range(0, 2π; length = nφ)
X = [sin(θ) * cos(φ) for θ in θs, φ in φs]
Y = [sin(θ) * sin(φ) for θ in θs, φ in φs]
Z = [cos(θ) for θ in θs, φ in φs]
R = [r([X[i], Y[i], Z[i]]) for i in eachindex(X)]
R = reshape(R, size(X))
rmin, rmax = extrema(R)

ink = colorant"#191919"
# purple (r < 0) through neutral grey to orange (r > 0), with r = 0 at the grey: each arm
# spans its own sign's range, so the band and the caps are both saturated
z0 = -rmin / (rmax - rmin)                           # where r = 0 falls in the colour range
# the orange arm leaves the neutral grey quickly, so the lower cap (r ≈ 0.05) still reads
# as orange next to the band
diverging = cgrad([colorant"#3b1f73", colorant"#7b5bb5", colorant"#efefef", colorant"#f7b77a",
                   colorant"#e8741a", colorant"#a33d00"],
                  [0, z0 / 2, z0, z0 + 0.06 * (1 - z0), z0 + 0.45 * (1 - z0), 1])
markers = Dict(0 => :circle, 1 => :diamond, 2 => :rect)
labels = Dict(0 => "index 0 (max of |r|)", 1 => "index 1 (saddle)", 2 => "index 2")

faded = 0.45                                         # alpha of whatever is behind the sphere
lw_flow, lw_locus = 6, 10

# a flat arrowhead lying on the sphere at p, pointing along the tangent direction d
function arrowhead!(ax, p, d; size = 0.09, color = ink)
    side = normalize(cross(p, d))
    tri = [p + size .* d, p - 0.4size .* d + 0.45size .* side, p - 0.4size .* d - 0.45size .* side]
    mesh!(ax, [Point3f(1.012 .* normalize(v)) for v in tri], [1 2 3]; color = color, shading = NoShading)
end

set_theme!(fontsize = 24, font = "TeX Gyre Heros Makie")
fig = Figure(size = (1000, 930), backgroundcolor = :white, figure_padding = 10)
ax = Axis3(fig[1, 1]; aspect = :data, azimuth = azimuth, elevation = elevation,
           perspectiveness = 0.0, protrusions = 0, viewmode = :free)
ax.zoom_mult[] = 0.62                                # zoom in: crop the empty space around the sphere
hidedecorations!(ax)
hidespines!(ax)

# the sphere. CairoMakie does not depth-test lines against surfaces, so whatever is
# behind it is drawn on top, faded and dashed, and the front after that.
surface!(ax, X, Y, Z; color = R, colormap = diverging, colorrange = (rmin, rmax),
         shading = NoShading)

for front_side in (false, true)
    α = front_side ? 1.0 : faded
    style = front_side ? :solid : :dash
    for h in (-0.5, 0.5)
        front, back = split_curve(circle(h))
        # V(f) drawn as a cut: a heavy black seam on a white halo
        curve = front_side ? front : back
        front_side && lines!(ax, curve; color = :white, linewidth = lw_locus + 8)
        lines!(ax, curve; color = (ink, front_side ? 1.0 : 0.35), linewidth = front_side ? lw_locus : 5,
               linestyle = style)
    end
    for path in paths
        front, back = split_curve(path)
        lines!(ax, front_side ? front : back; color = (ink, α), linewidth = lw_flow, linestyle = style)
        # an arrowhead halfway along, pointing up the flow
        m = length(path) ÷ 2
        p = path[m]
        visible(p) == front_side &&
            arrowhead!(ax, p, normalize(path[min(m + 3, end)] - path[max(m - 3, 1)]); color = (ink, α))
    end
    for (P, i) in zip(pts, indices)
        visible(P) == front_side || continue
        scatter!(ax, [Point3f(1.012 .* P)]; marker = markers[i], markersize = 28,
                 color = (i == 0 ? ink : :white, α), strokecolor = (ink, α), strokewidth = 3)
    end
end

# legend
present = sort(unique(indices))
elements = Any[MarkerElement(marker = markers[i], color = i == 0 ? ink : :white,
                             strokecolor = ink, strokewidth = 3, markersize = 32) for i in present]
push!(elements, LineElement(color = ink, linewidth = lw_flow))
push!(elements, LineElement(color = ink, linewidth = lw_locus))
Legend(fig[2, 1:2], elements, [[labels[i] for i in present]; "gradient path"; "V(f): z = ±1/2"];
       orientation = :horizontal, framevisible = false, tellheight = true, nbanks = 2,
       labelsize = 34, patchsize = (40, 30), colgap = 30, padding = (0, 0, 0, 0))
Colorbar(fig[1, 2]; colormap = diverging, limits = (rmin, rmax), label = "r", labelsize = 30, ticklabelsize = 24,
         height = Relative(0.6), tellheight = false)

out = joinpath(@__DIR__, "sphere.png")
rowgap!(fig.layout, 0)
CairoMakie.save(out, fig; px_per_unit = 3)                    # 3000 × 2700 px
println("saved ", out)
