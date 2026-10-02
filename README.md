# ConnectedComponents.jl

**ConnectedComponents.jl** computes the connected components of a real algebraic
variety with a hypersurface removed,

```
V(G) ∖ V(f)  =  { x ∈ ℝⁿ : g₁(x) = … = g_k(x) = 0,  f(x) ≠ 0 },
```

together with the Euler characteristic of each component, using *routing functions*.

It implements the algorithm of
> **Smooth Connectivity in Real Algebraic Varieties**
> *Joseph Cummings, Jonathan Hauenstein, Hoon Hong, and Clifford Smyth*
> *Numerical Algorithms, 100(1), 63–84, 2025*

See also `HypersurfaceRegions.jl` and `ProjectedHypersurfaces.jl`, which use routing
functions for the complement of a hypersurface; many ideas from those packages are used
here. Claude was used to optimize the performance of this package. 

## Installation

```julia
using Pkg
Pkg.add(url = "(https://github.com/jcu237/routing_functions.git)")
```

The package re-exports [HomotopyContinuation.jl](https://www.juliahomotopycontinuation.org),
so `@var`, `differentiate`, `System`, … are available after `using ConnectedComponents`.
The first call in a session compiles a lot of code and takes a minute or so; later calls
are fast.

## Quick start

Two concentric circles:

```julia
using ConnectedComponents

@var x y
G = [(x^2 + y^2 - 1) * (x^2 + y^2 - 9)]     # the variety V(G)
r = RoutingFunction(1, [x, y])              # f = 1: remove nothing
C = connected_components(r, G)
```

```
2-element Vector{Component}:
 Component: 2 routing points of index 0, 1, χ = 0
 Component: 2 routing points of index 0, 1, χ = 0
```

Two components, each a circle (Euler characteristic 0).

## What goes in

**The variety.** `G` is a vector of polynomials (or a single polynomial, or a
HomotopyContinuation `System`). It must cut out `V(G)` with a jacobian of full rank:
reduced equations (no repeated factors), as many as the codimension.

**The removed hypersurface.** `RoutingFunction(f, vars)` builds the routing function
`r = f / gᵈ` with `g = ‖x − c‖² + 1`. The zero set of `f` is removed:

| you want | `f` |
|---|---|
| the components of `V(G)` itself | `1` |
| to cut along hypersurfaces `h₁ = 0, h₂ = 0, …` | `h₁ * h₂ * ⋯` |
| to separate by the signs of the coordinates | `x₁ * x₂ * ⋯ * xₙ` |
| to remove the singular points of `V(G)` (required, see below) | multiply by `singular_locus(G, vars)` |

Always pass `vars`, the ambient variables in the order your points are written in:
`f` may not involve all of them (a constant `f` involves none).

**Singular points must be removed.** The method needs `V(G)` to be smooth outside
`V(f)`. If `V(G)` has singular points, put them into `V(f)`:

```julia
@var x y z
g = x^2 + y^2 - z^2 + z^3                       # a cone point at the origin
r = RoutingFunction(singular_locus([g], [x, y, z]), [x, y, z])   # = ‖∇g‖²
connected_components(r, [g])                    # the bell (χ = 1) and the funnel (χ = 0)
```

**The centre `c`.** `RoutingFunction(f, vars, c)` fixes the centre of `g`; otherwise it is
random. It must be generic: a centre of symmetry of `V(G)` (the centre of a circle, say)
makes `r` degenerate and the answer meaningless. The package warns when it detects this.

## What comes out

A vector of `Component`s, one per connected component found:

```julia
for comp in C
    comp.points                 # routing points on this component (vectors in ℝⁿ)
    comp.indices                # the Morse index of each one
    comp.euler_characteristic   # ∑ (-1)^index
end
euler_characteristic(C)         # of all of V(G) ∖ V(f)
```

The **routing points** are the critical points of `r` on `V(G) ∖ V(f)`. Every component
contains at least one, a local maximum of `|r|` (index 0). The **index** of a routing
point is the number of directions in which `|r|` increases.

## Step by step

`connected_components(r, G)` runs four stages. To run them yourself, build a
`RoutingCache` once and pass it to each (the `(r, G)` forms build a new cache on
every call, which repeats all the symbolic work):

```julia
cache = RoutingCache(r, G)
pts   = routing_points(cache)                 # 1. critical points of r on V(G) ∖ V(f)
idxs  = routing_point_indices(cache, pts)     # 2. their Morse indices
A, pts = find_connectivity_matrix(cache, pts) # 3. join them by gradient flow
C     = connected_components(cache, pts, A)   # 4. group into components
```

## Options

The most useful keywords (all accepted by `connected_components`; the full table is in
[PIPELINE.md §6](PIPELINE.md#6-keywords)):

| keyword | default | when to change it |
|---|---|---|
| `box` | `3.0` | `V(G)` lies far from the origin: starting points are drawn from `[-box, box]ⁿ` |
| `nstarts` | `100` | more starting points find more small components |
| `starts` | — | your own starting points, e.g. points near `V(f)` to find small components |
| `monodromy_options` | `(;)` | `(timeout = 60,)` caps the slowest stage (see below) |
| `monodromy_family` | `:centre` | `:affine` if `V(G)` is reducible and the flows may miss one of its components |
| `start_step_size`, `tol` | `0.1`, `1e-2` | scale of the paths between routing points; raise both for large varieties |
| `locus_tol` | `1e-6` | points closer than this (relative to their size) to `V(f)` count as on it |
| `verbose` | `false` | progress messages |

## Reliability and speed

- **The result is only as complete as the search for routing points.** Routing points
  are found by gradient flows from random starting points, then by monodromy on a
  polynomial system (HomotopyContinuation's `monodromy_solve`), which stops heuristically.
  A component is reported only if a routing point on it was found, and its Euler
  characteristic is right only if all of them were. Small components are the easiest to
  miss; raise `nstarts` or supply `starts`.
- **Monodromy can be slow** on large systems (many variables, or high-degree `f`). Cap it
  with `monodromy_options = (timeout = 120,)`, or skip it: `flow_to_routing_points(cache)`
  returns the routing points the flows alone find, and
  `connected_components(cache, pts)` takes them from there.
- **Reducible varieties.** By default monodromy moves the centre of the routing
  function, which only reaches the irreducible components of `V(G)` that already carry
  a routing point (or a constructed start). If the flows may miss a whole component of
  `V(G)`, use `monodromy_family = :affine`: slower, but it reaches every component.
- **Checks:** every component must contain an index-0 routing point (the package warns
  otherwise); a connected curve has χ = 0 or 1.

## Examples

| file | what it shows | time |
|---|---|---|
| [examples/01_curves.jl](examples/01_curves.jl) | circles, an elliptic curve, the twisted cubic; the step-by-step API | seconds |
| [examples/02_singular_surfaces.jl](examples/02_singular_surfaces.jl) | removing singular points with `singular_locus` | a minute |
| [examples/kuramoto.jl](examples/kuramoto.jl) | a torus in ℝ⁶; what monodromy adds to the flows | a minute |
| [examples/3RPR.jl](examples/3RPR.jl) | a parallel manipulator; a variety thousands of units across (`box`, long paths) | a few minutes |
| [examples/clebsch_cubic.jl](examples/clebsch_cubic.jl) | the Clebsch cubic minus its 27 lines (141 regions); custom `starts` | two minutes |
| [examples/27lines.jl](examples/27lines.jl) | the same surface in another chart, two choices of `f` | five minutes |
| [examples/positive_landau.jl](examples/positive_landau.jl) | which components of a Landau variety have a positive point; `stop_when` | minutes |

## Troubleshooting

| message | meaning |
|---|---|
| `only m of n starts landed on V(G)` | `V(G)` is far from the origin (raise `box`), or `G` is not reduced / has too many equations, or `V(G)` has no real points |
| `no routing points were found` | as above, or `V(G) ∖ V(f)` is empty |
| `component(s) contain no index 0 routing point` | a path between routing points failed; try a smaller `start_step_size` or `tol` |
| `degenerate critical points` | `r` is not Morse: use a random centre `c` |
| `monodromy stopped at its timeout` | some routing points may be missing |
| `ArgumentError: G involves variables that r does not` | build `r` over all the variables: `RoutingFunction(f, vars)` |

**Name clashes.** If you also load a package that exports `connected_components` (e.g.
Graphs.jl) or `solve` (e.g. OrdinaryDiffEq.jl), qualify the call:
`ConnectedComponents.connected_components(...)`, `HomotopyContinuation.solve(...)`.

Comments and suggestions are welcome!
