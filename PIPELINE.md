# ConnectedComponents.jl — how the pipeline computes

Reference for the whole computation, from `r` and `G` to a list of `Component`s.
Each stage names the function that does the work, the file it lives in, and what it
actually does. Written against the current source.

The mathematics is from *Smooth Connectivity in Real Algebraic Varieties* (Cummings,
Hauenstein, Hong, Smyth), with ideas borrowed from `HypersurfaceRegions.jl` and
`ProjectedHypersurfaces.jl`.

---

## 0. The object being computed

Given

- `X = V(G) ⊆ ℝⁿ`, a real variety cut out by `G = [g₁, …, g_k]`, and
- a **routing function** `r = f / gᵈ`,

the package returns the connected components of `X ∖ V(f)`. Note the `∖ V(f)`: the
zero locus of the numerator is *removed*, which is how singularities and other
unwanted loci get excluded — you put them inside `V(f)`.

The whole method rests on `r|X` being a Morse function that vanishes on the boundary
of every component and at infinity. Then each component contains at least one
critical point, gradient flow connects those critical points within a component and
never between components, and `χ = ∑(−1)^index` per component.

---

## 1. `RoutingFunction` — the Morse function

**[`src/routing_functions.jl`](src/routing_functions.jl)**

```julia
r = RoutingFunction(f, vars, c)      # or (f, vars), (f, c), (f)
```

The constructor builds

| quantity | definition | why |
|---|---|---|
| `g` | `∑ᵢ (xᵢ − cᵢ)² + 1` | strictly positive on `ℝⁿ`, so `r` has no poles and `sign(r) = sign(f)` |
| `d` | `degree(f) ÷ 2 + 1` | makes `deg gᵈ > deg f`, so `r → 0` at infinity |
| `grad` | `∇r`, symbolically | |
| `grad_num` | `∇f·g − d·f·∇g` | equals `g^(d+1)·∇r`, and is **polynomial** |

`c` is random unless given; a generic centre is what makes `r|X` Morse (distinct
critical values, non-degenerate Hessians).

`grad_num` is the key trick: `∇r` is a rational function, but clearing the
denominator gives a polynomial with the same zero locus, which is what
HomotopyContinuation can solve.

Four `InterpretedSystem`s are compiled once and stored: `r_sys`, `f_sys`,
`grad_sys`, `grad_num_sys`. This is purely for speed — `HC.evaluate(expr, vars => P)`
rebuilds an interpreter tape on every call.

Accessors: `evaluate_r`, `evaluate_grad_r`, `evaluate_g` (the last evaluates the sum
of squares directly, skipping the interpreter).

---

## 2. `RoutingCache` — everything derived from `(r, X)`

**[`src/cache.jl`](src/cache.jl)**

```julia
cache = RoutingCache(r, G; reg = 1e-8)
```

Building the cache is the expensive part: every symbolic `differentiate` and every
`InterpretedSystem` construction happens here, once. Afterwards the numerical
routines only touch preallocated buffers.

Contents:

- `G_sys` — `G` compiled. Its Jacobian is `JG`.
- `∇G_sys` — the gradients `[∇g₁; …; ∇g_k]` stacked row-wise, chosen so that the
  `i`-th `n × n` block of *its* Jacobian is exactly `∇²gᵢ`. That is how second
  derivatives are obtained without a second symbolic pass.
- `sys`, `sys_interp` — the routing system (§3).
- `param_sys` — the parametrised family monodromy runs over (§5).
- `flow!`, `flow_unit!` — the two gradient fields (§4).
- scratch: `G_val`, `JG_val`, `HG_val`, `grad_val`, `Hr_val`, `grad_num_val`, `M`,
  `wk`, `wn`, `wn2`, all sized from `n` and `k`.

Two linear-algebra helpers used throughout:

- **`normal_factor!(M, J, reg)`** — Cholesky-factors `J Jᵀ + reg·I` in place, lower
  triangle. The regularisation means a rank-deficient `J` (a singular point of `X`)
  *damps* the step instead of blowing it up, and guarantees positive-definiteness so
  the factorisation never fails. Hand-rolled because `k` is usually 1 or 2, where
  LAPACK's dispatch overhead dwarfs the arithmetic.
- **`normal_solve!(y, L)`** — in-place `L Lᵀ y = y`.

**`_quiet(f)`** runs `f` under a `NullLogger`. The ODE solvers warn on every path
that exhausts `maxiters`, and there can be thousands; the advice to raise `maxiters`
is measurably useless here, and every caller inspects `retcode`/`isfinite`/residuals
instead.

> **Caches are stateful.** The buffers are shared, so one cache must not be used
> from two threads at once. The flow fields are the exception: each closure owns
> private buffers, so an integration in progress cannot be clobbered by a call to,
> say, `hessian`.

---

## 3. The routing system

**`routing_system(r, G)` in [`src/routing_points.jl`](src/routing_points.jl)**

The Lagrange conditions for a critical point of `r|X`, cleared of denominators:

```
grad_num(x) − g(x)^(d+1) · JG(x)ᵀ λ  =  0        (n equations)
G(x)                                 =  0        (k equations)
```

in the `N = n + k` unknowns `(x, λ)`. Real solutions with `f(x) ≠ 0` are exactly the
**routing points** — the critical points of `r|X` away from the removed locus.

Degrees matter: for the Clebsch example this is `[15, 15, 15, 3]`, a Bézout bound of
10125, which is why the system is never solved directly.

---

## 4. The projected gradient field

**`_projected_gradient_field(r, G_sys, n, k, reg, unit_speed)` in
[`src/path_tracking.jl`](src/path_tracking.jl)**

Returns an in-place ODE right-hand side `flow!(du, u, dir, t)` computing

```
du  =  dir · sign(f) · (I − Jᵀ(JJᵀ + reg·I)⁻¹J) ∇r     ← tangential: ascend |r| within X
       −  Jᵀ(JJᵀ + reg·I)⁻¹ G                          ← normal: pull back onto X
```

- The **tangential** term is `∇r` projected onto `T_x X`, oriented by `sign(r)` —
  which equals `sign(f)` since `g > 0` — so the flow always ascends `|r|`.
- The **normal** term is a continuous Newton correction that holds the path on `X`
  rather than letting it drift off.
- `dir` is the ODE *parameter*, not a constant: `+1` flows to attracting critical
  points, `−1` to repelling ones. Neither direction alone finds both.
- `unit_speed` rescales the tangential part to norm 1, making `t` arc length. The
  path tracker needs this because it starts next to a critical point where `∇r ≈ 0`
  and an unscaled field would crawl; the routing-point search does not.

Both halves solve against the same `JJᵀ + reg·I`, so it is factored once per call.

**`_evaluate_field!`** guards the three system evaluations. A path running off to
infinity produces `NaN` — either because `u` itself went non-finite, or because `u`
is merely enormous and a high-degree numerator overflows mid-evaluation — and
`HC.evaluate!` cannot write a complex `NaN` into a `Float64` buffer, so it throws an
`InexactError` out of the solver. On failure the field returns a zero derivative and
lets the terminating callback end the path. Only `InexactError` is swallowed.

---

## 5. Stage 1 — finding the routing points

### 5a. `flow_to_routing_points` — real seeds by gradient flow

**[`src/routing_points.jl`](src/routing_points.jl)**

```julia
seeds = flow_to_routing_points(cache; nstarts, box, starts, tspan, maxiters,
                               f_tol, proj_tol, max_attempts, stop_when, verbose)
```

Per start:

1. **Choose a start.** Either a uniform draw from `[-box, box]ⁿ`, or — if `starts` is
   supplied — the next of the caller's own points, each tried exactly once (`nstarts`,
   `box` and `max_attempts` then go unused).
2. **Project onto `X`** with `project_to_variety_residual!`. A start that never
   reaches `X` is discarded rather than integrated: the field's normal term would
   have to drag it back first, which is most of the cost and none of the answer.
   With box sampling it resamples, so `nstarts` counts usable starts, not attempts.
3. **Integrate `flow!` in both time directions** (`dir = ±1`), `reltol = abstol =
   1e-8`, terminating on a `DiscreteCallback` that fires when `|f| < f_tol`. `r`
   changes sign across `V(f)`, so the field is discontinuous there and the integrator
   would otherwise thrash. The callback also fires on a non-finite state.
4. **Certify the endpoint.** Recover `λ` by least squares from
   `g^(d+1)·JGᵀ λ = grad_num`, then run `HC.newton` on the full routing system. Only
   Newton-certified solutions are kept.

`HC.unique_points` deduplicates. `stop_when` is an early exit: if a certified point
satisfies the caller's predicate, return it immediately and skip the rest — including
monodromy.

### 5b. `routing_points` — monodromy expansion

```julia
pts = routing_points(cache; all_vars, zero_tol, nstarts, box, starts,
                     proj_tol, max_attempts, stop_when, verbose)
```

1. Get flow seeds from 5a.
2. Build a random parameter point `p0` for `param_sys`, choosing its constant column
   so that a random `S0` solves the family there.
3. **Track each seed from `p_target = 0` to `p0`.** The seeds solve the *target*
   system, so each must be moved onto the generic fibre individually.
4. **`monodromy_solve`** on `param_sys` from all those start solutions.
5. **Track the whole fibre back** to `p_target = 0`.
6. Keep real solutions, plus the real flow seeds directly — they are Newton-certified
   already, so they survive even if the homotopy loses their path — then drop anything
   with `|r| ≤ zero_tol` as lying on the removed locus.

**Why the seeding matters so much.** `param_sys` subtracts *generic affine-linear
forms* from `sys` rather than constants, which enlarges the parameter space and brings
the monodromy group closer to transitive — but on hard systems it is still
intransitive, so monodromy only ever reaches the orbits its start solutions already
lie in. A critical point in an untouched orbit is unreachable no matter how long
monodromy runs. Real seeds in new orbits are the only lever, which is what `starts`
exists for.

---

## 6. Stage 2 — indices

**[`src/hessian.jl`](src/hessian.jl)**

The Hessian of `r` *restricted to* `X` is not the ambient Hessian; it needs the
second fundamental form.

- **`_compute_matrices(JG, HG, k)`** — computes `V`, an orthonormal basis of
  `T_x X` (as `nullspace(JG)`, which is SVD-based and therefore rank-revealing), and
  the `W` matrices of the paper. The `W`s solve one least-squares system whose rows
  say `∑ᵢ JG[j,i]·Wᵢ = −Vᵀ ∇²gⱼ V` and `∑ᵢ V[i,j]·Wᵢ = 0`.
- **`hessian_and_tangent(cache, point)`** — returns

  ```
  H = Vᵀ (∇²r) V + ∑ᵢ (∂r/∂xᵢ) Wᵢ
  ```

  together with `V`, in one pass. The ambient `∇²r` is free: it is the Jacobian of the
  already-compiled `grad_sys`.
- **`idx(r, H, P)`** — the Morse index, counted as the number of eigenvalues of
  `Symmetric(H)` whose sign matches `sign(r(P))`. Matching the sign of `r` is what
  makes this the index of `|r|` regardless of which side of `V(f)` the point is on.
- **`routing_point_indices`** preserves input order, so indices and the connectivity
  matrix can be read off together. **`sort_routing_points_by_index`** buckets them
  into a `Dict`.

---

## 7. Stage 3 — connecting the points

### 7a. `gradient_flow!` — one path

**[`src/path_tracking.jl`](src/path_tracking.jl)**

Integrates `flow_unit!` (arc length) from a point until it comes within `tol` of an
index-0 routing point. Arrival is detected by a **`ContinuousCallback`** on
`distance_to_endpoints(u) − tol`, which root-finds the crossing instead of
overshooting it. `dtmax` is capped at `tol` so a single step cannot jump clean over
the ball and leave the root-finder nothing to see. Returns whether it arrived
(`retcode == Terminated`); a stalled path is reported by the return value, not by the
solver.

### 7b. `find_starting_points_for_flow` — where to leave from

At a critical point `P` with Hessian `H` and tangent basis `V`, the **unstable**
directions are the eigenvectors whose eigenvalue matches `sign(r(P))`. Returns
`P ± step_size·v` for each, projected back onto `X`.

### 7c. `solve_ivp` — one positive-index point to its index-0 neighbours

Calls `hessian_and_tangent` once (getting `H` and `V` together), then
`find_starting_points_for_flow`, then `gradient_flow!` from each start. Paths that
never arrive are **dropped**, not reported at wherever they stalled.

### 7d. `find_connectivity_matrix` — the graph

**[`src/connectivity.jl`](src/connectivity.jl)**

```julia
M, routPoints = find_connectivity_matrix(cache; grad_step_size, start_step_size,
                                         tol, nstarts, box, starts, ...)
```

1. `routing_points` (§5), then `sort_routing_points_by_index` (§6).
2. `final_points = index_dict[0]`; `initial_points` = everything else.
3. Start from the identity matrix `A`. For each positive-index point, run `solve_ivp`
   and set `A[i,j] = A[j,i] = 1` for the index-0 point each path lands on, matched by
   `nearest_index`.
4. Return `boolean_power_sum(A)` — the reflexive-transitive closure `⋁ₖ Aᵏ` in Boolean
   arithmetic — together with the routing points in matching order.

The justification is the Mountain Pass Theorem: two index-0 critical points in the
same component are joined through an index-1 point, so flowing out of every
positive-index point along its unstable directions produces a graph whose connected
components are exactly the components of `X ∖ V(f)`.

### 7e. `connected_components` — grouping

**[`src/components.jl`](src/components.jl)**

`component_labels(A)` is a depth-first traversal labelling points by component in
order of first appearance; it reads only which entries are nonzero, so the raw
adjacency matrix and its closure give the same answer. `connected_components` then
groups points and indices per label and builds a `Component`, which stores the
points, their indices, and `χ = ∑(−1)^index`.

The one-call entry point is

```julia
components = connected_components(cache; kwargs...)
```

which is just `find_connectivity_matrix` followed by the grouping.

---

## 8. Call graph

```
connected_components(cache)                                  components.jl
└── find_connectivity_matrix(cache)                           connectivity.jl
    ├── routing_points(cache)                                 routing_points.jl
    │   ├── flow_to_routing_points(cache)
    │   │   ├── project_to_variety_residual! → _project_to_variety!    path_tracking.jl
    │   │   ├── SciMLBase.solve(flow!)        ← _projected_gradient_field
    │   │   └── HC.newton(sys_interp)
    │   ├── HC.solve(param_sys)               seeds → generic fibre
    │   ├── monodromy_solve(param_sys)
    │   └── HC.solve(param_sys)               fibre → target
    ├── sort_routing_points_by_index                          hessian.jl
    │   └── routing_point_indices → idx ∘ hessian
    │       └── hessian_and_tangent → compute_matrices + ambient_gradient_hessian!
    ├── solve_ivp(cache, P, final_points)                     path_tracking.jl
    │   ├── hessian_and_tangent
    │   ├── find_starting_points_for_flow → project_to_variety!
    │   └── gradient_flow! → SciMLBase.solve(flow_unit!)
    └── boolean_power_sum → boolean_matmul!                   connectivity.jl
└── connected_components(cache, pts, A) → component_labels, Component
```

---

## 9. Knobs, and what they actually control

| keyword | default | stage | effect |
|---|---|---|---|
| `c` (on `RoutingFunction`) | random | §1 | centre of `g`; genericity is what makes `r|X` Morse |
| `reg` | `1e-8` | §2 | damping of `JJᵀ`; larger tolerates worse conditioning |
| `nstarts` | `100` | §5a | number of usable box starts |
| `box` | `3.0` | §5a | sampling box. Too small misses the variety entirely |
| `starts` | `nothing` | §5a | your own seeds; overrides box sampling |
| `proj_tol` | `1e-8` | §5a | how well a start must land on `X` |
| `max_attempts` | `20` | §5a | resampling budget, as a multiple of `nstarts` |
| `f_tol` | `1e-8` | §5a | when a flow is deemed to have hit `V(f)` |
| `zero_tol` | `1e-5` | §5b | `|r|` below which a point counts as on the removed locus |
| `stop_when` | `nothing` | §5 | early exit on the first matching routing point |
| `tol` | `1e-2` | §7a | arrival radius around index-0 points |
| `grad_step_size` | `0.05` | §7a | passed as `dtmax`; `0` means use `tol` |
| `start_step_size` | `0.1` | §7b | how far to step off a critical point |

---

## 10. Failure modes worth knowing

- **The answer is a lower bound.** Every stage can lose things: a component is only
  reported if some routing point in it was found, and routing points come from flow
  seeds plus whatever monodromy reaches from them. `monodromy_solve` returning
  `heuristic_stop` means it gave up without exhausting the fibre.
- **Uniform seeding is area-weighted.** A random surface point lands in a component
  in proportion to its size, so small components are found rarely and `nstarts`
  saturates. `starts` is the fix; see [`examples/clebsch_cubic.jl`](examples/clebsch_cubic.jl), which
  seeds from the arrangement's arcs instead.
- **All indices zero ⇒ the connectivity stage does nothing.** `initial_points` is
  empty, `A` stays the identity, no path is ever tracked, and the component count is
  just the number of index-0 points found. This is correct when every component is a
  disc, but worth recognising.
- **`find_connectivity_matrix` assumes an index-0 point exists.** `index_dict[0]`
  raises a `KeyError` if the search found none.
- **Badly scaled `f` cuts both ways.** If `|r|` on `X` sits below `zero_tol`,
  genuine routing points are discarded as lying on the removed locus; scaling `f` by
  a positive constant fixes that and changes neither the critical points nor their
  indices. Conversely a huge `‖∇r‖` makes the seeding flow stiff.
- **Singular `X`.** The theory needs `X` smooth. The package's idiom is to put the
  singular locus inside `V(f)` — e.g. `f = ‖∇g‖²` — so it is removed along with
  everything else the numerator kills.
