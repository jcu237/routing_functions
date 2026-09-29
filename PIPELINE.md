# ConnectedComponents.jl — how the pipeline computes

Reference for the whole computation, from `r` and `G` to a list of `Component`s: the
mathematics (§1–§3), then every source file in pipeline order with what each function
computes (§4), the call graph (§5), the keywords (§6), and what can go wrong (§7).

The mathematics is from *Smooth Connectivity in Real Algebraic Varieties* (Cummings,
Hauenstein, Hong, Smyth), with ideas borrowed from `HypersurfaceRegions.jl` and
`ProjectedHypersurfaces.jl`.

---

## 0. The object being computed

Given

- `X = V(G) ⊆ ℝⁿ`, a real variety cut out by `G = [g₁, …, g_k]`, and
- a **routing function** `r = f / gᵈ`,

the package returns the connected components of `U = X ∖ V(f)`, each with its Euler
characteristic.

Since `r|X` is a routing function, each component contains at least one
critical point, gradient flow connects those critical points within a component and
never between components.

**Standing assumption.** `X` is smooth of dimension `n − k` at every point of `U`:
`JG` has rank `k` there. Singular points of `X` must lie in `V(f)`; multiply them into
`f` with `singular_locus(G, vars)` (for a hypersurface `g` that is `‖∇g‖²`).

Notation, as in the code:

| symbol | meaning | code |
|---|---|---|
| `x ∈ ℝⁿ` | ambient coordinates | `vars`, `n` |
| `G = (g₁,…,g_k)` | equations of `X` | `G`, `k` |
| `JG` | `k × n` jacobian of `G` | `JG_val`, `jacobian_G!` |
| `f` | numerator; `V(f)` is removed | `r.f` |
| `c`, `g = ‖x − c‖² + 1` | centre, denominator base | `r.c`, `r.g` |
| `d = deg f ÷ 2 + 1` | denominator exponent | `r.d` |
| `V` | orthonormal basis of `T_P X` (`nullspace(JG)`) | `V` |
| `λ` | Lagrange multipliers, `JGᵀλ = ∇r` | `λ` |
| `μ` | scaled multipliers `μ = g^(d+1) λ`, `JGᵀμ = grad_num` | `μ` |

---

## 1. Routing functions and Morse theory

### 1.1 Properties of `r = f / gᵈ`

1. `g ≥ 1`, so `r` is smooth on `ℝⁿ`, `sign r = sign f`, and `r = 0` exactly on `V(f)`.
2. `2d > deg f`, so `r → 0` as `‖x‖ → ∞`.
3. On a component `C` of `U`, `|r| > 0` and `|r| → 0` on `∂C ⊆ V(f)` and at infinity, so
   the superlevel sets `{x ∈ C : |r| ≥ ε}` are compact and `|r|` has a maximum on `C`.
   **Every component contains a routing point** (a critical point of `r|X`), in fact a
   local maximum of `|r|`.
4. For a generic centre `c`, `r|X` is Morse: its critical points are nondegenerate.
   That is why `c` is random unless given. A symmetric `c` (the centre of a circle,
   say) can make it degenerate.

### 1.2 Index and Euler characteristic

Morse theory for `−|r|` on `C`: `C` has the homotopy type of a CW complex with one cell
of dimension `index(P)` for each critical point `P ∈ C`, where

> **index(P)** = number of eigenvalues of the hessian of `r|X` at `P` with the sign of
> `r(P)` = number of directions in which `|r|` increases.

Index 0 is a local maximum of `|r|`, index `dim X` a local minimum. Hence

```
χ(C) = Σ_{P ∈ C} (−1)^index(P)
```

which is only right if every routing point of `C` was found.

### 1.3 The routing system

`P ∈ X` is critical for `r|X` iff `∇r(P) = JGᵀλ` for some `λ ∈ ℝᵏ`. By the quotient rule

```
∇r = (g ∇f − d f ∇g) / g^(d+1) = grad_num / g^(d+1)
```

and clearing denominators (`μ = g^(d+1) λ`) gives the **routing system** in the
`N = n + k` unknowns `(x, μ)`:

```
F₁(x, μ) = grad_num(x) − JG(x)ᵀ μ  =  0        (n equations)
F₂(x)    = G(x)                    =  0        (k equations)
```

Its real solutions with `f ≠ 0` are exactly the routing points (`g ≥ 1` on `ℝⁿ`, so real
`(x, λ)` and `(x, μ)` correspond one to one). It also has real solutions on `V(f)`, which
are discarded (§3.4), and complex ones with `g = 0`. Writing it in `μ` rather than `λ`
keeps the degree of `F₁` at `max(deg f + 1, deg G)` instead of
`max(deg f + 1, 2d + deg G + 2)` (Clebsch: `[10, 10, 10, 3]` instead of `[15, 15, 15, 3]`).

### 1.4 The gradient flow on `X`

With `P_T = I − JGᵀ(JGJGᵀ)⁻¹JG` the projection onto the tangent space, the flow is

```
ẋ = s · sign(r) · P_T ∇r  −  JGᵀ(JGJGᵀ)⁻¹ G,        s = ±1
```

- Tangential part: on `X`, `d|r|/dt = s ‖P_T ∇r‖²`. With `s = +1` `|r|` increases, so a
  path never reaches `V(f)` (where `r = 0`), stays in its component, keeps the sign of
  `r`, and converges to a critical point — generically a local maximum of `|r|`.
- Normal part: `d G(x)/dt = −G(x)` (because `JG P_T = 0`), so drift off `X` decays like `e⁻ᵗ`.
- `JGJGᵀ` is replaced by `JGJGᵀ + reg·I` so that a rank-deficient `JG` damps the step.

The routing-point search rescales time by `1/|r(x₀)|` (`x₀` the start), so the speed
does not depend on the scale of `f`. The connectivity stage uses the unit-speed field
(tangential part normalised, time = arc length), because it starts next to a critical
point where `∇r ≈ 0`.

### 1.5 The hessian on `X`

For a geodesic `γ` of `X` through `P` with `γ'(0) = u`,
`(r∘γ)''(0) = uᵀ∇²r u + ∇r·γ''(0)`, and differentiating `G(γ(t)) = 0` twice gives
`JG γ''(0) = −(uᵀ∇²gⱼ u)ⱼ`. Only the normal part `JGᵀλ` of `∇r` sees `γ''`, so

```
H = Vᵀ ( ∇²r − Σⱼ λⱼ ∇²gⱼ ) V,        JGᵀλ = ∇r  (least squares)          (1)
```

the projected hessian of the Lagrangian `r − λᵀG`. (1) holds at every point of `X`.
The package uses (1).

*The paper's form.* `H = Vᵀ∇²r V + Σᵢ ∂ᵢr · Wᵢ`, with the `Wᵢ` solving a
`(k·d' + d'²) × (n·d')` linear system (`d' = dim X`). The `Wᵢ` encode the second
fundamental form, and `Σᵢ ∂ᵢr Wᵢ = −Σⱼ λⱼ Vᵀ∇²gⱼ V`, so this is the same matrix as (1).
`compute_matrices` still builds the `Wᵢ` (the tests reproduce the paper's Example 2.5b).

*At a critical point* (the only place the pipeline needs `H`) two more forms agree with (1):

- **(2)** With `c₀ = r(P)` and the polynomial `h = f − c₀ gᵈ = (r − c₀) gᵈ`:
  `∇h = gᵈ∇r` and `∇²h = gᵈ∇²r + ∇r ∇(gᵈ)ᵀ + ∇(gᵈ) ∇rᵀ` at `P`. The cross terms vanish on
  `T_P X` (because `∇r` is normal there), so `H = Vᵀ(∇²h − Σⱼ νⱼ∇²gⱼ)V / gᵈ` with `JGᵀν = ∇h`.
  No quotient rule: `r` is replaced by the polynomial `f − r(P) gᵈ`.
- **(3)** For `F₁` of the routing system, `grad_num = g^(d+1)∇r` gives
  `∂ₓF₁ = g^(d+1)(∇²r − Σⱼ λⱼ∇²gⱼ) + ∇r ∇(g^(d+1))ᵀ` at a solution (`μ = g^(d+1)λ`), and
  `Vᵀ(·)V` kills the last term (`Vᵀ∇r = 0`), so `H = Vᵀ ∂ₓF₁(P, μ) V / g^(d+1)`: the
  tangential block of the jacobian of the routing system, which HomotopyContinuation has
  compiled already, at the `μ` it returns.

(2) and (3) perform the same cancellations as (1) and are no more accurate; (1) needs no
case split, so the code uses it and keeps (3) as `_hessian_from_routing_system`, a
cross-check in the tests.

The ambient derivatives come in closed form from `f, ∇f, ∇²f` (∇g = 2(x − c), ∇²g = 2I):

```
∇r  = (∇f − (d f / g) ∇g) / gᵈ
∇²r = ∇²f / gᵈ − d (∇f ∇gᵀ + ∇g ∇fᵀ + f ∇²g) / g^(d+1) + d(d+1) f ∇g ∇gᵀ / g^(d+2)
```

---

## 2. Connecting routing points

Inside one component `C`:

- **A routing point of positive index flows up to maxima.** Leave `P` along an unstable
  eigenvector `v` of `H` (eigenvalue with the sign of `r(P)`), in either sense, and follow
  the ascending flow: it ends at a critical point with larger `|r|`, generically a local
  maximum (index 0).
- **Two maxima of `C` are joined through index-1 points** (mountain pass theorem; the
  Palais–Smale condition holds because the superlevel sets of `|r|` are compact).

So the graph with the routing points as vertices and an edge `P — Q` whenever a flow
leaving `P` arrives at the maximum `Q` has the components of `U` as its connected
components, provided every routing point was found and every flow arrived. Two checks:

- an ascending flow keeps the sign of `r`, so it can only arrive at a maximum of that
  sign (`solve_ivp` offers no others as destinations);
- every component contains a maximum, so a graph component without an index-0 point
  means a flow failed (`connected_components` warns).

**Step size off `P`.** `P ± εv` is projected back onto `X`; to second order `|r|`
increases by `½|μ|ε²` (`μ` the eigenvalue). Too large an `ε` jumps across `V(f)`
(routing points can sit within `1e-4` of it) or through an isolated removed point (the
nodes of the Chubs surface). `find_starting_points_for_flow` starts at `start_step_size`
and halves `ε` until the observed increase is within a factor 2 of `½|μ|ε²` at both `ε`
and `ε/2`, with `r` keeping its sign. Below the scale where `½|μ|ε²` is lost in the
rounding of `|r|` it takes the smallest step that kept the sign and increased `|r|`.

---

## 3. Finding the routing points

### 3.1 Seeds by gradient flow (`flow_to_routing_points`)

Per start:

1. **Choose a start**: uniform in `[−box, box]ⁿ`, or the next of the caller's `starts`
   (each tried once; `nstarts`, `box`, `max_attempts` then go unused).
2. **Project onto `X`** (`project_to_variety_residual!`). A start with `‖G‖ ≥ proj_tol`
   afterwards is discarded; with box sampling another is drawn, so `nstarts` counts
   starts on `X`.
3. **Integrate the flow in both directions** (`s = ±1`), time rescaled by `1/|r(x₀)|`,
   `reltol = abstol = 1e-8`, stopped by a callback when the path comes within
   `locus_tol·(1 + ‖x‖)` of `V(f)` (Newton distance `|f|/‖∇f‖`) or goes non-finite.
4. **Refine the end point**: `μ` by least squares from `JGᵀμ = grad_num`, then
   `HC.newton` on the routing system. Kept if it is a routing point (§3.4).

Uniform sampling finds a component in proportion to its size, so small components are
found rarely; `starts` placed near `V(f)` is the remedy (`examples/clebsch_cubic.jl`).
`stop_when` returns the first routing point satisfying a predicate.

### 3.2 Monodromy (`routing_points`)

Monodromy collects the solutions of the routing system by moving it around loops in the
parameter space of a family containing it: along a loop the solutions are permuted, and
known solutions turn into new ones. Two families (`monodromy_family`):

**The centre family** (default). The routing systems of `f / (‖x − c‖² + a)ᵈ` for all
`(c, a) ∈ ℂⁿ⁺¹`:

```
F₁(x, μ; c, a) = (‖x − c‖² + a) ∇f − 2d f (x − c) − JGᵀμ,     F₂ = G.
```

The routing system of `r` is the member at `p₀ = (r.c, 1)`, and monodromy runs there
directly: no tracking to or from a generic member.

*Transitivity.* `x ∈ X ∖ V(f)` is critical for `(c, a)` iff
`(uᵀu + a)∇f − 2d f u = JGᵀμ` with `u = x − c`. Given `x`, every solution is
`u = (s∇f − JGᵀμ)/(2d f)`, `a = s − uᵀu` for a unique `(s, μ) ∈ ℂ^(k+1)` (`s = g(x)`). So the
incidence variety `{(x, μ, c, a)}` over `X ∖ V(f)` is `(X ∖ V(f)) × ℂ^(k+1)`: irreducible when
`X` is, of dimension `n + 1`, the number of parameters. Monodromy from any one routing
point therefore reaches every complex critical point of `r` on `X ∖ V(f)`, and nothing
else (the solutions on `V(f)` lie on other components). When `X` is reducible, a start
solution on each irreducible component is needed.

*Projective multipliers.* `μ` grows like `‖x‖^(deg f + 1)`: on 3RPR it is `10⁹` times `x`,
and HomotopyContinuation rejects such solutions as singular (jacobian condition number
`~10²⁰`). The family is solved in `(x, μ₀, μ̂)` with `μ = μ̂/μ₀` on a random chart:

```
μ₀ grad_num_{c,a}(x) − JGᵀμ̂ = 0,     G = 0,     ℓ₀μ₀ + ℓ·μ̂ = 1,
```

which keeps the multipliers of every solution of size about 1 (`_to_chart`, `_from_chart`).

1. Start solutions: the seeds of §3.1, and six constructed ones: a random complex point
   `x₀` of `X` (Gauss–Newton from random complex points at 1, 2, …, 32 times the extent of
   the points of `X` seen), random `(s, μ₀)` giving `(c₁, a₁)` as above, and the solution
   `(x₀, μ₀)` at `(c₁, a₁)` tracked to `p₀`. They make monodromy possible when the flows
   find nothing, and can reach components of a reducible `X` without seeds (two circles,
   seeds on the inner one only: the outer one is found in 25 of 30 runs).
2. `monodromy_solve` at `p₀`, in three phases of growing loop size (below), each ending
   after 10 loops without a new solution.
3. The real solutions, plus the seeds, are the candidates.

**Loop sampling.** HomotopyContinuation draws loop nodes from a standard normal
distribution: loops of size 1 around the origin of parameter space, which never reach
the parts of a large variety far from the origin. `_centre_sampler` draws `c` around the
centre of the points of `X` seen so far (the starts that landed, and the seeds) at scale
`σ = s·t`, `s` their spread and `t` log-uniform in `[1, T]`, and `a` at scale `σ²`. The phases
use `T = 1, 10, 100` in turn, each starting from everything found so far. Large loops find
solutions far out (the 141st Clebsch region needs them), but they rarely find new ones,
and mixed in from the start they end the search early (one of 14 regions missed in 3 of
40 runs on a sphere; 0 of 40 with phases). The sampler only knows the part of `X` the
flows reached: on a variety far larger than the sampling box, raise `box` (3RPR).

**The affine family** (`monodromy_family = :affine`). With `z = (x, μ)`,

```
F(z) − (Q z + q),        parameters Q ∈ ℂ^(N×N), q ∈ ℂ^N;  (Q, q) = 0 is F itself.
```

A random `z₀`, `Q₀` and `q₀ = F(z₀) − Q₀z₀` give a start pair at `p₁ = (Q₀, q₀)`; the seeds
are tracked from `0` to `p₁`; `monodromy_solve` runs at `p₁`; the fibre is tracked back to
`0`. The incidence variety `{(z, Q, q) : q = F(z) − Qz}` is a graph over `(z, Q)`, hence
irreducible even when `X` is not, but its fibre is much larger: it contains every
isolated solution of `F(z) = Qz + q` for generic `(Q, q)`, most of them unrelated to `r`.
Its loop nodes are drawn at the size of the base point, `|p₁|/√m` per coordinate.

`monodromy_options` overrides the defaults of either family (`parameter_sampler`, which
then replaces the phases, `max_loops_no_progress`, `timeout` for all phases together, ...).

The monodromy stage on the examples (solutions found, time, real routing points found):

| example | before (affine, λ, unit loops) | centre family, phased loops |
|---|---|---|
| Chubs | 930, 5 s, 80 | 172, 1.3 s, 80 |
| Kuramoto | 1966, timeout at 300 s, 12 | 66, 1.4 s, 12 |
| Clebsch (example's seeds) | 1458, 203 s, 140 | 162, 12 s, **141** |
| 27 lines, planes | 1548, 302 s, 125 | 171, 22 s, **145** |
| 3RPR (`box = 2000`) | — | 40–55, 1–13 s, 9–10 of 10 |

By parameter continuation every isolated solution of the routing system is reached from
a complete fibre. Completeness is not certified.

### 3.3 The index of each point

`routing_point_indices` computes `H` by (1) and counts eigenvalues with the sign of
`r(P)`. It warns when an eigenvalue is below `1e-10·|r(P)|/(1 + ‖P‖)²` (degenerate:
`r` is not Morse there).

### 3.4 Which solutions count (`_is_routing_point`)

A candidate `x` is kept iff it is finite, real (`|Im xᵢ| < 1e-8`), off `V(f)`, and critical.

- **Off `V(f)`**: its distance to `V(f)` exceeds `locus_tol·(1 + ‖x‖)`, with the
  distance bounded below by the Taylor model of `f` (`|f|` can drop by at most
  `‖∇f‖δ + ½‖∇²f‖δ²` within `δ`):

  ```
  δ(x) = 2|f| / (‖∇f‖ + sqrt(‖∇f‖² + 2‖∇²f‖·|f|))
  ```

  and `|f(x)|` above the rounding floor `1e3·eps·Σₐ|cₐ||x|ᵃ` (catches points where even
  `∇²f` vanishes, e.g. three sheets of `V(f)` through a point). Both are independent of
  the scale of `f`; a test on `|r|` is not, and `|r|` is tiny in thin or far-away regions.
- **Critical**: the Newton step to the nearest critical point, `‖H⁻¹Vᵀ∇r‖`, is below
  `crit_tol·(1 + ‖x‖)` (or, where `H` is singular, the tangential part of `∇r` is below
  `crit_tol·‖∇r‖`). A relative gradient test alone has a rounding floor of about
  `eps·(scale/width)²` and fails in regions thinner than about `1e-5`.

---

## 4. The source, file by file

Files are included in pipeline order; each uses only what the ones before it define.
Functions starting with `_` are internal.

### `src/utils.jl`
- `_quiet(f)` — runs `f` with logging off (the ODE solvers warn on every path that hits
  `maxiters`; callers check `retcode`/`isfinite` instead).
- `_ignored_keyword`, `_verbose` — warnings for the old keywords `f_tol`, `zero_tol`, `Verbose`.
- `IMAG_TOL = 1e-8` — a coordinate is real when its imaginary part is below this.
- `_as_expressions(G)` — accepts a vector, a single expression, or a `System`.
- `_as_points(P)` — user points (vectors, tuples) as `Vector{Vector{Float64}}`.

### `src/routing_functions.jl`
- `RoutingFunction(f, vars, c)` (also `(f, vars)`, `(f, c)`, `(f)`) — stores `f`, `g`, `c`,
  `d`, `vars`, `grad_num = g∇f − d f ∇g = g^(d+1)∇r`, and three compiled systems:
  `f_sys` (`f`), `∇f_sys` (`∇f`; its jacobian is `∇²f`), `f_abs_sys` (`Σₐ|cₐ|xᵃ`, for the
  rounding floor). `f` may be a constant or a variable; `vars` must contain its variables.
- `evaluate_f`, `evaluate_g`, `evaluate_r` (also `r(P)`), `evaluate_grad_r`,
  `evaluate_grad_hessian_r` — the closed forms of §1.5.
- `_grad_r!` — allocation-free `∇r` from `f(x)`, `∇f(x)`, for the flow fields.
- `distance_to_zero_locus(r, P)` — `δ(P)` of §3.4. `on_zero_locus(r, P; locus_tol)` — the
  §3.4 test. `_newton_distance` — `|f|/‖∇f‖`, for the flow callback.

### `src/variety.jl`
- `normal_factor!(M, J, reg)` — Cholesky factor of `JJᵀ + reg·I` in `M`'s lower triangle,
  pivots clamped at `reg` (rounding can make them negative when `J` is rank deficient with
  large entries). LAPACK's version of this method is designed for larger matrices and so
  the overhead is not worth it, thus the bespoke implementation.
- `normal_solve!(y, L)` — in-place `L Lᵀ y = y`.
- `_project_to_variety!` — Gauss–Newton steps `−Jᵀ(JJᵀ + reg·I)⁻¹G`, each backtracked until
  `‖G‖` decreases; stops when `‖G‖ < tol` or it stops decreasing; returns the point and `‖G‖`.
- `project_to_variety!(P, G, vars)` — the same without a cache.
- `singular_locus(G, vars)` — `Σ (maximal minors of JG)²`; vanishes where `rank JG < k`.

### `src/gradient_field.jl`
- `_projected_gradient_field(r, G_sys, n, k, reg, unit_speed)` — returns the ODE
  right-hand side `flow!(du, u, p, t)` of §1.4. `p` is `s` times a time scale. One
  factorisation of `JJᵀ + reg·I` per call serves both parts. The closure owns its buffers.
- `_evaluate_field!` — evaluates `G, JG, f, ∇f, ∇r`; returns `false` instead of throwing
  when the state overflows (a path running off to infinity makes `HC.evaluate!` throw an
  `InexactError`); the field then returns zero and the solver ends the path.

### `src/routing_system.jl`
- `routing_system(r, G)` — the system of §1.3, in `(x, μ)`. `μ` gets fresh names
  (`@unique_var`), so a user variable called `μ` cannot collide.
- `_centre_family(r, G, sys_vars)` — the centre family of §3.2, with parameters
  `(c₁, …, cₙ, a)` (fresh names).

### `src/cache.jl`
- `RoutingCache(r, G; reg = 1e-8)` — everything precomputed from `(r, G)`: `G_sys`
  (`G`; its jacobian is `JG`), `∇G_sys` (the gradients stacked, so the `i`-th `n×n` block
  of its jacobian is `∇²gᵢ`), `sys`/`sys_interp` (routing system), `centre_sys` and
  `param_sys` (the centre and affine families of §3.2), `flow!` and `flow_unit!` (the two fields),
  and scratch buffers. Checks that `G` uses only `r`'s variables. Stateful: one per thread.
- `evaluate_G!`, `jacobian_G!` — into the cache's buffers (overwritten by the next call).
- `project_to_variety!(P, cache)`, `project_to_variety(P, cache)`,
  `project_to_variety_residual!(P, cache)` — §4 `variety.jl`, with the cache's `reg`.
- `projected_gradient_field(cache; unit_speed)` — the stored field and `f_sys`.

### `src/hessian.jl`
- `_projected_hessian(JG, HG, ∇φ, ∇²φ)` — formula (1) for any function `φ`; returns `(H, V)`.
- `hessian_and_tangent(cache, P)` — `(H, V)` for `r`. `hessian(cache, P)`, `hessian(r, G, P)`,
  `hessian(φ, G, vars, P)` — `H` alone.
- `_hessian_from_routing_system(cache, P)` — form (3); tests only.
- `_criticality(cache, x)` — `(‖H⁻¹Vᵀ∇r‖, relative tangential gradient)`;
  `critical_distance(cache, P)` exports the first. `_is_critical` — the §3.4 test.
- `ambient_gradient_hessian` — `∇`, `∇²` in `ℝⁿ` of `r` or of any expression.
- `compute_matrices(cache, P)`, `compute_matrices(G, vars, P)` — the paper's `Wᵢ` and `V`.
- `idx(r, H, P)`, `morse_index(cache, P)` — the index of §1.2.
- `routing_point_indices(cache, pts)` — indices in input order, with the degeneracy warning.
- `sort_routing_points_by_index(cache, pts)` — `Dict(index => points)`.

### `src/routing_points.jl`
- `_flow_seeds(cache; …)` — §3.1; returns solutions `(x, μ)` as complex vectors, and
  records where the starts landed on `X`.
- `flow_to_routing_points(cache; all_vars = false, …)` — the same as real points.
- `_flow_and_refine` — one flow and its Newton refinement; `_grad_num` evaluates
  `grad_num` without powers of `g`. `_near_zero_locus_callback` — stops a flow near `V(f)`.
- `routing_points(cache; …)` — §3.2 and the filter §3.4; returns points of `ℝⁿ`
  (`all_vars = true`: with `μ`).
- `_monodromy_centre`, `_monodromy_affine` — the two families of §3.2. Helpers:
  `_extent` (centre and spread of the points of `X` seen), `_centre_sampler`,
  `_affine_sampler` (loop nodes), `_complex_point_on_variety`, `_centre_start_pair`
  (constructed start solutions).
- `_is_routing_point`, `_satisfies` — the §3.4 test; `stop_when` on the `x` part.
- `_unique_by_x`, `_real_x` — duplicates and realness judged on `x` alone (`μ` can be
  orders of magnitude larger than `x`).
- `_monodromy_options` — passes `monodromy_options` on, with `timeout` as a Float64.

### `src/mountain_pass.jl`
- `find_starting_points_for_flow(cache, P, H, V; step_size, max_halvings)` — the points
  `P ± εv` of §2.
- `gradient_flow!(cache, P, targets; tol, dtmax, tspan, maxiters, origin)` — unit-speed
  ascending flow until within `tol` of a target (a `ContinuousCallback` on
  `distance_to_endpoints(u) − tol`, so arrival inside a step is not missed). Returns
  whether it arrived. A start already within `tol` counts as arrived, unless the critical
  point it leaves (`origin`) is nearer; then the balls are shrunk.
- `solve_ivp(cache, P, final_points; …)` — `H` and `V` at `P`, the start points, and a flow
  from each to the index-0 points with the sign of `r(P)`. Returns where the arriving
  paths ended; the others are dropped.
- `distance_to_endpoints`, `nearest_index` — allocation-free distance helpers.

### `src/connectivity.jl`
- `component_labels(A)` — depth-first labelling of the graph with adjacency `A`.
- `reachability(A)` — `labels .== labelsᵀ` (same components as the Boolean power sum
  `⋁ₖ Aᵏ`, in `O(n²)`).
- `find_connectivity_matrix(cache[, pts]; …)` — routing points (unless given), indices, a
  `solve_ivp` from each point of positive index, an edge to the nearest target for each
  arriving path; returns `(reachability(A), pts)`. Errors if no routing point has index 0.

### `src/components.jl`
- `Component` — `points`, `indices`, `euler_characteristic`.
- `euler_characteristic(indices | component | components)`.
- `connected_components(cache; …)` — the whole pipeline. Also `(r, G; …)`,
  `(cache, pts; …)` (skip the search), `(cache, pts, A)` and `(pts, indices, A)` (skip
  the paths). Warns about components without an index-0 point.

---

## 5. Call graph

```
connected_components(cache)                                   components.jl
└── find_connectivity_matrix(cache)                           connectivity.jl
    ├── routing_points(cache)                                 routing_points.jl
    │   ├── _flow_seeds(cache)
    │   │   ├── project_to_variety_residual! → _project_to_variety!    variety.jl
    │   │   └── _flow_and_refine
    │   │       ├── SciMLBase.solve(flow!)    ← _projected_gradient_field   gradient_field.jl
    │   │       └── HC.newton(sys_interp)
    │   ├── _monodromy_centre                 (default; _monodromy_affine otherwise)
    │   │   ├── _complex_point_on_variety, _centre_start_pair, HC.solve(centre_sys)
    │   │   └── monodromy_solve(centre_sys; parameter_sampler = _centre_sampler)
    │   └── _is_routing_point → on_zero_locus, _is_critical → _criticality
    └── find_connectivity_matrix(cache, pts)
        ├── routing_point_indices → hessian → hessian_and_tangent  hessian.jl
        ├── solve_ivp(cache, P, targets)                      mountain_pass.jl
        │   ├── hessian_and_tangent
        │   ├── find_starting_points_for_flow → project_to_variety!
        │   └── gradient_flow! → SciMLBase.solve(flow_unit!)
        └── reachability → component_labels                   connectivity.jl
└── connected_components(cache, pts, A) → component_labels, Component
```

---

## 6. Keywords

| keyword | default | where | effect |
|---|---|---|---|
| `c` (in `RoutingFunction`) | random in `[0,1]ⁿ` | §1.1 | centre of `g`; must be generic |
| `reg` (in `RoutingCache`) | `1e-8` | §1.4 | damping of `JJᵀ` |
| `nstarts` | `100` | §3.1 | starts that must land on `X` |
| `box` | `3.0` | §3.1 | sampling box; too small misses a far-away `X` |
| `max_attempts` | `20` | §3.1 | sampling budget, × `nstarts` |
| `starts` | `nothing` | §3.1 | your own starting points |
| `proj_tol` | `1e-8` | §3.1 | `‖G‖` a start must reach |
| `flow_options` | `(;)` | §3.1 | `tspan = (0.0, 200.0)`, `maxiters = 10_000` of each flow |
| `stop_when` | `nothing` | §3.1 | return the first routing point satisfying it |
| `locus_tol` | `1e-6` | §3.4 | distance to `V(f)`, relative to `1 + ‖P‖` |
| `crit_tol` | `1e-6` | §3.4 | Newton step to a critical point, relative to `1 + ‖P‖` |
| `monodromy_family` | `:centre` | §3.2 | `:affine` if `X` is reducible and a component may lack seeds |
| `monodromy_options` | `(;)` | §3.2 | to `monodromy_solve`, e.g. `(timeout = 60,)` |
| `start_step_size` | `0.1` | §2 | first step off a critical point |
| `max_halvings` | `30` | §2 | halvings of that step |
| `tol` | `1e-2` | §2 | arrival radius around index-0 points |
| `grad_step_size` | `0.05` | §2 | largest ODE step of a path (`0` means `tol`) |
| `path_options` | `(;)` | §2 | `tspan = (0.0, 1e3)`, `maxiters = 10^6` of each path |
| `verbose` | `false` | all | progress messages |

`f_tol`, `zero_tol` (replaced by `locus_tol`) and `Verbose` (by `verbose`) are still
accepted, with a warning.

---

## 7. Failure modes

- **The answer is a lower bound on what is found.** A component is only reported if a
  routing point in it was found, and its `χ` is only right if all of them were.
  Monodromy stops heuristically; `returncode == :heuristic_stop` does not mean the fibre
  is complete.
- **Uniform seeding is area-weighted.** Small components are found rarely; use `starts`.
- **Reducible `X` with the centre family.** Monodromy only reaches the irreducible
  components of `X` that carry a start solution (a seed, or one of the two constructed
  starts). If the flows reach no real point of some component, its routing points can be
  missed; `monodromy_family = :affine` does not have this limitation.
- **A missed index-1 point** leaves two maxima of one component unjoined: overcount.
- **A failed path** leaves a point of positive index alone: overcount, with a warning.
  On a large variety the usual cause is the arc-length limit (`tspan = (0.0, 1e3)`): 3RPR
  has saddles 2000 units from its maxima and needs `path_options = (tspan = (0.0, 1e5),)`.
- **No index-0 point found** (but some routing points): `find_connectivity_matrix` errors,
  since every component has a maximum.
- **Singular `X` off `V(f)`**: the theory does not apply; put the singular locus into `f`.
- **Degenerate critical points** (non-generic `c`): indices and paths unreliable, with a
  warning.
- **`G` not reduced, or more equations than the codimension**: `JG` is rank deficient
  everywhere, the multipliers are undetermined, and nothing downstream is meaningful.

Consistency checks: every component contains an index-0 point; `χ` is plausible for
`dim X` (a connected curve has `χ ∈ {0, 1}`; a connected surface `χ ≤ 2`, and `χ ≤ 1` if
it is not compact); adjacent regions across a simple zero of `f` have opposite signs of
`r`, so a component containing both signs is a wrong merge.

---

## 8. A worked example: two concentric circles

`G = (x² + y² − 1)(x² + y² − 9)`, `f = 1` (so `d = 1`, `r = 1/g`). On each circle `|r|`
is largest at the point nearest `c` and smallest at the farthest: 4 routing points,
indices `0, 1, 0, 1`. From each index-1 point `solve_ivp` leaves in both senses along the
circle and arrives at that circle's maximum, never the other's (the circles are disjoint).
The graph has two components, each `{max, min}` with `χ = 1 − 1 = 0`:

```julia
@var x y
C = connected_components(RoutingFunction(1, [x, y]), [(x^2 + y^2 - 1) * (x^2 + y^2 - 9)])
# 2-element Vector{Component}:
#  Component: 2 routing points of index 0, 1, χ = 0
#  Component: 2 routing points of index 0, 1, χ = 0
```
