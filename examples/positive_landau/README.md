# Positive Landau discriminants, diagram by diagram

The computation of [`../positive_landau.jl`](../positive_landau.jl) and
[`../positive_landau.m2`](../positive_landau.m2) for every Feynman diagram of Fig. 1 of
Mizera and Telen, *Landau Discriminants* (arXiv:2109.08036): which irreducible components
of the incidence variety `Y` of the Landau equations contain a point with `α > 0` and a
positive mass ratio `M/m`? Equal masses throughout, in the charts `α_E = 1` and `m = 1`.

## Running a diagram

```
M2 --script examples/positive_landau/acn.m2                # components and singular locus of Y
julia -t auto --project=. examples/positive_landau/acn.jl  # routing points, and the positive ones
```

The Macaulay2 script comes first. It writes `results/acn.txt`, which the Julia script
reads to build the routing numerator `f`; the Julia script prints its report and writes
it to `results/acn_positive.txt`.

## Running all of them

```
DRY_RUN=1 examples/positive_landau/run_all.sh                    # what it would do
nohup examples/positive_landau/run_all.sh > run_all.out 2>&1 &   # do it, e.g. overnight
```

`run_all.sh` goes through the diagrams in the order of `diagrams.jl`, running both
scripts of each, or through the ones named on its command line in the order given. A
diagram that already has `results/<name>_positive.txt` is skipped, so an interrupted run
can be started again. Only one `run_all.sh` runs at a time: a second one refuses to
start. Progress is in `logs/summary.txt`, the output of each step in
`logs/<name>.m2.log` and `logs/<name>.jl.log`.

Every step has a limit, set by a variable: `M2_LIMIT=2h` and `M2_MEMORY_GB=48` for a
Macaulay2 script, `MONODROMY_SECONDS=3600` for monodromy on each stratum (the routing
points found by then are still reported), `JULIA_LIMIT=3h` for a Julia script; `0` lifts
one. If Macaulay2 leaves no results, the Julia script runs with `--no-singular-locus`
(see the last section), and the summary says so.

## The equations

`equations/` is already there. It only has to be rebuilt after `diagrams.jl` changes:

```
julia --project=. examples/positive_landau/generate_equations.jl
```

## Files

| file | what it holds |
|---|---|
| `diagrams.jl` | the diagrams (edges, legs) and `LandauSystem`, which builds the equations with Landau.jl |
| `generate_equations.jl` | writes `equations/<name>.m2`, the same systems in Macaulay2 syntax |
| `singular_locus.m2` | the Macaulay2 functions: components of `Y`, where the jacobian drops rank, the factors of `f` |
| `positive.jl` | the Julia functions: reading the Macaulay2 results, `f`, routing points, the report |
| `check_diagrams.jl` | compares the discriminant degree of every diagram with Table 1 of the paper |
| `<name>.m2`, `<name>.jl` | the two scripts of each diagram |

The diagrams: `par`, `acn`, `env`, `npltrb`, `tdetri`, `debox`, `tdebox`, `pltrb`, `dbox`,
`pentb` as in the paper, and for the two families of Fig. 1 the triangle `A3`, the box
`A4`, and the bananas `B2` (the bubble), `B3` (the sunrise) and `B4`. Another member of a
family is one more line in `diagrams.jl`, a copy of the two scripts, and `generate_equations.jl`.

## How it differs from the parachute example

- **Landau.jl is not loaded as a package.** It needs Arpack and GenericSVD, which are not
  installed here. `diagrams.jl` evaluates the five functions that build the equations
  straight from its source file, unchanged.
- **Macaulay2 works without `y`**, and takes the jacobian modulo one component of `Y` at a
  time. With `y` in the ring the decomposition of `npltrb` alone took three minutes;
  without it, a second. The top of `singular_locus.m2` says why the answer is the same.
- **`f` is chosen by Macaulay2**, one factor for each piece of the singular locus: an
  element of the piece's ideal of lowest degree that does not vanish on a whole component
  of `Y` worth keeping.
- **Components that lie in the singular locus as a whole** (the jacobian of the Landau
  equations is rank deficient on all of them) cannot be seen by the routing function on
  `Y`. If a generator shows they have no positive points, that settles them. Otherwise
  they are treated on their own, with equations whose jacobian has full rank on them. All
  of `Y` is like this for `npltrb`.
- **Starting points** are drawn half from the positive part of the box and half from the
  whole box.

## What has been run

`check_diagrams.jl` on everything: the discriminant degrees agree with Table 1 of the
paper for all diagrams (45 for `env`, 12 for `pentb`). `par` and `npltrb` come out below
the table from nonsingular solutions alone, as they should: the missing component has
fibres of positive dimension in `Y`, and shows up as a singular solution.

| diagram | `Y` | `<name>.m2` | `<name>.jl` |
|---|---|---|---|
| `B2`, `B3`, `B4` | points | run, under a second | `B3` run: one positive point, `α = (1, 1, 1)`, `s = 9` |
| `A3` | curve | run, under a second | run: 4 of 8 routing points positive, on both components |
| `A4` | surface | run, under a second | not run |
| `par` | curve | run, under a second | run, 2 minutes: the result of `../positive_landau.jl` |
| `pltrb` | curve | run, under a second | not run |
| `npltrb` | two surfaces | run, a second | run with 20 starting points only |
| `tdetri` | curve | run, a second | not run |
| `debox` | surface | run, a second | not run |
| `tdebox` | surface | run, 2 seconds | not run |
| `acn` | surface | run | run: 11 of 147 routing points positive, on both components |
| `dbox` | surface | its steps took 40 seconds in a trial; the script itself has not been run | not run |
| `env` | surface | the first saturation did not finish in 15 minutes | not run |
| `pentb` | dimension 5 | not run | not run |

The `results/*.txt` files present are from these runs.

## What the answer does and does not say

- A positive routing point on a component is a positive point of `Y` on it.
- "No positive routing point" on a component means no positive point off the singular
  pieces `f` removes, *if every routing point was found*. The search (gradient flows, then
  monodromy) stops heuristically.
- The singular pieces themselves are reported separately: without positive points (a
  generator has one sign), a positive rational point, or undecided. An undecided piece
  needs a look by hand.
- The chart `m = 1` does not see the part of the discriminant in `m = 0` (one of the two
  components for `npltrb`).
- If a Macaulay2 script does not finish, `positive_landau(name; singular_locus = false)`
  runs with `f = α₁⋯α_{E-1}·M`. Its positive routing points are still positive points of
  `Y`, but a positive piece can be missed and the components are not listed.
