# Positive Landau discriminants: the envelope (env, Fig. 1e of [LD] = Mizera and
# Telen, "Landau Discriminants", arXiv:2109.08036).
#
# Which irreducible components of the incidence variety Y of its Landau equations contain a
# point with α real and positive and a positive mass ratio M/m? Equal masses, in the charts
# α_E = 1 and m = 1; the unknowns are (α₁, …, α₅, y, s, t, M), and Y is a surface.
#
# [LD], Ex. 12: the Landau discriminant has five components, of degrees 8, 8, 8, 9 and 12.
#
# Run env.m2 first. It writes results/env.txt: the components of Y and its singular locus,
# which the routing numerator f has to vanish on. positive.jl has the method.
#
#   M2 --script examples/positive_landau/env.m2
#   julia -t auto --project=. examples/positive_landau/env.jl
#
# Monodromy is the slow stage. `monodromy_options = (timeout = 3600,)` bounds it, at the price
# of possibly missing routing points; `monodromy_family = :affine` is slower still but does
# not need a start on every component of Y.
#
# If env.m2 does not finish, `singular_locus = false` runs without it, with
# f = α₁⋯α_{E-1}·M. The positive routing points it finds are positive points of Y all the
# same. What is lost is the guarantee that every positive piece of Y gets one, and the list
# of components (`nid = true` computes them numerically instead).

include(joinpath(@__DIR__, "positive.jl"))

# options given on the command line (see script_options in positive.jl) come last
result = positive_landau(:env; nstarts = 200, box = 10.0, script_options()...)
