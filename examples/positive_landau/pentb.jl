# Positive Landau discriminants: the penta-box (pentb, Fig. 1l of [LD] = Mizera and
# Telen, "Landau Discriminants", arXiv:2109.08036).
#
# Which irreducible components of the incidence variety Y of its Landau equations contain a
# point with α real and positive and a positive mass ratio M/m? Equal masses, in the charts
# α_E = 1 and m = 1; the unknowns are (α₁, …, α₇, y, s12, s23, s34, s45, s51, M), and Y is of dimension 5.
#
# [LD], Ex. 13: the Landau discriminant is a hypersurface of degree 12 in the ℙ⁶ with
# coordinates (s12 : s23 : s34 : s45 : s51 : M : m).
#
# Run pentb.m2 first. It writes results/pentb.txt: the components of Y and its singular locus,
# which the routing numerator f has to vanish on. positive.jl has the method.
#
#   M2 --script examples/positive_landau/pentb.m2
#   julia -t auto --project=. examples/positive_landau/pentb.jl
#
# Monodromy is the slow stage. `monodromy_options = (timeout = 3600,)` bounds it, at the price
# of possibly missing routing points; `monodromy_family = :affine` is slower still but does
# not need a start on every component of Y.
#
# If pentb.m2 does not finish, `singular_locus = false` runs without it, with
# f = α₁⋯α_{E-1}·M. The positive routing points it finds are positive points of Y all the
# same. What is lost is the guarantee that every positive piece of Y gets one, and the list
# of components (`nid = true` computes them numerically instead).

include(joinpath(@__DIR__, "positive.jl"))

# options given on the command line (see script_options in positive.jl) come last
result = positive_landau(:pentb; nstarts = 200, box = 10.0, script_options()...)
