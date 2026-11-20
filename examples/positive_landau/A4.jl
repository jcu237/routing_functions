# Positive Landau discriminants: the one-loop box (A4, Fig. 1a of [LD] = Mizera and
# Telen, "Landau Discriminants", arXiv:2109.08036).
#
# Which irreducible components of the incidence variety Y of its Landau equations contain a
# point with α real and positive and a positive mass ratio M/m? Equal masses, in the charts
# α_E = 1 and m = 1; the unknowns are (α₁, α₂, α₃, y, s, t, M), and Y is a surface.
#
# [LD], eq. (17): the Landau discriminant is st + 4m(4M − s − t) − 4M² = 0.
#
# Run A4.m2 first. It writes results/A4.txt: the components of Y and its singular locus,
# which the routing numerator f has to vanish on. positive.jl has the method.
#
#   M2 --script examples/positive_landau/A4.m2
#   julia -t auto --project=. examples/positive_landau/A4.jl
#
# Monodromy is the slow stage. `monodromy_options = (timeout = 3600,)` bounds it, at the price
# of possibly missing routing points; `monodromy_family = :affine` is slower still but does
# not need a start on every component of Y.

include(joinpath(@__DIR__, "positive.jl"))

# options given on the command line (see script_options in positive.jl) come last
result = positive_landau(:A4; nstarts = 200, box = 10.0, script_options()...)
