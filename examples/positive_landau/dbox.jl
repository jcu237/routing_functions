# Positive Landau discriminants: the double-box (dbox, Fig. 1k of [LD] = Mizera and
# Telen, "Landau Discriminants", arXiv:2109.08036).
#
# Which irreducible components of the incidence variety Y of its Landau equations contain a
# point with α real and positive and a positive mass ratio M/m? Equal masses, in the charts
# α_E = 1 and m = 1; the unknowns are (α₁, …, α₆, y, s, t, M), and Y is a surface.
#
# [LD], Thm. 2: the Landau discriminant has two components, of degrees 2 and 4. The first
# is 4mM − ms − 4mt + st = 0.
#
# Run dbox.m2 first. It writes results/dbox.txt: the components of Y and its singular locus,
# which the routing numerator f has to vanish on. positive.jl has the method.
#
#   M2 --script examples/positive_landau/dbox.m2
#   julia -t auto --project=. examples/positive_landau/dbox.jl
#
# Monodromy is the slow stage. `monodromy_options = (timeout = 3600,)` bounds it, at the price
# of possibly missing routing points; `monodromy_family = :affine` is slower still but does
# not need a start on every component of Y.

include(joinpath(@__DIR__, "positive.jl"))

# options given on the command line (see script_options in positive.jl) come last
result = positive_landau(:dbox; nstarts = 200, box = 10.0, script_options()...)
