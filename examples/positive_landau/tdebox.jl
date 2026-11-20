# Positive Landau discriminants: the twice doubled-edge box (tdebox, Fig. 1i of [LD] = Mizera and
# Telen, "Landau Discriminants", arXiv:2109.08036).
#
# Which irreducible components of the incidence variety Y of its Landau equations contain a
# point with α real and positive and a positive mass ratio M/m? Equal masses, in the charts
# α_E = 1 and m = 1; the unknowns are (α₁, …, α₅, y, s, t, M), and Y is a surface.
#
# [LD], Thm. 2: the Landau discriminant is
#     {t − 4m = 40m + 4M − 11t = 0} ∪ {36m² − 40mM + 16ms + 4mt + 4M² − st = 0}.
#
# Run tdebox.m2 first. It writes results/tdebox.txt: the components of Y and its singular locus,
# which the routing numerator f has to vanish on. positive.jl has the method.
#
#   M2 --script examples/positive_landau/tdebox.m2
#   julia -t auto --project=. examples/positive_landau/tdebox.jl
#
# Monodromy is the slow stage. `monodromy_options = (timeout = 3600,)` bounds it, at the price
# of possibly missing routing points; `monodromy_family = :affine` is slower still but does
# not need a start on every component of Y.

include(joinpath(@__DIR__, "positive.jl"))

# options given on the command line (see script_options in positive.jl) come last
result = positive_landau(:tdebox; nstarts = 200, box = 10.0, script_options()...)
