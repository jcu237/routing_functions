# Positive Landau discriminants: the bubble, the banana diagram with 2 edges (B2, Fig. 1b of [LD] = Mizera and
# Telen, "Landau Discriminants", arXiv:2109.08036).
#
# Which irreducible components of the incidence variety Y of its Landau equations contain a
# point with α real and positive? (The external mass M does not occur.) Equal masses, in the charts
# α_E = 1 and m = 1; the unknowns are (α₁, y, s), and Y is a finite set of points.
#
# [LD], Ex. 5: the Landau discriminant is s = 4m.
#
# Run B2.m2 first. It writes results/B2.txt: the components of Y and its singular locus,
# which here is empty. Y is finite, so there is nothing to route: the system is solved and
# its real solutions are sorted. positive.jl has the method.
#
#   M2 --script examples/positive_landau/B2.m2
#   julia -t auto --project=. examples/positive_landau/B2.jl

include(joinpath(@__DIR__, "positive.jl"))

# options given on the command line (see script_options in positive.jl) come last
result = positive_landau(:B2; script_options()...)
