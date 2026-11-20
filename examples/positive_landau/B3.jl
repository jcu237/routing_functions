# Positive Landau discriminants: the sunrise, the banana diagram with 3 edges (B3, Fig. 1b of [LD] = Mizera and
# Telen, "Landau Discriminants", arXiv:2109.08036).
#
# Which irreducible components of the incidence variety Y of its Landau equations contain a
# point with α real and positive? (The external mass M does not occur.) Equal masses, in the charts
# α_E = 1 and m = 1; the unknowns are (α₁, α₂, y, s), and Y is a finite set of points.
#
# [LD], Prop. 2: the Landau discriminant is s = 9m and s = m.
#
# Run B3.m2 first. It writes results/B3.txt: the components of Y and its singular locus,
# which here is empty. Y is finite, so there is nothing to route: the system is solved and
# its real solutions are sorted. positive.jl has the method.
#
#   M2 --script examples/positive_landau/B3.m2
#   julia -t auto --project=. examples/positive_landau/B3.jl

include(joinpath(@__DIR__, "positive.jl"))

# options given on the command line (see script_options in positive.jl) come last
result = positive_landau(:B3; script_options()...)
