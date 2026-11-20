# Positive Landau discriminants: the parachute (par, Fig. 1c of [LD] = Mizera and
# Telen, "Landau Discriminants", arXiv:2109.08036).
#
# Which irreducible components of the incidence variety Y of its Landau equations contain a
# point with α real and positive and a positive mass ratio M/m? Equal masses, in the charts
# α_E = 1 and m = 1; the unknowns are (α₁, α₂, α₃, y, s, M), and Y is a curve.
#
# [LD], Thm. 2: the Landau discriminant is (M − m)(M² − 10Mm + 9m² + 4ms) = 0.
# This is the diagram of examples/positive_landau.jl and positive_landau.m2, where f was
# chosen by hand.
#
# Run par.m2 first. It writes results/par.txt: the components of Y and its singular locus,
# which the routing numerator f has to vanish on. positive.jl has the method.
#
#   M2 --script examples/positive_landau/par.m2
#   julia -t auto --project=. examples/positive_landau/par.jl
#
# Monodromy is the slow stage. `monodromy_options = (timeout = 3600,)` bounds it, at the price
# of possibly missing routing points; `monodromy_family = :affine` is slower still but does
# not need a start on every component of Y.

include(joinpath(@__DIR__, "positive.jl"))

# options given on the command line (see script_options in positive.jl) come last
result = positive_landau(:par; nstarts = 200, box = 10.0, script_options()...)
