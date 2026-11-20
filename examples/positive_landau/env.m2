-- The incidence variety Y of the Landau equations of the envelope (env, Fig. 1e of
-- [LD] = Mizera and Telen, "Landau Discriminants", arXiv:2109.08036), with equal masses, in the
-- charts α_E = 1 and m = 1: its irreducible components, where the jacobian of the Landau
-- equations drops rank on it, and the factors of the routing numerator f of env.jl that
-- remove that locus. The system is in equations/env.m2 and the method in singular_locus.m2.
--
-- [LD], Ex. 12: the Landau discriminant has five components, of degrees 8, 8, 8, 9 and 12.
--
-- run with: M2 --script examples/positive_landau/env.m2
-- In a test run the first saturation (by U) did not finish in 15 minutes, and nothing after it
-- has been run. If it does not finish, env.jl can do without it: see env.jl.

load (currentFileDirectory | "singular_locus.m2");
singularLocusOfDiagram "env";
