-- The incidence variety Y of the Landau equations of the parachute (par, Fig. 1c of
-- [LD] = Mizera and Telen, "Landau Discriminants", arXiv:2109.08036), with equal masses, in the
-- charts α_E = 1 and m = 1: its irreducible components, where the jacobian of the Landau
-- equations drops rank on it, and the factors of the routing numerator f of par.jl that
-- remove that locus. The system is in equations/par.m2 and the method in singular_locus.m2.
--
-- [LD], Thm. 2: the Landau discriminant is (M − m)(M² − 10Mm + 9m² + 4ms) = 0.
-- This is the diagram of examples/positive_landau.jl and positive_landau.m2, where f was
-- chosen by hand.
--
-- run with: M2 --script examples/positive_landau/par.m2
-- It took under a second in a test run.

load (currentFileDirectory | "singular_locus.m2");
singularLocusOfDiagram "par";
