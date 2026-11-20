-- The incidence variety Y of the Landau equations of the banana diagram with 4 edges (B4, Fig. 1b of
-- [LD] = Mizera and Telen, "Landau Discriminants", arXiv:2109.08036), with equal masses, in the
-- charts α_E = 1 and m = 1: its irreducible components, where the jacobian of the Landau
-- equations drops rank on it, and the factors of the routing numerator f of B4.jl that
-- remove that locus. The system is in equations/B4.m2 and the method in singular_locus.m2.
--
-- [LD], Prop. 2: the Landau discriminant is s = 16m and s = 4m.
--
-- run with: M2 --script examples/positive_landau/B4.m2
-- It took under a second in a test run.

load (currentFileDirectory | "singular_locus.m2");
singularLocusOfDiagram "B4";
