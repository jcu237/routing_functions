-- The incidence variety Y of the Landau equations of the one-loop box (A4, Fig. 1a of
-- [LD] = Mizera and Telen, "Landau Discriminants", arXiv:2109.08036), with equal masses, in the
-- charts α_E = 1 and m = 1: its irreducible components, where the jacobian of the Landau
-- equations drops rank on it, and the factors of the routing numerator f of A4.jl that
-- remove that locus. The system is in equations/A4.m2 and the method in singular_locus.m2.
--
-- [LD], eq. (17): the Landau discriminant is st + 4m(4M − s − t) − 4M² = 0.
--
-- run with: M2 --script examples/positive_landau/A4.m2
-- It took under a second in a test run.

load (currentFileDirectory | "singular_locus.m2");
singularLocusOfDiagram "A4";
