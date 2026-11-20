-- The incidence variety Y of the Landau equations of the twice doubled-edge triangle (tdetri, Fig. 1g of
-- [LD] = Mizera and Telen, "Landau Discriminants", arXiv:2109.08036), with equal masses, in the
-- charts α_E = 1 and m = 1: its irreducible components, where the jacobian of the Landau
-- equations drops rank on it, and the factors of the routing numerator f of tdetri.jl that
-- remove that locus. The system is in equations/tdetri.m2 and the method in singular_locus.m2.
--
-- [LD], Thm. 2: the Landau discriminant is
--     {s − 4M = s − 4m = 0} ∪ {9m² − 10mM + ms + M² = 0}.
--
-- run with: M2 --script examples/positive_landau/tdetri.m2
-- It took about a second in a test run.

load (currentFileDirectory | "singular_locus.m2");
singularLocusOfDiagram "tdetri";
