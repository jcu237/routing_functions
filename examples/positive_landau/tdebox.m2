-- The incidence variety Y of the Landau equations of the twice doubled-edge box (tdebox, Fig. 1i of
-- [LD] = Mizera and Telen, "Landau Discriminants", arXiv:2109.08036), with equal masses, in the
-- charts α_E = 1 and m = 1: its irreducible components, where the jacobian of the Landau
-- equations drops rank on it, and the factors of the routing numerator f of tdebox.jl that
-- remove that locus. The system is in equations/tdebox.m2 and the method in singular_locus.m2.
--
-- [LD], Thm. 2: the Landau discriminant is
--     {t − 4m = 40m + 4M − 11t = 0} ∪ {36m² − 40mM + 16ms + 4mt + 4M² − st = 0}.
--
-- run with: M2 --script examples/positive_landau/tdebox.m2
-- It took a few seconds in a test run.

load (currentFileDirectory | "singular_locus.m2");
singularLocusOfDiagram "tdebox";
