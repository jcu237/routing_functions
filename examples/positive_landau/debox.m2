-- The incidence variety Y of the Landau equations of the doubled-edge box (debox, Fig. 1h of
-- [LD] = Mizera and Telen, "Landau Discriminants", arXiv:2109.08036), with equal masses, in the
-- charts α_E = 1 and m = 1: its irreducible components, where the jacobian of the Landau
-- equations drops rank on it, and the factors of the routing numerator f of debox.jl that
-- remove that locus. The system is in equations/debox.m2 and the method in singular_locus.m2.
--
-- [LD], Thm. 2: the Landau discriminant is
--     {t − M = t − m = 0} ∪ {36m²M − 9m²s − 28mMt + 10mst + 4mt² + 4M²t − st² = 0}.
--
-- run with: M2 --script examples/positive_landau/debox.m2
-- It took about 2 seconds in a test run.

load (currentFileDirectory | "singular_locus.m2");
singularLocusOfDiagram "debox";
