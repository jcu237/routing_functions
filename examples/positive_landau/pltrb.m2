-- The incidence variety Y of the Landau equations of the planar triangle-box (pltrb, Fig. 1j of
-- [LD] = Mizera and Telen, "Landau Discriminants", arXiv:2109.08036), with equal masses, in the
-- charts α_E = 1 and m = 1: its irreducible components, where the jacobian of the Landau
-- equations drops rank on it, and the factors of the routing numerator f of pltrb.jl that
-- remove that locus. The system is in equations/pltrb.m2 and the method in singular_locus.m2.
--
-- [LD], Thm. 2: the Landau discriminant has codimension 2,
--     {s − 3m = s − M = 0} ∪ {s − 3m = s − 3M = 0}.
--
-- run with: M2 --script examples/positive_landau/pltrb.m2
-- It took under a second in a test run.

load (currentFileDirectory | "singular_locus.m2");
singularLocusOfDiagram "pltrb";
