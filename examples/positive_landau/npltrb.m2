-- The incidence variety Y of the Landau equations of the non-planar triangle-box (npltrb, Fig. 1f of
-- [LD] = Mizera and Telen, "Landau Discriminants", arXiv:2109.08036), with equal masses, in the
-- charts α_E = 1 and m = 1: its irreducible components, where the jacobian of the Landau
-- equations drops rank on it, and the factors of the routing numerator f of npltrb.jl that
-- remove that locus. The system is in equations/npltrb.m2 and the method in singular_locus.m2.
--
-- [LD], Thm. 2: the Landau discriminant is m(s − M) = 0. By Rk. 6 the equations define Y
-- with a multiplicity, so the jacobian is rank deficient on all of Y.
--
-- run with: M2 --script examples/positive_landau/npltrb.m2
-- It took under a second in a test run.

load (currentFileDirectory | "singular_locus.m2");
singularLocusOfDiagram "npltrb";
