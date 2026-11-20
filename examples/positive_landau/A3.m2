-- The incidence variety Y of the Landau equations of the one-loop triangle (A3, Fig. 1a of
-- [LD] = Mizera and Telen, "Landau Discriminants", arXiv:2109.08036), with equal masses, in the
-- charts α_E = 1 and m = 1: its irreducible components, where the jacobian of the Landau
-- equations drops rank on it, and the factors of the routing numerator f of A3.jl that
-- remove that locus. The system is in equations/A3.m2 and the method in singular_locus.m2.
--
-- p₁ and p₂ sit on one vertex, because Landau.jl has kinematics for 4 and 5 legs only.
-- [LD] does not list the discriminant of this diagram.
--
-- run with: M2 --script examples/positive_landau/A3.m2
-- It took under a second in a test run.

load (currentFileDirectory | "singular_locus.m2");
singularLocusOfDiagram "A3";
