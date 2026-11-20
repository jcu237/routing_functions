-- The incidence variety Y of the Landau equations of the double-box (dbox, Fig. 1k of
-- [LD] = Mizera and Telen, "Landau Discriminants", arXiv:2109.08036), with equal masses, in the
-- charts α_E = 1 and m = 1: its irreducible components, where the jacobian of the Landau
-- equations drops rank on it, and the factors of the routing numerator f of dbox.jl that
-- remove that locus. The system is in equations/dbox.m2 and the method in singular_locus.m2.
--
-- [LD], Thm. 2: the Landau discriminant has two components, of degrees 2 and 4. The first
-- is 4mM − ms − 4mt + st = 0.
--
-- run with: M2 --script examples/positive_landau/dbox.m2
-- Its steps took about 40 seconds in a trial. The script itself has not been run.

load (currentFileDirectory | "singular_locus.m2");
singularLocusOfDiagram "dbox";
