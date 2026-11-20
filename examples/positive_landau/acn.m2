-- The incidence variety Y of the Landau equations of the acnode (acn, Fig. 1d of
-- [LD] = Mizera and Telen, "Landau Discriminants", arXiv:2109.08036), with equal masses, in the
-- charts α_E = 1 and m = 1: its irreducible components, where the jacobian of the Landau
-- equations drops rank on it, and the factors of the routing numerator f of acn.jl that
-- remove that locus. The system is in equations/acn.m2 and the method in singular_locus.m2.
--
-- [LD], Ex. 10: the Landau discriminant has two components, of degrees 4 and 8.
--
-- run with: M2 --script examples/positive_landau/acn.m2
-- It took about 20 seconds in a test run, with an earlier version of singular_locus.m2.

load (currentFileDirectory | "singular_locus.m2");
singularLocusOfDiagram "acn";
