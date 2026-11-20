-- The incidence variety Y of the Landau equations of the penta-box (pentb, Fig. 1l of
-- [LD] = Mizera and Telen, "Landau Discriminants", arXiv:2109.08036), with equal masses, in the
-- charts α_E = 1 and m = 1: its irreducible components, where the jacobian of the Landau
-- equations drops rank on it, and the factors of the routing numerator f of pentb.jl that
-- remove that locus. The system is in equations/pentb.m2 and the method in singular_locus.m2.
--
-- [LD], Ex. 13: the Landau discriminant is a hypersurface of degree 12 in the ℙ⁶ with
-- coordinates (s12 : s23 : s34 : s45 : s51 : M : m).
--
-- run with: M2 --script examples/positive_landau/pentb.m2
-- It has not been run. If it does not finish, pentb.jl can do without it: see pentb.jl.

load (currentFileDirectory | "singular_locus.m2");
singularLocusOfDiagram "pentb";
