-- Exact computations for examples/positive_landau.jl: the parachute incidence variety Y,
-- in the charts α_4 = 1 and m = 1, with variables (α_1, α_2, α_3, y, s, M). The equations
-- are the Landau equations from positive_landau.jl, computed with Landau.jl.
--
-- run with: M2 --script examples/positive_landau.m2

R = QQ[α_1..α_3, y, s, M];

U = (α_1 + α_2)*(α_3 + 1) + α_3;   -- first Symanzik polynomial at α_4 = 1
G = {
    M*α_3 - 2*α_1*α_3 - 2*α_1 + α_2*α_3*s - 2*α_2*α_3 + α_2*s - 2*α_2 - α_3^2 - 3*α_3 - 1,
    M*α_3 + α_1*α_3*s - 2*α_1*α_3 + α_1*s - 2*α_1 - 2*α_2*α_3 - 2*α_2 - α_3^2 - 3*α_3 - 1,
    M*α_1 + M*α_2 - α_1^2 + α_1*α_2*s - 2*α_1*α_2 - 2*α_1*α_3 - 3*α_1 - α_2^2 - 2*α_2*α_3 - 3*α_2 - 2*α_3 - 1,
    M*α_1*α_3 + M*α_2*α_3 - α_1^2 + α_1*α_2*s - 2*α_1*α_2 - 3*α_1*α_3 - 2*α_1 - α_2^2 - 3*α_2*α_3 - 2*α_2 - α_3^2 - 2*α_3,
    y*α_1*α_2*α_3*U - 1};
Y = ideal G;



-- 1. Where the 6 x 5 jacobian of G drops rank on Y. V(G) needs to be smooth of dimension 1 off V(f),
-- so the routing numerator f has to vanish on all of this locus.
rankDrop = minimalPrimes(Y + minors(5, jacobian matrix{G}));
print "where the jacobian of G drops rank on Y:";
scan(rankDrop, Q -> print("    dim " | toString dim Q | ": " | toString flatten entries mingens Q));

-- This prints three pieces:
--   * the surface M = 1, α_3 = -1. It is a whole component of Y, and it has dimension 2
--     although G is 5 equations in 6 unknowns, so the jacobian has rank <= 4 on all of it;
--   * α = (1/2, 1/2, 1), s = 0, M = 9, where the line M = 9 crosses the quadric component
--     (the curve over M^2 - 10M + 9 + 4s = 0, see part 2). This point is positive;
--   * α = (-1/2, -1/2, 1), s = 0, M = 1, where the line M = 1 crosses the quadric component.
--
-- The routing numerator of positive_landau.jl vanishes on all three: M - 1 on the surface
-- and the second crossing, and (α_1 - α_2)^2 + (M - 9)^2 on the first.
f = α_1*α_2*α_3*M*(M - 1)*((α_1 - α_2)^2 + (M - 9)^2);
print("f vanishes on all of it: " | toString all(rankDrop, Q -> f % Q == 0));


-- 2. The irreducible components of Y, i.e. what nid finds in positive_landau.jl.
print "";
print "irreducible components of Y:";
scan(minimalPrimes Y, Q -> print("    dim " | toString dim Q | ": " | toString flatten entries mingens Q));

-- This prints four components:
--   * the surface M = 1, α_3 = -1, which has no positive points;
--   * the line M = 1, s = 0, α_3 = 1, α_1 + α_2 = -1, which has no positive points;
--   * the line M = 9, s = 0, α_3 = 1, α_1 + α_2 = 1, which is positive for 0 < α_1 < 1;
--   * the quadric component α_1 = α_2 = 2/(M - 5), α_3 = 1, M^2 - 10M + 9 + 4s = 0, which
--     is positive for M > 5.
