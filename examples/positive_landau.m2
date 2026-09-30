-- Exact check of examples/positive_landau.jl: which irreducible components of
--
--     V = {(x,p) in (C*)^4 x C^3 : F = dF/dx1 = ... = dF/dx4 = 0}
--
-- contain a point whose x is real and strictly positive.
--
-- Saturating by x1*x2*x3*x4 removes the coordinate hyperplanes, so the minimal primes
-- of J are the irreducible components of V over QQ. Each one gets a certificate:
--
--   * positive: an exact point on it with x > 0;
--   * not positive: a nonzero polynomial in x alone, with positive coefficients, that
--     lies in the prime. It is > 0 on the open orthant, so no point of the component
--     has x > 0.
--
-- run with: M2 --script examples/positive_landau.m2

R = QQ[x1,x2,x3,x4,s,m,M];

F = m*x2*x3^2 + m*x2^2*x3 + m*x3*x1^2 + m*x3^2*x1 + m*x4*x1^2 + m*x4*x2^2 +
    m*x4*x3^2 + m*x4^2*x1 + m*x4^2*x2 + m*x4^2*x3 - M*x4*x2*x3 - M*x4*x3*x1 +
    2*m*x2*x3*x1 + 2*m*x4*x2*x1 + 3*m*x4*x2*x3 + 3*m*x4*x3*x1 -
    s*x2*x3*x1 - s*x4*x2*x1;

I = ideal(F, diff(x1,F), diff(x2,F), diff(x3,F), diff(x4,F));
J = saturate(I, x1*x2*x3*x4);
P = minimalPrimes J;
assert(#P == 7);
assert(J == intersect P);  -- J is radical and P is its prime decomposition

-- exact points with x > 0. The first is on {p = 0}. The other two come from the
-- positive routing points of the .jl file, whose coordinates show the relations
-- exactly: one has s = 0, M = 9m and x3 = x4 = x1 + x2; the other two have
-- x1 = x2, x3 = x4 and (s, m, M) = m (4 - t^2, 1, 5 + 2t), where t = x3/x1.
-- Each is chosen to lie on only one component (checked below), so the three
-- positive components are certified by three different points.
witnesses = {
    {1,2,3,4, 0,0,0},
    {1,2,3,3, 0,1,9},
    {1,1,1,1, 3,1,7}
};
scan(witnesses, q -> assert(all(q_{0..3}, c -> c > 0)));

isOn = (q, Q) -> (phi := map(QQ, R, q); all(flatten entries gens Q, g -> phi g == 0));
allPositive = h -> h != 0 and all(flatten entries last coefficients h, c -> lift(c, QQ) > 0);

scan(witnesses, q -> assert(#positions(P, Q -> isOn(q, Q)) == 1));

npositive = 0;
scan(#P, i -> (
    Q := P#i;
    ws := select(witnesses, q -> isOn(q, Q));
    cert := select(flatten entries gens eliminate({s,m,M}, Q), h -> allPositive h or allPositive(-h));
    -- exactly one of the two certificates, never both and never neither
    assert(#ws > 0 xor #cert > 0);
    print("component " | toString i | ": dim " | toString dim Q | ", degree " | toString degree Q);
    print("    " | toString flatten entries mingens Q);
    if #ws > 0 then (
        npositive = npositive + 1;
        print("    POSITIVE: contains (x; s,m,M) = " | toString ws#0))
    else
        print("    not positive: contains " | toString cert#0 | ", which is > 0 for x > 0");
));
print(toString npositive | " of " | toString(#P) | " components meet the positive orthant");
assert(npositive == 3);

-- the line (M = 9m) and the cubic cross inside the positive orthant, at t = x3/x1 = 2.
-- So V is singular at a point with x > 0, which f = x1*x2*x3*x4 does not remove.
crossing = {1,1,2,2, 0,1,9};
assert(#positions(P, Q -> isOn(crossing, Q)) == 2);
print("the line and the cubic meet at (x; s,m,M) = " | toString crossing);
