-- Shared functions of the <name>.m2 scripts: the irreducible components of the incidence
-- variety Y of the Landau equations of a Feynman diagram, in the charts α_E = 1 and m = 1,
-- and its singular locus. It is the computation of examples/positive_landau.m2, for any
-- diagram. The results go to the screen and to results/<name>.txt, which positive.jl reads
-- to build the routing numerator f.
--
-- Three things differ from positive_landau.m2. The diagrams beyond the parachute need all
-- of them, and on the parachute they give the same four components and the same three
-- pieces of the singular locus.
--
--   * No y. The last equation of the system, y*α_1*...*α_{E-1}*U = 1, makes Y the graph of
--     1/(α_1*...*α_{E-1}*U) over the part of V(landauEquations) where no α_e and not U
--     vanishes. So Y and that part have the same components, and the jacobian of the full
--     system has rank 1 + (rank of the jacobian of landauEquations). We work in the ring
--     without y, and remove the components inside α_1*...*α_{E-1}*U = 0 by saturation.
--   * The jacobian is taken modulo each component P of Y in turn, instead of adding its
--     minors to the ideal of Y: the union over P of V(P + minors) is V(Y + minors).
--   * A component on which the jacobian has rank below the number of equations everywhere
--     (its dimension is too large, or Y is not reduced along it) lies in the singular
--     locus as a whole, so the routing function on Y cannot look at it. If no generator
--     rules out positive points on it, it becomes a stratum of its own: as many equations
--     as its codimension, whose jacobian has full rank at its general point, and the
--     locus on it where that rank drops.

say = x -> (stdio << x << endl << flush;);        -- these computations can run for hours

polynomials = Q -> flatten entries mingens Q;
gensString = Q -> concatenate between(" ; ", apply(polynomials Q, toString));
byDegree = L -> apply(sort apply(#L, i -> (first degree L#i, size L#i, i)), d -> L#(last d));
-- low-degree elements of Q to choose from: its minimal generators and a Gröbner basis
candidates = Q -> byDegree unique(polynomials Q | flatten entries gens gb Q);

-- ----------------------------------------------------------------------------------------
-- positivity
-- ----------------------------------------------------------------------------------------
-- Whether g cannot vanish where all the variables in `pos` are positive: it involves only
-- those variables, and all its coefficients have the same sign. E.g. α_3 + 1.
signDefinite = (g, pos) -> (
    c := apply(terms g, leadCoefficient);
    g != 0 and isSubset(support g, pos) and (all(c, x -> x > 0) or all(c, x -> x < 0)));

-- An element of Q that shows V(Q) has no positive point, or null if none is found. Finding
-- none proves nothing.
positiveFree = (Q, pos) -> (
    L := select(1, candidates Q, g -> signDefinite(g, pos));
    if #L > 0 then first L else null);

-- What is known about positive points of V(Q): "none", "some: " and the coordinates when
-- V(Q) is a single rational point and it is positive, or "unknown".
positivity = (Q, pos) -> (
    if positiveFree(Q, pos) =!= null then return "none";
    if dim Q == 0 and degree Q == 1 then (
        if any(pos, v -> lift(v % Q, QQ) <= 0) then return "none";
        return "some: " | concatenate between(" ", apply(gens ring Q, v -> toString lift(v % Q, QQ))));
    "unknown");

-- ----------------------------------------------------------------------------------------
-- components and the rank of the jacobian
-- ----------------------------------------------------------------------------------------
-- The irreducible components of V(H) that do not lie in the zero set of an element of
-- `removed`. Saturating by U before the α_e is about ten times faster than the other way
-- round (8 s against 70 s on the acnode).
openComponents = (H, removed) -> (
    J := ideal H;
    scan(removed, h -> (J = saturate(J, h); say("    saturated by " | toString h)));
    minimalPrimes J);

-- Whether the jacobian of H has rank below #H at every point of V(P), P a prime.
rankMinors = (H, P) -> minors(#H, sub(jacobian matrix{H}, (ring P)/P));
degenerateOn = (H, P) -> rankMinors(H, P) == 0;

-- The components of the locus on V(P) where the jacobian of H has rank below #H, without
-- those inside the zero set of an element of `removed`.
rankDropOn = (H, P, removed) -> (
    L := minimalPrimes(P + lift(rankMinors(H, P), ring P));
    select(L, Q -> all(removed, h -> h % Q != 0)));

-- the primes of L that contain no other prime of L, i.e. the largest of the sets V(Q)
minimalOnes = L -> (
    L = unique L;
    select(L, Q -> not any(L, Q' -> Q' != Q and isSubset(Q', Q))));

-- ----------------------------------------------------------------------------------------
-- the factor of the routing numerator that removes one piece
-- ----------------------------------------------------------------------------------------
-- A polynomial that vanishes on V(Q) but on none of the components in `keep`, as a pair
-- (kind, polynomial):
--   "cut"       an element of Q of lowest degree that vanishes on no component in `keep`.
--               One that has a single sign on the positive points is preferred: it removes
--               nothing positive;
--   "squares"   no such element among the candidates and their random combinations: use
--               the sum of the squares of the generators of Q (the polynomial is null).
-- A cut also removes a hypersurface section of the components in `keep`. That is harmless:
-- it leaves a dense open part of each.
removalFactor = (Q, keep, pos) -> (
    safe := g -> all(keep, P -> g % P != 0);
    cert := positiveFree(Q, pos);
    cuts := select(1, candidates Q, safe);
    if cert =!= null and (#cuts == 0 or first degree cert <= first degree first cuts) then return ("cut", cert);
    if #cuts > 0 then return ("cut", first cuts);
    gs := polynomials Q;
    for trial from 1 to 20 do (
        g := sum(gs, h -> random(-9, 9) * h);
        if g != 0 and safe g then return ("cut", g));
    ("squares", null));

-- ----------------------------------------------------------------------------------------
-- equations for a component the jacobian is rank deficient on
-- ----------------------------------------------------------------------------------------
-- codim P elements of P whose jacobian has full rank at the general point of V(P), so that
-- V(P) is a component of their zero set, smooth and of the right dimension there: the
-- generators of P if there are codim P of them, else the first subset of them that works,
-- else random combinations. null if nothing works.
localEquations = P -> (
    c := codim P;
    gs := byDegree polynomials P;
    if #gs == c then return gs;
    for H in subsets(gs, c) do if not degenerateOn(H, P) then return H;
    for trial from 1 to 20 do (
        H := apply(c, i -> sum(gs, h -> random(-9, 9) * h));
        if not degenerateOn(H, P) then return H);
    null);

-- ----------------------------------------------------------------------------------------
-- the whole computation
-- ----------------------------------------------------------------------------------------
-- The lines of results/<name>.txt. Their fields are separated by " | "; positive.jl reads them.
resultLines = new MutableList;
emit = fields -> (resultLines#(#resultLines) = concatenate between(" | ", apply(fields, toString)););

-- The pieces of a stratum: what has to be removed from it, and with which factor of f.
reportPieces = (pieces, keep, pos) -> (
    if #pieces == 0 then say "    nowhere";
    scan(#pieces, i -> (
        Q := pieces#i;
        (kind, g) := removalFactor(Q, keep, pos);
        p := positivity(Q, pos);
        say("    " | toString(i + 1) | ". dim " | toString dim Q | ": " | toString polynomials Q);
        say("       " | (if p == "none" then "no positive points. " else if p == "unknown" then "" else "a positive point. ")
                      | (if kind == "cut" then "f gets the factor " | toString g
                         else "f gets the sum of the squares of these generators"));
        emit {"piece", i + 1, dim Q, p, kind, if g === null then "-" else g, gensString Q})));

-- landauSingularLocus(name, landauEquations, U, alphaVariables, positiveVariables)
landauSingularLocus = (name, LE, U, alphas, pos) -> (
    S := ring U;
    k := #LE;
    removed := {U} | alphas;
    emit {"diagram", name};
    emit {"ring", concatenate between(" ", apply(gens S, toString))};

    -- 1. the irreducible components of Y, i.e. what nid finds
    say("-- " | name | ": " | toString k | " Landau equations in " | toString numgens S | " unknowns (and y)");
    say "";
    say "irreducible components of Y:";
    comps := openComponents(LE, removed);
    degenerate := apply(comps, P -> degenerateOn(LE, P));
    certificate := apply(comps, P -> positiveFree(P, pos));
    emit {"stratum", 1, "Y", numgens S - k};
    scan(LE, g -> emit {"equation", g});
    scan(#comps, i -> (
        P := comps#i;
        say("    " | toString(i + 1) | ". dim " | toString dim P | ", degree " | toString degree P | ": " | toString polynomials P);
        if degenerate#i then say("       the jacobian has rank below " | toString k | " on all of it");
        if certificate#i =!= null then say("       no positive points: it contains " | toString certificate#i);
        emit {"component", i + 1, dim P, degree P, if degenerate#i then "degenerate" else "regular",
              if certificate#i === null then "unknown" else "none", gensString P}));

    -- 2. where the jacobian drops rank on Y. Y needs to be smooth of dimension
    -- (number of unknowns) - k off V(f), so the routing numerator f has to vanish on all of it.
    say "";
    say "where the jacobian of the Landau equations drops rank on Y:";
    pieces := minimalOnes flatten apply(#comps, i -> (
        if degenerate#i then {comps#i} else rankDropOn(LE, comps#i, removed)));
    -- a factor of f may vanish on a whole component only if that component is lost anyway
    keep := select(#comps, i -> not degenerate#i and certificate#i === null);
    reportPieces(pieces, comps_keep, pos);

    -- 3. the components the routing function on Y cannot look at, unless they have no positive points
    todo := select(#comps, i -> degenerate#i and certificate#i === null);
    scan(#todo, j -> (
        P := comps#(todo#j);
        say "";
        say("component " | toString(todo#j + 1) | " lies in the singular locus and could have positive points.");
        H := localEquations P;
        if H === null then (
            say "    found no equations with a jacobian of full rank on it: it needs a look by hand";
            emit {"unresolved", todo#j + 1})
        else (
            say("    it is a component of the zero set of " | toString H);
            say "    where the jacobian of these drops rank on it:";
            emit {"stratum", j + 2, "component " | toString(todo#j + 1) | " of Y", dim P};
            scan(H, g -> emit {"equation", g});
            emit {"component", todo#j + 1, dim P, degree P, "regular", "unknown", gensString P};
            reportPieces(rankDropOn(H, P, removed), {P}, pos))));
    emit {"done"};
    toList resultLines);

-- The script of one diagram: the equations written by generate_equations.jl, the
-- computation, and results/<name>.txt.
landauDirectory = currentFileDirectory;      -- where this file is, whoever loads it
singularLocusOfDiagram = name -> (
    load(landauDirectory | "equations/" | name | ".m2");
    lines := landauSingularLocus(name, landauEquations, U, alphaVariables, positiveVariables);
    if not isDirectory(landauDirectory | "results") then makeDirectory(landauDirectory | "results");
    file := landauDirectory | "results/" | name | ".txt";
    out := openOut file;
    out << "# The singular locus of the Landau incidence variety Y of the diagram " << name << "," << endl;
    out << "# written by singular_locus.m2 and read by positive.jl. The fields are separated by \" | \":" << endl;
    out << "#   ring | the variables, without y" << endl;
    out << "#   stratum | number | what it is | dimension      followed by its equations, components and pieces" << endl;
    out << "#   equation | polynomial                          the y equation is not listed" << endl;
    out << "#   component | number in Y | dimension | degree | regular or degenerate | positive points: none or unknown | generators" << endl;
    out << "#   piece | number | dimension | positive points: none, some: coordinates, or unknown | cut or squares | factor of f | generators" << endl;
    out << "#   unresolved | number in Y                       a degenerate component no equations were found for" << endl;
    scan(lines, line -> out << line << endl);
    close out;
    say "";
    say("wrote " | file););
