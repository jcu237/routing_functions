-- Written by generate_equations.jl with Landau.jl. Do not edit: change diagrams.jl and
-- run generate_equations.jl again.
--
-- The Landau equations of the bubble (banana with 2 edges) (B2), with equal masses, in the
-- charts α_2 = 1 and m = 1.
--   edges (the e-th one carries α_e): {{1, 2}, {1, 2}}
--   the leg p_i is attached to the vertex: {1, 1, 2, 2}

diagramName = "B2";
R = QQ[α_1, s];

-- the first Symanzik polynomial at α_2 = 1
U = α_1+1;

-- the derivatives of F by α_1, ..., α_2 at α_2 = 1
landauEquations = {
    -2*α_1+s-2,
    α_1*s-2*α_1-2
    };

-- The system has one more equation, y*α_1*U = 1, which says that no α_e
-- vanishes and neither does U. Macaulay2 saturates by these instead (see singular_locus.m2).
alphaVariables = {α_1};

-- the coordinates that are positive at a positive point: the α_e and the mass ratio M
positiveVariables = {α_1};
