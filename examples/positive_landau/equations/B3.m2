-- Written by generate_equations.jl with Landau.jl. Do not edit: change diagrams.jl and
-- run generate_equations.jl again.
--
-- The Landau equations of the sunrise (banana with 3 edges) (B3), with equal masses, in the
-- charts α_3 = 1 and m = 1.
--   edges (the e-th one carries α_e): {{1, 2}, {1, 2}, {1, 2}}
--   the leg p_i is attached to the vertex: {1, 1, 2, 2}

diagramName = "B3";
R = QQ[α_1, α_2, s];

-- the first Symanzik polynomial at α_3 = 1
U = α_1*α_2+α_1+α_2;

-- the derivatives of F by α_1, ..., α_3 at α_3 = 1
landauEquations = {
    -2*α_1*α_2-α_2^2+α_2*s-2*α_1-3*α_2-1,
    -α_1^2-2*α_1*α_2+α_1*s-3*α_1-2*α_2-1,
    α_1*α_2*s-α_1^2-3*α_1*α_2-α_2^2-2*α_1-2*α_2
    };

-- The system has one more equation, y*α_1*α_2*U = 1, which says that no α_e
-- vanishes and neither does U. Macaulay2 saturates by these instead (see singular_locus.m2).
alphaVariables = {α_1, α_2};

-- the coordinates that are positive at a positive point: the α_e and the mass ratio M
positiveVariables = {α_1, α_2};
