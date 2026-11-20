-- Written by generate_equations.jl with Landau.jl. Do not edit: change diagrams.jl and
-- run generate_equations.jl again.
--
-- The Landau equations of the parachute (par), with equal masses, in the
-- charts α_4 = 1 and m = 1.
--   edges (the e-th one carries α_e): {{1, 2}, {1, 3}, {2, 3}, {2, 3}}
--   the leg p_i is attached to the vertex: {1, 1, 3, 2}

diagramName = "par";
R = QQ[α_1, α_2, α_3, s, M];

-- the first Symanzik polynomial at α_4 = 1
U = α_1*α_3+α_2*α_3+α_1+α_2+α_3;

-- the derivatives of F by α_1, ..., α_4 at α_4 = 1
landauEquations = {
    α_2*α_3*s-2*α_1*α_3-2*α_2*α_3+α_2*s-α_3^2+α_3*M-2*α_1-2*α_2-3*α_3-1,
    α_1*α_3*s-2*α_1*α_3+α_1*s-2*α_2*α_3-α_3^2+α_3*M-2*α_1-2*α_2-3*α_3-1,
    α_1*α_2*s-α_1^2-2*α_1*α_2-2*α_1*α_3+α_1*M-α_2^2-2*α_2*α_3+α_2*M-3*α_1-3*α_2-2*α_3-1,
    α_1*α_2*s+α_1*α_3*M+α_2*α_3*M-α_1^2-2*α_1*α_2-3*α_1*α_3-α_2^2-3*α_2*α_3-α_3^2-2*α_1-2*α_2-2*α_3
    };

-- The system has one more equation, y*α_1*α_2*α_3*U = 1, which says that no α_e
-- vanishes and neither does U. Macaulay2 saturates by these instead (see singular_locus.m2).
alphaVariables = {α_1, α_2, α_3};

-- the coordinates that are positive at a positive point: the α_e and the mass ratio M
positiveVariables = {α_1, α_2, α_3, M};
