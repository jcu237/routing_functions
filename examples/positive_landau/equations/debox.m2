-- Written by generate_equations.jl with Landau.jl. Do not edit: change diagrams.jl and
-- run generate_equations.jl again.
--
-- The Landau equations of the doubled-edge box (debox), with equal masses, in the
-- charts α_5 = 1 and m = 1.
--   edges (the e-th one carries α_e): {{4, 1}, {1, 2}, {2, 3}, {3, 4}, {3, 4}}
--   the leg p_i is attached to the vertex: {1, 2, 3, 4}

diagramName = "debox";
R = QQ[α_1, α_2, α_3, α_4, s, t, M];

-- the first Symanzik polynomial at α_5 = 1
U = α_1*α_4+α_2*α_4+α_3*α_4+α_1+α_2+α_3+α_4;

-- the derivatives of F by α_1, ..., α_5 at α_5 = 1
landauEquations = {
    α_2*α_4*M+α_3*α_4*s-2*α_1*α_4-2*α_2*α_4+α_2*M-2*α_3*α_4+α_3*s-α_4^2+α_4*M-2*α_1-2*α_2-2*α_3-3*α_4-1,
    α_1*α_4*M+α_3*α_4*M-2*α_1*α_4+α_1*M-2*α_2*α_4-2*α_3*α_4+α_3*M-α_4^2+α_4*t-2*α_1-2*α_2-2*α_3-3*α_4-1,
    α_1*α_4*s+α_2*α_4*M-2*α_1*α_4+α_1*s-2*α_2*α_4+α_2*M-2*α_3*α_4-α_4^2+α_4*M-2*α_1-2*α_2-2*α_3-3*α_4-1,
    α_1*α_2*M+α_1*α_3*s+α_2*α_3*M-α_1^2-2*α_1*α_2-2*α_1*α_3-2*α_1*α_4+α_1*M-α_2^2-2*α_2*α_3-2*α_2*α_4+α_2*t-α_3^2-2*α_3*α_4+α_3*M-3*α_1-3*α_2-3*α_3-2*α_4-1,
    α_1*α_2*M+α_1*α_3*s+α_1*α_4*M+α_2*α_3*M+α_2*α_4*t+α_3*α_4*M-α_1^2-2*α_1*α_2-2*α_1*α_3-3*α_1*α_4-α_2^2-2*α_2*α_3-3*α_2*α_4-α_3^2-3*α_3*α_4-α_4^2-2*α_1-2*α_2-2*α_3-2*α_4
    };

-- The system has one more equation, y*α_1*α_2*α_3*α_4*U = 1, which says that no α_e
-- vanishes and neither does U. Macaulay2 saturates by these instead (see singular_locus.m2).
alphaVariables = {α_1, α_2, α_3, α_4};

-- the coordinates that are positive at a positive point: the α_e and the mass ratio M
positiveVariables = {α_1, α_2, α_3, α_4, M};
