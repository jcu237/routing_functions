-- Written by generate_equations.jl with Landau.jl. Do not edit: change diagrams.jl and
-- run generate_equations.jl again.
--
-- The Landau equations of the one-loop box (A4), with equal masses, in the
-- charts α_4 = 1 and m = 1.
--   edges (the e-th one carries α_e): {{4, 1}, {1, 2}, {2, 3}, {3, 4}}
--   the leg p_i is attached to the vertex: {1, 2, 3, 4}

diagramName = "A4";
R = QQ[α_1, α_2, α_3, s, t, M];

-- the first Symanzik polynomial at α_4 = 1
U = α_1+α_2+α_3+1;

-- the derivatives of F by α_1, ..., α_4 at α_4 = 1
landauEquations = {
    α_2*M+α_3*s-2*α_1-2*α_2-2*α_3+M-2,
    α_1*M+α_3*M-2*α_1-2*α_2-2*α_3+t-2,
    α_1*s+α_2*M-2*α_1-2*α_2-2*α_3+M-2,
    α_1*M+α_2*t+α_3*M-2*α_1-2*α_2-2*α_3-2
    };

-- The system has one more equation, y*α_1*α_2*α_3*U = 1, which says that no α_e
-- vanishes and neither does U. Macaulay2 saturates by these instead (see singular_locus.m2).
alphaVariables = {α_1, α_2, α_3};

-- the coordinates that are positive at a positive point: the α_e and the mass ratio M
positiveVariables = {α_1, α_2, α_3, M};
