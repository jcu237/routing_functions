-- Written by generate_equations.jl with Landau.jl. Do not edit: change diagrams.jl and
-- run generate_equations.jl again.
--
-- The Landau equations of the one-loop triangle, p₁ and p₂ on one vertex (A3), with equal masses, in the
-- charts α_3 = 1 and m = 1.
--   edges (the e-th one carries α_e): {{3, 1}, {1, 2}, {2, 3}}
--   the leg p_i is attached to the vertex: {1, 1, 2, 3}

diagramName = "A3";
R = QQ[α_1, α_2, s, M];

-- the first Symanzik polynomial at α_3 = 1
U = α_1+α_2+1;

-- the derivatives of F by α_1, ..., α_3 at α_3 = 1
landauEquations = {
    α_2*s-2*α_1-2*α_2+M-2,
    α_1*s-2*α_1-2*α_2+M-2,
    α_1*M+α_2*M-2*α_1-2*α_2-2
    };

-- The system has one more equation, y*α_1*α_2*U = 1, which says that no α_e
-- vanishes and neither does U. Macaulay2 saturates by these instead (see singular_locus.m2).
alphaVariables = {α_1, α_2};

-- the coordinates that are positive at a positive point: the α_e and the mass ratio M
positiveVariables = {α_1, α_2, M};
