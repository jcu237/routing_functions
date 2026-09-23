restart
R = QQ[x..z,a_0..c_1,t]

clebsch = 81*(x^3 + y^3 + z^3) - 189*(x^2*(y+z) + y^2*(z+x) + z^2*(x+y)) + 54*x*y*z + 126*(x*y + y*z + x*z) - 9*(x^2 + y^2 + z^2) - 9*(x + y + z) + 1

Lines = flatten entries (transpose matrix{{a_0, b_0, c_0}} + t * transpose matrix{{a_1,b_1,c_1}})

evalLines = sub(clebsch, {x => Lines#0, y => Lines#1, z => Lines#2})

evalLines = sub(evalLines, QQ[x..z,a_0..c_1][t])
coeffs = coefficients(evalLines)
coeffs = sub(coeffs#1, R)
J = saturate(ideal coeffs, sub(ideal(a_1,b_1,c_1), R))
use R
I = J + ideal {x - a_0, y - b_0, z - c_0}

I27 = eliminate(toList(a_0..c_1) | {t}, I)

G = gens I27

G_(0,0) == clebsch
g = G_(0,1)

f1 == clebsch
degree g == {9}

"clebsch_cubic.jl" << toExternalString g << endl << close 
