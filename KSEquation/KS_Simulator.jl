using DifferentialEquations


iMax = 200;
xVec = LinRange(0, 1, iMax);

uInit = cos.(π .* xVec);

function KS_RHS!(du, u, p, t)
    du = u;
    
end
