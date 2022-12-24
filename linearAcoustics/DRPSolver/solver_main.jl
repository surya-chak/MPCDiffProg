using Statistics
using JLD2
using Printf
using DifferentialEquations
using PyCall
using PyPlot
using Trapz
using Prettify
using IJulia
using DelimitedFiles
using Revise


# ** ----- For the ML stuff -----
# using Flux
# using DiffEqFlux
# using Optim
# using DiffEqSensitivity
# using Zygote: gradient, @ignore

# ----- Acoustic Dynamics -----
include("utils.jl")
using .GeneralVariables
# ==========================
# * Grid Generation
# ==========================

il = 500;
jl = 500;

x_vec = LinRange(-50, 50, il);
y_vec = LinRange(-50, 50, jl);

# ** ----- for the wavepacket sim -----
# x_vec = LinRange(-10, 90, il);
# y_vec = LinRange(-50, 50, jl);

dx = x_vec[2] - x_vec[1];
dy = y_vec[2] - y_vec[1];

x = zeros(il, jl);
y = zeros(il, jl);

for j = 1:1:jl
    x[:, j] = x_vec
end

for i = 1:1:il
    y[i, :] = y_vec
end



# ==============================
# * Details of the sim
# ==============================

dt = 1e-2;
# dt=5e-3;
nSnap = 5000;
tFin = (nSnap - 1) * dt;
tVec = LinRange(0, tFin, nSnap);

filtVec = 0.5 .* (tanh.(0.5 .* (x .+ 43)) + tanh.(0.5 .* (-x .+ 43)));
filtVec = filtVec .* 0.5 .* (tanh.(0.5 .* (y .+ 43)) + tanh.(0.5 .* (-y .+ 43)));
filtVec = filtVec ./ maximum(filtVec)

close("all")
figure(1, figsize = (5, 5));
clf();
ax = subplot(211)
plot(x[:, 1], filtVec[:, 1])

ax = subplot(212)
plot(y[1, :], filtVec[1, :])

# tight_layout()
# contourf(x, y, filtVec);
# colorbar();


sim1 = Simulation(tVec, x, y, filtVec);

# ================================================
# * Performing Forward prediction
# ================================================

tVec = Array(0:100*dt:15);


init = zeros(il * jl * 2);
init[1:il*jl] = source(sim1, 0.0)[:]
dState = zeros(il * jl * 2);
tspan = (tVec[1], tVec[end])
problem = ODEProblem(RHS_wave!, init[:], tspan, sim1)


sol = solve(problem, Tsit5(), reltol = 1e-4, abstol = 1e-4, saveat = tVec)


# ===================================================================
# * Initializing the machine learning problems
# ===================================================================

timeIntegre = 12.0;
dtSnap = 0.1
function takeStep(u,y0)
    probOneStep = ODEProblem(RHS_wave!, y0, (0.0,dtSnap), sim1);
    yFin=Array(solve(probOneStep, Tsit5(), saveat=dtSnap, reltol=1e-4))[:,2];
    return yFin
end


# ----- function to reInitialize the optimization problem at each step  -----
function initNewOptimization(y0)
    tSpan=(0.0,timeIntegre);
    global probOneShotControl = ODEProblem(RHS_wave!, y0, tSpan, sim1);
end


# ----- Forward pass function -----
function predict_adjoint() # Trainable layer
    return Array(solve(probOneShotControl, Tsit5(), saveat=dtSnap, reltol=1e-4));
end


# ----- Loss function -----
function loss_adjoint()
    prediction = predict_adjoint();
    loss=sum((prediction[1,:]).^2)+sum((prediction[3,:]).^2)+0.25.*sum(p.^2); # Minimizing the full trajectory value
    # loss=sum((prediction[:,end]).^2)+0.75*sum(p.^2); # Just care about the end value
    return loss;
end

# ----- callback function to monitor training -----
lossHistory = []
cb = function ()
    l=loss_adjoint();
    push!(lossHistory,l);
    println(l)
end




# ==========================================================
# * plotting source to check if it works
# ==========================================================
for it = 1:1:length(tVec)
    println(it)
    contourf(sim1.grid.x, sim1.grid.y, reshape(sol[1:il*jl, it], (sim1.grid.il, sim1.grid.jl)),
        30, cmap = ColorMap("jet"))
    colorbar
    clim(-0.05, 0.05)
    plot([-50, 50], [0, 0])
    plot([0, 0], [-50, 50])
    pause(0.1)
end

