using JLD2
using DifferentialEquations
using PyCall
using PyPlot
using Trapz
using Prettify
using IJulia
using DelimitedFiles

using Flux
using DiffEqFlux
using Optim
using DiffEqSensitivity
using Zygote: gradient, @ignore
# import ChainRulesCore: rrule, DoesNotExist,NO_FIELDS
include("SystemDynamics.jl")


@pyimport matplotlib.animation as anim

# ==========================
# Critical System parameters
# ==========================
# global const mP=0.23;           # Pendulum mass as 0.23 kg
# global const mC=2.4*0.25;            # Cart mass as 2.4 kg
# global const L=0.3;            # 30 cms for now allegedly the very limit of what humans can do 
# global const g=9.8;            # g=9.8m/s^2
# global const dtSnap=0.05;
# global const timeIntegre=5;

# ================================================================
# Critical System parameters for the non Dimensionalized equations
# ================================================================
global const δ=5;        # Mass ratio of cart/pendulum
global dtSnap=0.2;       # nonDimensional time step 
global timeIntegre=5;    # nonDimensional time horizon
global const L=1;        # L=1 ... doesn't matter equations are nonDim

# ----- Optimization hyperparameters -----
tVec=Vector(0:dtSnap:timeIntegre);
nSnap=size(tVec)

# ----- Initialize the control Vector -----
controlInit=0.001 .*(rand(length(tVec)).-0.5);
writedlm("ControlInitialization.txt",controlInit); # Constant
p=controlInit;

# ================================================
# Iterative MPC with reinitialization at each step
# ================================================

# ----- Initialize the full controlled run over nSim time steps -----
nSim=100;
tSimLong=(nSim-1).* dtSnap;
tVecLong=LinRange(0,tSimLong,nSim)

yFullSim=zeros(4,nSim);
yFullSim[:,1]=yInit;
controlFullSim=zeros(nSim);

p=0.001 .*(rand(length(tVec)).-0.5);



# ----- function to take one step using the first control Input -----
function takeStep(u,y0)
    probOneStep = ODEProblem(nonDimCartPendNonLin!, y0, (0.0,dtSnap), u);
    yFin=Array(solve(probOneStep, Tsit5(), saveat=dtSnap, reltol=1e-4))[:,2];
    return yFin
end


# ----- function to reInitialize the optimization problem at each step  -----
function initNewOptimization(y0)
    tSpan=(0.0,timeIntegre);
    global probOneShotControl = ODEProblem(nonDimCartPendNonLin!, y0, tSpan, p);
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

# ----- Loop to run over each time step where the optimization problem will be re run -----
for iTime in 1:(nSim-1)

    # ----- Initialize new optimization problem based on y this timestep -----
    initNewOptimization(yFullSim[:,iTime])
    p=0.001 .*(rand(length(tVec)).-0.5);



    tSpan=(0.0,timeIntegre);
    global probOneShotControl = ODEProblem(nonDimCartPendNonLin!, yFullSim[:,iTime], tSpan, p);
    params=Flux.params(p);
    # ----- Solve optimization problem for this timestep -----
    opt=ADAM(0.1);
    @info "Start training"
    Flux.train!(loss_adjoint, params, Iterators.repeated((), 300), opt, cb = cb)
    @info "Finished Training"

    controlFullSim[iTime]=p[1];
    # ----- take one step using the first control input -----
    yFullSim[:,iTime+1]=takeStep(p[1:2],yFullSim[:,iTime]);
end


close("all")
figure(2,figsize=(10,10));clf();
subplot(211)
plot(tVecLong,yFullSim[1,:],label="Simulation",linewidth=3)


subplot(212)
plot(tVecLong,controlFullSim,label="Simulation",linewidth=3)

tight_layout()

