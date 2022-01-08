using JLD2
using Printf
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

yInit=[0.05*π,0,0,0]; #init conditions

yFullSim=zeros(4,nSim);
yFullSim[:,1]=yInit;
controlFullSim=zeros(nSim);

p=zeros(size(tVecLong));

tSpan=(0.0,tSimLong);

probOneShotControl = ODEProblem(nonDimCartPendNonLin!, yInit, tSpan, p);
solMat=Array(solve(probOneShotControl, Tsit5(), saveat=dtSnap, reltol=1e-4));

makeMovie("noControlMovie.mp4");
