using JLD2
using DifferentialEquations
using PyCall
using PyPlot
using Trapz
using Prettify
using IJulia

using Flux
using DiffEqFlux
using Optim
using DiffEqSensitivity
using Zygote

@pyimport matplotlib.animation as anim
# ==========================
# Critical System parameters
# ==========================
global const mP=0.23;           # Pendulum mass as 0.23 kg
global const mC=2.4;            # Cart mass as 2.4 kg
global const L=0.3;            # 30 cms for now allegedly the very limit of what humans can do 
global const g=9.8;            # g=9.8m/s^2
global const dtSnap=0.0001;
global timeIntegre=2;

tVec=Vector(0:dtSnap:timeIntegre);
global nSnap=size(tVec)[1];

include("SystemDynamics.jl")
# ====================================
# Example solve of a noControl problem
# ====================================

# ----- Small note about how the state vector is arranged -----
# The state vector is arranged as: y=
#      [  θ  ]
#      [dθ/dt]
#      [  x  ]
#      [dx/dt]

yInit=[π-0.2*π,0,0,0]; #init conditions
tSpan=(0.0,timeIntegre);
forcingVec=zeros(length(tVec));

# ----- Solving using the Euler integrator -----
solEuler=Array(eulerCartPendNonLin(yInit,forcingVec));

# ----- Solving using the Higher Order integrator -----
probNoControl = ODEProblem(cartPendNonLin!, yInit, tSpan, forcingVec);
solHighOrder = Array(solve(probNoControl, Tsit5(),saveat=0.0:dtSnap:timeIntegre));

error=solHighOrder-solEuler;

prtein(j);

j=j+10;


print(i);
