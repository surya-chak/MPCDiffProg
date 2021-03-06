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
global const timeIntegre=2;

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

println("Solving inverted pendulum on a cart without control...")
sol=Array(eulerCartPendNonLin(yInit,forcingVec));
# sol = Array(solve(probNoControl, Tsit5(),saveat=0.0:dtSnap:timeIntegre));
solMat=sol;
# ----- plot Sample Move -----
# makeMovie("eulerInvertedPendRandomControl.mp4")

# ==================================
# Declaring Loss and flux parameters  
# ==================================

pInit=0.1 .*(rand(length(tVec)).-0.5);
p=pInit;
params=Flux.params(p);


# ----- Forward pass function -----
function predict_adjoint() # Trainable layer
    return Array(eulerCartPendNonLin(yInit,p));
end

# ----- Loss function -----
function loss_adjoint()
    prediction = predict_adjoint();
    loss = sum((prediction[1,:] - π.*ones(nSnap)).^2); # L2 norm
    return loss;
end

opt=ADAM(0.1);

# ----- callback function to monitor training -----
lossHistory = []

cb = function () 
    l=loss_adjoint();
    push!(lossHistory,l);
end


# ====================
# Training begins here
# ====================
cb()
@info "Start training"
Flux.train!(loss_adjoint, params, Iterators.repeated((), 5000), opt, cb = cb)
@info "Finished Training"


