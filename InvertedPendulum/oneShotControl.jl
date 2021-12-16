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

tVec=Vector(0:dtSnap:timeIntegre);
nSnap=size(tVec)

# ----- Initialize the control Vector -----
controlInit=0.001 .*(rand(length(tVec)).-0.5);
writedlm("ControlInitialization.txt",controlInit); # Constant

pInit=controlInit;
p=controlInit;

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

# ----- define initial conditions and dy/dt vector -----
dy=zeros(4);
yInit=[0.05*π,0,0,0]; #init conditions
# yInit=[π-0.2*π,0,0,0]; #init conditions

# ----- define time span -----
tSpan=(0.0,timeIntegre);

# ----- Define Forcing vector (currently zero force) -----
forcingVec=zeros(length(tVec));




# ----- solve test ODE problem with no forcing -----
probNoControl = ODEProblem(nonDimCartPendNonLin!, yInit, tSpan, forcingVec);
println("Solving inverted pendulum on a cart withotu control...")
sol = Array(solve(probNoControl, Tsit5(),saveat=0.0:dtSnap:timeIntegre));
solMat=sol;



# ----- plot State just to check in rough -----
close("all")
figure(1,figsize=(10,10));clf();
setLatex(10);

subplot(211)
plot(tVec,solMat[1,:],label=L"$\theta$",linewidth=3)
plot(tVec,solMat[2,:],label=L"$d \theta/dt$",linewidth=3)
legend()

subplot(212)
plot(tVec,solMat[3,:],label="x",linewidth=3)
plot(tVec,solMat[4,:],label="velocity",linewidth=3)
legend()

tight_layout()

# ----- plot Sample Move -----
# makeMovie("invertedPendRandomControl.mp4")

# ========================================================================
# Creating minimization framework for one iteration of the control problem 
# ========================================================================

# ----- Declaring Loss and flux parameters -----

probOneShotControl = ODEProblem(nonDimCartPendNonLin!, yInit, tSpan, p);
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
controlRMSHistory=[]
cb = function () 
    l=loss_adjoint();
    push!(lossHistory,l);
    println(l)

    controlRMS=sqrt(sum(p.^2));
    push!(controlRMSHistory,controlRMS);
end



# ----- testing if gradient alone works -----
# gradsPredict = gradient(() -> sum(predict_adjoint()), Params([p]))
# gradsLoss = gradient(() -> sum(loss_adjoint()), Params([p]))

cb()
opt=ADAM(0.1);
# opt=Descent(0.008);
params=Flux.params(p);
# ====================
# Training begins here
# ====================
@info "Start training"
Flux.train!(loss_adjoint, params, Iterators.repeated((), 100), opt, cb = cb)
@info "Finished Training"


# =========================
# Plotting the loss history 
# =========================
controlSol=predict_adjoint();

controlInit=readdlm("ControlInitialization.txt");

close("all")
figure(1,figsize=(10,10));clf();

subplot(221)
semilogy(lossHistory,label="Loss History",linewidth=3)
legend()

subplot(222)
plot(tVec,p,"-o",label="Control Input",linewidth=3)
plot(tVec,controlInit,label="Initial Control Input",linewidth=3)
legend()

subplot(223)
plot(tVec,solMat[1,:],label=L"$\theta$ no control",linewidth=3)
plot(tVec,solMat[2,:],label=L"$d \theta/dt$ no control",linewidth=3)
plot(tVec,controlSol[1,:],label=L"$\theta$ with control",linewidth=3)
plot(tVec,controlSol[2,:],label=L"$d \theta/dt$ with control",linewidth=3)

legend()

subplot(224)
plot(tVec,solMat[3,:],label="x",linewidth=3)
plot(tVec,solMat[4,:],label="velocity",linewidth=3)
plot(tVec,controlSol[3,:],label="x with control",linewidth=3)
plot(tVec,controlSol[4,:],label="Velocity with control",linewidth=3)
legend()


tight_layout()

# ----- make movie to plot final results for one shot control -----
# solMat=predict_adjoint()
# makeMovie("InvertedpendOneShotControl.mp4")

