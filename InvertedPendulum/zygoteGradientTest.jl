using DiffEqFlux
using DiffEqSensitivity
using Flux
using OrdinaryDiffEq
using Zygote: gradient
using Test #using Plots
using PyCall
using PyPlot



# ===============================
# Setting up the dynamical system
# ===============================


# ----- RHS of the ODE -----
function lotka_volterra(du,u,p,t)
    x, y = u
    α, β, δ, γ = [2.2, 1.0, 2.0, 0.4]
    timeIndex=Int(floor(t/dtSnap))+1;
    f1=p[timeIndex*2-1];
    f2=p[timeIndex*2];
    
    du[1] = dx = (α - β*y)x + f1
    du[2] = dy = (δ*x - γ)y + f2
end

# ----- Details of the time series being tested -----
timeIntegre=10.0;
dtSnap=0.1
tVec=Vector(0:dtSnap:timeIntegre)

# uVec=rand(size(tVec));


# ====================================
# Setting up the learning problem here
# ====================================
p = 0.01 .*(rand(2*length(tVec)).- 0.5)
u0 = [1.0,1.0]                  # Initial Condition
prob = ODEProblem(lotka_volterra,u0,(0.0,timeIntegre),p)


# ----- Prediction function -----
function predict_rd()
    Array(solve(prob,Tsit5(),saveat=dtSnap,reltol=1e-4))
end

# ----- Loss function -----
loss_rd() = sum(abs2,x-1 for x in predict_rd())

opt = ADAM(0.1)


# ----- callback function to observe training -----
lossHistory=[]
cb = function ()
    loss=loss_rd()
    display(loss)
    push!(lossHistory,loss);
    #display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6)))
end



# Display the ODE with the current parameter values.
Flux.train!(loss_rd, Flux.params(p), Iterators.repeated((), 100), opt, cb = cb)



# ----- Calculating gradient of the loss function -----
grads = gradient(() -> loss_rd(), Params([p]))
