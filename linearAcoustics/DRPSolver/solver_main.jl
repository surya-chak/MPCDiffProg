using Statistics
using PyCall
using PyPlot
using DifferentialEquations

# ----- Acoustic Dynamics -----
include("utils.jl")
using .GeneralVariables
# ===============
# Grid Generation 
# ===============

il=500;
jl=500;

x_vec=LinRange(-10,90,il);
y_vec=LinRange(-50,50,jl);

dx=x_vec[2]-x_vec[1];
dy=y_vec[2]-y_vec[1];

x=zeros(il,jl);
y=zeros(il,jl);

for j=1:1:jl
    x[:,j]=x_vec;
end

for i=1:1:il
    y[i,:]=y_vec;
end



# ===========================
# Details of the source field  
# ===========================

dt=1e-2;
# dt=5e-3;
nSnap=5000;
tFin=(nSnap-1)*dt;
tVec=LinRange(0,tFin,nSnap);

sim1=Simulation(tVec,x,y)

# ================================================
# performing the first 3 iterations using ODE45
# ================================================

tStartVec=[0,dt,2*dt,3*dt];
init=zeros(il*jl*2);
dState=zeros(il*jl*2);
# solving for the first 4 time steps
tspan=(tStartVec[1],tStartVec[end])
problem=ODEProblem(RHS_wave!,init[:],tspan,sim1)

sol=solve(problem,Tsit5(),reltol=1e-8,abstol=1e-8,saveat=tStartVec)


# [t,sol]=ode45(@(t,state) RHS_wave(t,state,x,y),t_startVec,init);



soln=squeeze(sol[4,:])';
solnm1=squeeze(sol[3,:])';
solnm2=squeeze(sol[2,:])';
solnm3=squeeze(sol[1,:])';


it=4;
RHS_soln=RHS_wave(t[it],soln,x,y);
RHS_solnm1=RHS_wave(t[it-1],solnm1,x,y);
RHS_solnm2=RHS_wave(t[it-2],solnm2,x,y);
RHS_solnm3=RHS_wave(t[it-3],solnm3,x,y);


saved_sol=zeros(il*jl*2,nn/50+1);

ctr=3;
saved_ctr=1;

# --------------------------
# Time marching coefficients
# --------------------------

b_0=2.30255809;
b_1=-2.49100760;
b_2=1.57434093;
b_3=-0.38589142;


# =========================================
# Marching through time for every iteration
# =========================================
for it=4:1:nn

    solnp1=soln+dt*(b_0*RHS_soln+b_1*RHS_solnm1+b_2*RHS_solnm2+ ...
           b_3*RHS_solnm3);
    
    ctr=ctr+1;

    if ctr==50
        saved_sol[:,saved_ctr]=solnp1;
        println(string("saving snap" num2str(saved_ctr));
        saved_ctr=saved_ctr+1;
        println(string("iteration = " num2str(it));
        
        ctr=0;
    end

    # updating RHS values for the next time step
    RHS_solnm3=RHS_solnm2;
    RHS_solnm2=RHS_solnm1;
    RHS_solnm1=RHS_soln;
    RHS_soln=RHS_wave(tVec[it+1],solnp1,x,y);

    # updating solution value for the next time step
    soln=solnp1;

end



