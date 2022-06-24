using Statistics
using PyCall
using PyPlot
using DifferentialEquations

# ----- Acoustic Dynamics -----
include("utils.jl")
using .GeneralVariables
# ===============
# * Grid Generation 
# ===============

il = 500;
jl = 500;

x_vec = LinRange(-10, 90, il);
y_vec = LinRange(-50, 50, jl);

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



# ===========================
# * Details of the source field  
# ===========================

dt = 1e-2;
# dt=5e-3;
nSnap = 5000;
tFin = (nSnap - 1) * dt;
tVec = LinRange(0, tFin, nSnap);

sim1 = Simulation(tVec, x, y);

# ================================================
# * performing the first 3 iterations using ODE45
# ================================================

# tStartVec = [0, dt, 2 * dt, 3 * dt];

# tStartVec=LinRange(0,1,dt);
tStartVec = Array(0:10*dt:50)

init = zeros(il * jl * 2);
init[1:il*jl] = source(sim1, 0.0)[:]
dState = zeros(il * jl * 2);
# solving for the first 4 time steps
tspan = (tStartVec[1], tStartVec[end])
problem = ODEProblem(RHS_wave!, init[:], tspan, sim1)

sol = solve(problem, Tsit5(), reltol = 1e-4, abstol = 1e-4, saveat = tStartVec)


# [t,sol]=ode45(@(t,state) RHS_wave(t,state,x,y),t_startVec,init);


# ====================================
# plotting source to check if it works
# ====================================
for it = 1:10:length(tStartVec)
    println(it)
    contourf(sim1.grid.x, sim1.grid.y, reshape(sol[1:il*jl, it], (sim1.grid.il, sim1.grid.jl)),
        30, cmap = ColorMap("jet"))
    clim(-0.05, 0.05)
    pause(0.1)
end

# soln = sol[:, 4];
# solnm1 = sol[:, 3];
# solnm2 = sol[:, 2];
# solnm3 = sol[:, 1];

# contourf(sim1.grid.x, sim1.grid.y, reshape(soln, (il, jl, 2))[:, :, 1], cmap = ColorMap("jet"))

# print(string(maximum(soln), " and ", minimum(soln)), "\n")
# print(string(maximum(solnm1), " and ", minimum(solnm1)), "\n")
# print(string(maximum(solnm2), " and ", minimum(solnm2)), "\n")
# print(string(maximum(solnm3), " and ", minimum(solnm3)), "\n")
# # reshape(sol[:, 1],(il,jl,1))

# it = 4;
# RHS_soln = zeros(size(soln));
# RHS_solnm1 = zeros(size(soln));
# RHS_solnm2 = zeros(size(soln));
# RHS_solnm3 = zeros(size(soln));


# RHS_wave!(RHS_soln, soln, sim1, tVec[it]);
# RHS_wave!(RHS_solnm1, solnm1, sim1, tVec[it-1]);
# RHS_wave!(RHS_solnm2, solnm2, sim1, tVec[it-2]);
# RHS_wave!(RHS_solnm3, solnm3, sim1, tVec[it-3]);

# saved_sol = zeros(il * jl * 2, Int(nSnap / 50) + 1);

# ctr = 3;
# saved_ctr = 1;

# # --------------------------
# # Time marching coefficients
# # --------------------------

# b_0 = 2.30255809;
# b_1 = -2.49100760;
# b_2 = 1.57434093;
# b_3 = -0.38589142;


# # =========================================
# # Marching through time for every iteration
# # =========================================
# ctr = 0
# for it = 4:1:nSnap

#     solnp1 = soln + dt * (b_0 * RHS_soln + b_1 * RHS_solnm1 + b_2 * RHS_solnm2 +
#                           b_3 * RHS_solnm3)

#     ctr += 1

#     if ctr == 50
#         saved_sol[:, saved_ctr] = solnp1
#         print("saving snap $saved_ctr \n")
#         saved_ctr += 1
#         print("iteration = $ctr \n")
#         print(string(maximum(solnp1), " and ", minimum(solnp1)), "\n")

#         ctr = 0
#     end
#     # print("$ctr \n")
#     # updating RHS values for the next time step
#     RHS_solnm3 = RHS_solnm2
#     RHS_solnm2 = RHS_solnm1
#     RHS_solnm1 = RHS_soln
#     RHS_wave!(RHS_soln, solnp1, sim1, tVec[it+1])
#     # updating solution value for the next time step
#     soln = solnp1

# end



