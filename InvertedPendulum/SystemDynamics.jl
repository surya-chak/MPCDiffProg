using JLD2
using DifferentialEquations
using PyCall
using PyPlot
using Trapz
using Prettify
using IJulia
# ==========================
# Critical System parameters
# ==========================

global const mP=0.23;           # Pendulum mass as 0.23 kg
global const mC=2.4*0.25;            # Cart mass as 2.4 kg
global const L=0.3;            # 30 cms for now allegedly the very limit of what humans can do 
global const g=9.8;            # g=9.8m/s^2


timeIntegre=5;
dtSnap=0.05;
# ================
# Nonlinear system
# ================


# ----- General Description -----
# A dynamical system consisting of an inverted pendulum on a cart.
# The state is fully determined as y=[θ , dθ/dt, x, dx/dt]
# Full nonlinear state can be determined using the Eular Lagrange equations:
# dy1/dt=y2
# dy2/dt=(f*cos(y1)+(mC+mP)g*sin(y1)+mP*L*cos(y1)*sin(y1)*y2^2)/(mP*L*cos(y1)^2-(mP+mC)*L)
# dy3/dt=y4
# dy4/dt=(f+mP*L*sin(y1)y2^2-mPgcos(y1)sin(y1))/(mC+mP*sin(y1)^2)


# ----- RHS of the NonLinearSystem -----
function cartPendNonLin!(dy,y,params,t)

    f=params;                 # Control Input
    # ----- Deconstruct the state -----
    y1=y[1];                    # θ
    y2=y[2];                    # dθ/dt
    y3=y[3];                    # x
    y4=y[4];                    # dx/dt

    # ----- Assign sin(y1) and cos(y1) -----
    Sy=sin(y1);
    Cy=cos(y1);
    println(Cy)
    
    # ----- θ equations -----
    dy[1]=y2;
    dy[2]=(f*Cy+(mC+mP)*g*Sy+mP*L*Cy*Sy*y2^2)/
        (mP*L*Cy^2-(mP+mC)*L);

    # ----- X equations -----
    dy[3]=y4;
    dy[4]=(f+mP*L*Sy*y2^2-mP*g*Cy*Sy)/
        (mC+mP*Sy^2);
    
end

# ====================================
# Example solve of a noControl problem
# ====================================
yInit=[π-0.2*π,0,0,0]; #init conditions
tSpan=(0.0,timeIntegre);
p=0.0;                          # No control case for now
tVec=Vector(0:dtSnap:timeIntegre);
probNoControl = ODEProblem(cartPendNonLin!, yInit, tSpan, p);

println("Solving inverted pendulum on a cart withotu control...")
sol = Array(solve(probNoControl, Tsit5(),saveat=0.0:dtSnap:timeIntegre));
solMat=sol;

# =========================================
# Make a animation of results from solution 
# =========================================
close("all")
fig=figure(1,figsize=(10,10));clf();
setLatex(21);

function plotSnap(i)
    println(i+1);
    clf();
    time=tVec[i+1];
    θ=solMat[1,i+1];

    # ----- Calculating the location of the cart -----
    xCart=solMat[3,i+1];
    yCart=0;

    # ----- Calclating the locaiton of the Pendulum -----
    xPend=xCart+L*sin(θ);
    yPend=L*cos(θ);
    # ----- Calculating the line connecting cart to pendulum -----
    xLine=[xCart,xPend];
    yLine=[yCart,yPend];

    scatter(xCart,yCart,100,"k") # Plotting the cart using scatter
    scatter(xPend,yPend,100,"r") # Plotting the pendulum using scatter
    plot(xLine,yLine,"-k",linewidth=2)

    # ----
    xlim(-10,10)
    xlabel(L"x",FontSize=21)
    ylim(-1,1)
    ylabel(L"y",FontSize=21)

    title("Cart animation at t=$time",FontSize=21)

    # ax.spines["right"].set_visible(false);
    # ax.spines["top"].set_visible(false);
    # ax.spines["left"].set_linewidth(3);
    # ax.spines["bottom"].set_linewidth(3);
    # ax.minorticks_on()
    # ax.tick_params(axis="both",which="major",length=8,width=3)
    # ax.tick_params(axis="both",which="minor",length=5,width=2)
    # ----

    tight_layout()    
end


nFil=size(solMat,2)
@pyimport matplotlib.animation as anim
withfig(fig) do
    myanim = anim.FuncAnimation(fig, plotSnap, frames=nFil, interval=nFil)
    myanim[:save]("invertedPendNoControl.mp4", bitrate=-1)
    # extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
end




# ====================================================
# Linear system linearized about the vertical position
# ====================================================

# System can be linearized about its fixed points and represented as:
# dy/dt= Ay + Bu
# Where Bu is a control input to be used to stabilize the system 


function dy = cartPendLin(y,m,M,L,g,d,u)



end
