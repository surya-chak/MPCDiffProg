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
function cartPendNonLin!(dy,y,p,t)

    fVec=p;                 # Control Input


    # ----- Just chosing the chontrol input using floor function probably not differentiable -----
    f=fVec[Int(floor(timeIntegre/dtSnap))+1];
    
    # ----- Deconstruct the state -----
    y1=y[1];                    # θ
    y2=y[2];                    # dθ/dt
    y3=y[3];                    # x
    y4=y[4];                    # dx/dt

    # ----- Assign sin(y1) and cos(y1) -----
    Sy=sin(y1);
    Cy=cos(y1);
    
    # ----- θ equations -----
    dy[1]=y2;
    dy[2]=(f*Cy+(mC+mP)*g*Sy+mP*L*Cy*Sy*y2^2)/
        (mP*L*Cy^2-(mP+mC)*L);

    # ----- X equations -----
    dy[3]=y4;
    dy[4]=(f+mP*L*Sy*y2^2-mP*g*Cy*Sy)/
        (mC+mP*Sy^2);
end



# ----- New systemDynamics integrator using Euler -----
function eulerCartPendNonLin(yInit,p)

    yVec=zeros(4,nSnap);


    # ----- Initialize the problem -----
    yVec[:,1]=yInit;
    
    for iTime=1:(nSnap-1)
        f=p[iTime];                 # Control Input
        # println("forcing $f at time $iTime")        
        # ----- Deconstruct the state -----
        y1=yVec[1,iTime];                    # θ
        y2=yVec[2,iTime];                    # dθ/dt
        y3=yVec[3,iTime];                    # x
        y4=yVec[4,iTime];                    # dx/dt
        # ----- Assign sin(y1) and cos(y1) -----
        Sy=sin(y1);
        Cy=cos(y1);

        # ----- θ equations -----
        dy1=y2;
        dy2=(f*Cy+(mC+mP)*g*Sy+mP*L*Cy*Sy*y2^2)/
            (mP*L*Cy^2-(mP+mC)*L);
        # ----- X equations -----
        dy3=y4;
        dy4=(f+mP*L*Sy*y2^2-mP*g*Cy*Sy)/
            (mC+mP*Sy^2);
        # ----- Calculating next timestep using Euler  -----
        yVec[1,iTime+1]=yVec[1,iTime]+dtSnap*dy1;
        yVec[2,iTime+1]=yVec[2,iTime]+dtSnap*dy2;
        yVec[3,iTime+1]=yVec[3,iTime]+dtSnap*dy3;
        yVec[4,iTime+1]=yVec[4,iTime]+dtSnap*dy4;
    end
    return yVec
end



# =========================================
# Make a animation of results from solution 
# =========================================

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
    yPend=-L*cos(θ);
    # ----- Calculating the line connecting cart to pendulum -----
    xLine=[xCart,xPend];
    yLine=[yCart,yPend];
    ax=subplot(111,aspect=1)
    scatter(xCart,yCart,100,marker="s","k") # Plotting the cart using scatter
    scatter(xPend,yPend,100,"r") # Plotting the pendulum using scatter
    plot(xLine,yLine,"-k",linewidth=2)


    # ----
    xlim(-3,3)
    xlabel(L"x",FontSize=21)
    ylim(-1,1)
    ylabel(L"y",FontSize=21)

    title("Cart animation at t=$time",FontSize=21)
    # ax.set_aspect=1
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

function makeMovie(outFname)
    close("all")
    fig=figure(1,figsize=(10,10));clf();
    setLatex(21);
    nFil=size(solMat,2)
    withfig(fig) do
        myanim = anim.FuncAnimation(fig, plotSnap, frames=nFil, interval=100);
        myanim[:save](outFname, bitrate=-1);
        # extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    end
    
end 

# ====================================================
# Linear system linearized about the vertical position
# ====================================================

# System can be linearized about its fixed points and represented as:
# dy/dt= Ay + Bu
# Where Bu is a control input to be used to stabilize the system 


# function dy = cartPendLin(y,m,M,L,g,d,u)



# end
