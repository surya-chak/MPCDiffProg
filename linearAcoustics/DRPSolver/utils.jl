module GeneralVariables

using PyPlot

dState = 0.0;

struct Grid
    il :: Int
    jl :: Int
    dx :: Float64
    dy :: Float64
    x :: Array{Float64,2}
    y :: Array{Float64,2}
    # ----- Constructor for the structure -----
    function Grid(x::Matrix{Float64},y::Matrix{Float64})
        il=size(x,1);
        jl=size(x,2);
        dx=x[2,1]-x[1,1];
        dy=y[1,2]-y[1,1];
        new(il,jl,dx,dy,x,y);
    end
end

export Grid

struct Simulation
    # ----- Details of the timeline -----
    dt :: Float64
    tVec :: Array{Float64,1}
    # ----- Details of the grid -----
    grid :: Grid
    filtVec::Array{Float64,2}

    # ----- Constructor for the structure -----
    function Simulation(tVec,x::Matrix{Float64},y::Matrix{Float64},filtVec)
        grid=Grid(x,y);
        dt=tVec[2]-tVec[1];
        soln = zeros(grid.il*grid.jl*2)
        new(dt,
            tVec,
            grid,
            filtVec);
    end
end

export Simulation


function printLimits(sim::Simulation)
    print(string(sim.grid.il," ",sim.grid.jl))
end


export printLimits




# export c, L, kh, freq           # Acoustic propagation parameters
c=1.0;




function RHS_wave!(dState::Array{Float64,1}, state::Array{Float64,1}, sim::Simulation, t::Float64)
    # println(t)
    state_new=reshape(state,(sim.grid.il,sim.grid.jl,2));

    v1=state_new[:,:,1] .* sim.filtVec;
    v2=state_new[:,:,2] .* sim.filtVec;
    # out_val=zeros(sim.grid.il*sim.grid.jl*2,1);

    v1_x=uniGrad(v1,1,sim);
    v1_xx=uniGrad(v1_x,1,sim);
    
    v1_y=uniGrad(v1,2,sim);
    v1_yy=uniGrad(v1_y,2,sim);

    dv1dt=v2;
    dv2dt=c^2*(v1_xx+v1_yy)+source(sim,t);

    dState[:]=vcat(dv1dt[:],dv2dt[:]);
end



function source(sim::Simulation,t::Float64)
    # source_val=10*sin(freq*t)*exp(-10*(x.^2+y.^2)); % monopole source
    freq=0.3;                       #Frequency of the field being simulated 
    freq=2*pi*freq;

    # Jet Mach number
    Mj=1.3;
    Ma=Mj*0.6;                              # convective Mach number
    kh=freq/(Ma)^2;
    L=5/kh;

    # source_val=sin.(freq*t .- kh.*sim.grid.x).*
    #     exp.(-((sim.grid.x .- 2*L)./L).^2).*
    #     exp.(-2 .*(sim.grid.y).^2);
    source_val=sin.(freq*t).*
        exp.(-((sim.grid.x .- 10.0)).^2).*
        exp.(-2 .*(sim.grid.y).^2);

    return source_val
end

export source

function uniGrad(fun::Array{Float64,2},dir::Int,sim::Simulation)

    a1=0.79926643;
    am1=-a1;

    a2=-0.18941314;
    am2=-a2;

    a3=0.02651995;
    am3=-a3;

    fun_der=zeros(size(fun));

    if dir == 1                 # X derivative
        fun_der[4:end-3,:]=a1*fun[5:end-2,:]+am1*fun[3:end-4,:];
        fun_der[4:end-3,:]=fun_der[4:end-3,:]+a2*fun[6:end-1,:]+am2*fun[2:end-5,:];
        fun_der[4:end-3,:]=fun_der[4:end-3,:]+a3*fun[7:end,:]+am3*fun[1:end-6,:];
        fun_der=fun_der./(sim.grid.dx);

    elseif dir == 2             # Y derivative
        fun_der[:,4:end-3]=a1*fun[:,5:end-2]+am1*fun[:,3:end-4];
        fun_der[:,4:end-3]=fun_der[:,4:end-3]+a2*fun[:,6:end-1]+am2*fun[:,2:end-5];
        fun_der[:,4:end-3]=fun_der[:,4:end-3]+a3*fun[:,7:end]+am3*fun[:,1:end-6];
        fun_der=fun_der./(sim.grid.dy);
    end

    return fun_der
end



function plotResult(i::Int,sim::Simulation,ax)
    ax.contourf(sim.grid.x,sim.grid.x,sim.savedSol[i],cmap=ColorMap("jet"))
    # ----
    xlim(0,10)
    ylim(-5,5)
    clim(-1e-5,1e-5)
    xlabel(L"x/D_e",FontSize=21)
    ylabel(L"y/D_e",FontSize=21)
    title(L"snapshot number $$i",FontSize=21)
    ax.minorticks_on()
    ax.tick_params(axis="both",which="major",length=8,width=3)
    ax.tick_params(axis="both",which="minor",length=5,width=2)
    # ----



    

end


export RHS_wave!, plotResult, uniGrad, source, dState



end                             # end of module
