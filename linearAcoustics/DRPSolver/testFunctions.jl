include("./utils.jl")
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

dt=1e-2;
# dt=5e-3;
nSnap=5000;
tFin=(nn-1)*dt;
tVec=LinRange(0,tFin,nn);


