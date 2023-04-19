function [t,T]=SolveFOM(u0,t,xc,yc)
global model FEM_M FEM_K FEM_F 
odesolve_tol = 1e-6;
options = odeset('Mass',FEM_M ,'AbsTol',odesolve_tol ,'RelTol',odesolve_tol ,'Stats','on');
disp('Solving FOM');
tic
[t , T] = ode45(@FOM_rhs,t,u0',options );
toc
T = T';