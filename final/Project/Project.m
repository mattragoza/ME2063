clear,clc,close all;
set(0,'defaulttextinterpreter','latex')
global model FEM_M FEM_K FEM_F U Kr fr EID;
load('theta')
model = createpde(1);
%% Geometry
R1 = [3,4,-1,1,1,-1,0.5,0.5,-0.75,-0.75]';
C1 = [1,0.5,-0.25,0.25]';
C1 = [C1;zeros(length(R1) - length(C1),1)];
gm = [R1,C1];
sf = 'R1-C1';
ns = char('R1','C1');
ns = ns';
g = decsg(gm,sf,ns);
geometryFromEdges(model,g);
pdegplot(model,'EdgeLabels','on')
axis equal
xlim([-1.1,1.1])
%% Mesh
msh = generateMesh(model,'GeometricOrder','quadratic','Hmax',.05);
Nnode = size(msh.Nodes,2);
set(gca,'Fontsize',15);
xlabel('$x$')
ylabel('$y$')
drawnow
[P,E,~] = meshToPet( msh );
EID = findNodes(msh,'region','Edge',[1:4]);
%% ------ Time for Simulation Setup ----------
Ntime = 201;
startTime = 0;
endTime = .05;
t = linspace(startTime,endTime,Ntime);

%% This evaluates Initial Condition
u0 = IC(P(1,:), P(2,:));
%% Generate Data by solving the FEM
F = [];
Ns = 7;
% theta = legpts(Ns,[0,180]);
T = zeros(Nnode,Ntime,Ns);
xc = zeros(Ns,1);
yc = zeros(Ns,1);
for n=1:Ns
    xc(n) = 0.5*cosd(theta(n))+0.5;
    yc(n) = 0.5*sind(theta(n))-0.25;
    [model,FEM_M,FEM_K,FEM_F] = GetFEMMatmodel(xc(n),yc(n),model);
    [t , Tn] = SolveFOM(u0,t,xc(n),yc(n));
    figure; pdeplot(model ,'XYData',Tn(:,end),'ZData',Tn(:,end),'Contour','on','ColorMap','jet'); view([0 0 1]); axis equal;axis tight;xlabel('$x$')
    ylabel('$y$'); set(gca,'Fontsize',15); drawnow
    T(:,:,n) = Tn;
end
%% Part 1: POD modes






%% Part 2: DEIM

% This is a sample code that you can use for plotting the DEIM points,
% where P is the vector DEIM indexes (selected points). Here random indexes
% are used. Also , instead of Tn, POD modes must be plotted.
figure
P = randperm(Nnode,10);
figure; pdeplot(model ,'XYData',Tn(:,end),'ZData',Tn(:,end),'Contour','on','ColorMap','jet'); view([0 0 1]); axis equal;axis tight; hold on
plot3(msh.Nodes(1,P),msh.Nodes(2,P),300*ones(10,1),'o','Color','k','MarkerFaceColor','k')
xlabel('$x$')
ylabel('$y$')
set(gca,'Fontsize',15);
%% Part 3: Surrogate Modeling Using DEIM







%% Part 4: Surrogate Modeling Using Polynomial Basis Function
Phi = phi(theta, Ns);
