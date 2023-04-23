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
title('Geometry');
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
    figure;
    pdeplot(model, ...
        'XYData', Tn(:,end), ...
        'ZData',  Tn(:,end), ...
        'Contour', 'on', ...
        'ColorMap', 'hot' ...
    );
    title('Temperature simulation');
    view([0 0 1]); axis equal; axis tight;
    xlabel('$x$'); ylabel('$y$'); set(gca,'Fontsize',15);
    drawnow
    T(:,:,n) = Tn;
end
%% Part 1: POD modes
Nnode % 4274
Ntime %  201
Ns    %    7
T = reshape(T, Nnode, Ntime*Ns);
[U, S, V_T] = svd(T, 0);

% plot first 4 POD modes
for r=1:4
    mode = U(:,r);
    figure;
    pdeplot(...
        model, ...
        'XYData', mode, ...
        'ZData',  mode, ...
        'Contour',  'on', ...
        'ColorMap', 'hot' ...
    );
    title('POD modes');
    view([0 0 1]); axis tight; axis equal;
    xlabel('$x$'); ylabel('$y$'); set(gca, 'Fontsize', 15);
    drawnow
end

% plot singular values
figure;
semilogy(diag(S));
title('Singular values');
ylabel('value');
xlabel('index');

% low-rank POD basis
p = cumsum(diag(S).^2);
p = p / p(length(p)) * 100;
[m, r] = max(p > 99.9);
U_r = U(:,1:r);
rank = r
power = p(r)

%% Part 2: DEIM

[Q, R, P] = qr(U_r');
P_r = P(1:r,:);
[m, p] = max(P_r');

c = '#00AAEE';
for i=1:4
    mode = U(:,i);
    figure;
    pdeplot(...
        model, ...
        'XYData', mode, ...
        'ZData',  mode, ...
        'Contour',  'on', ...
        'ColorMap', 'hot' ...
    );
    hold on;
    plot3( ...
        msh.Nodes(1, p), ...
        msh.Nodes(2, p), ...
        300*ones(r, 1), ...
        'o', ...
        'Color', c, ...
        'MarkerFaceColor', c ...
    );
    title('QDEIM sensor locations');
    view([0 0 1]); axis tight; axis equal;
    xlabel('$x$'); ylabel('$y$'); set(gca, 'Fontsize', 15);
    drawnow
end
%%
T = reshape(T, [Nnode, Ntime, Ns]);
T_true = T(:, Ntime, Ns);
T_data = T_true(p);
U_data = U_r(p, :);
theta = U_data \ T_data;
T_pred = U_r * theta;

figure;
subplot(3,1,1);
pdeplot(...
    model, ...
    'XYData', T_pred, ...
    'ZData',  T_pred, ...
    'Contour',  'on', ...
    'ColorMap', 'hot' ...
);
view([0 0 1]); axis tight;
title('Interpolated temperature');

subplot(3,1,2);
pdeplot(...
    model, ...
    'XYData', T_true - T_pred, ...
    'ZData',  T_true - T_pred, ...
    'Contour',  'on', ...
    'ColorMap', 'hot' ...
);
view([0 0 1]); axis tight;
title('Error');

subplot(3,1,3);
pdeplot(...
    model, ...
    'XYData', T_true, ...
    'ZData',  T_true, ...
    'Contour',  'on', ...
    'ColorMap', 'hot' ...
);
view([0 0 1]); axis tight;
title('True temperature');

%% Part 3: Surrogate Modeling Using DEIM







%% Part 4: Surrogate Modeling Using Polynomial Basis Function
Phi = phi(theta, Ns);
