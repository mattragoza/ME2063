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
figure;
for i=1:4
    mode = U(:,i);
    subplot(2,2,i);
    pdeplot(...
        model, ...
        'XYData', mode, ...
        'ZData',  mode, ...
        'Contour',  'on', ...
        'ColorMap', 'hot' ...
    );
    title(sprintf("POD mode %d", i));
    view([0 0 1]); axis tight; axis equal;
    xlabel('$x$'); ylabel('$y$'); set(gca, 'Fontsize', 15);
    axis tight;
    drawnow
end

% plot singular values
figure;
semilogy(diag(S(1:50,1:50)));
title('Singular values');
ylabel('value');
xlabel('index');
axis tight;

% low-rank POD basis
p = cumsum(diag(S).^2);
p = p / p(length(p)) * 100;
[m, r] = max(p > 99.9);
U_r = U(:,1:r);
rank = r
power = p(r)

%% Part 2: DEIM

% get QDEIM sensor locations
[Q, R, P] = qr(U_r');
P_r = P(1:r,:);
[m, p] = max(P_r');

c = '#00AAEE';
figure;
for i=1:4
    mode = U(:,i);
    subplot(2,2,i);
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
    title(sprintf("POD mode %d", i));
    view([0 0 1]); axis tight; axis equal;
    xlabel('$x$'); ylabel('$y$'); set(gca, 'Fontsize', 15);
    drawnow
end

% interpolate temperature with QDEIM
T = reshape(T, [Nnode, Ntime, Ns]);
T_true = T(:, end, end);
T_data = T_true(p);
U_data = U_r(p, :);
params = U_data \ T_data;
T_pred = U_r * params;
panelplot(model, T_true, T_pred, "train", "QDEIM");

%% Part 3: Surrogate Modeling Using DEIM

% generate new temperature simulation as test set
theta_new = 120;
xc_new = 0.5*cosd(theta_new) + 0.5;
yc_new = 0.5*sind(theta_new) - 0.25;
[model, FEM_M, FEM_K, FEM_F] = GetFEMMatmodel(xc_new, yc_new, model);
[t, T_new] = SolveFOM(u0, t, xc_new, yc_new);
figure;
pdeplot(model, ...
    'XYData', T_new(:,end), ...
    'ZData',  T_new(:,end), ...
    'Contour', 'on', ...
    'ColorMap', 'hot' ...
);
title('New temperature simulation');
view([0 0 1]); axis equal; axis tight;
xlabel('$x$'); ylabel('$y$'); set(gca,'Fontsize',15);
drawnow

% interpolate test simulation with QDEIM
for t_idx=41:80:201
    T_true = T_new(:, t_idx);
    T_data = T_true(p);
    U_data = U_r(p, :);
    params = U_data \ T_data;
    T_pred = U_r * params;
    panelplot(model, T_true, T_pred, sprintf("test, t=%.2f", t(t_idx)), "QDEIM");
end

%% Part 4: Surrogate Modeling Using Polynomial Basis Function

% train polynomial model
Phi = phi(theta, Ns);
T_true = squeeze(T(:,Ntime,:));

%size(Phi)    % Ntheta x Ns
%size(T_true) %  Nnode x Ns

% T_true = T_hat * Phi';
% T_true' = Phi * T_hat';
% T_hat' = Phi \ T_true';
T_hat = (Phi \ T_true')';

%size(T_hat);  % Nnode x Ntheta

% plot polynomial coefficients
figure;
for i=1:4
    mode = T_hat(:,i);
    subplot(2,2,i);
    pdeplot(...
        model, ...
        'XYData', mode, ...
        'ZData',  mode, ...
        'Contour',  'on', ...
        'ColorMap', 'hot' ...
    );
    title(sprintf("Polynomial coefficients %d", i));
    view([0 0 1]); axis tight; axis equal;
    xlabel('$x$'); ylabel('$y$'); set(gca, 'Fontsize', 15);
    axis tight;
    drawnow
end

% test polynomial model
Phi_pred = phi(120, Ns);
T_pred = T_hat * Phi_pred';
T_true = T_new(:, Ntime);
panelplot(model, T_true, T_pred, "test", "polynomial");

