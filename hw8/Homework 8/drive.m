clc
close all
clear all

rng('default')
LW = 'LineWidth';
set(0,'defaulttextinterpreter','latex')
R     = [216, 82,  24 ]/255;
global model counter L 

set(gcf, 'Position', [100, 100, 1200, 300]);
colororder([[0.3,0.3,1]; [0,0.7,0]]);

m = 40; % num hidden units (default 40)
D = 2;
counter = 0;
model=initialize_NN(m,D);



model.Nd  = 300;                    %number of training points (data)
model.Ndb = 290;                    %number of training points on the boundary(data) and it is a subset of total training data points. Therefore,  model.Ndb< model.Nd.
model.Nm = 1000;                    %number of training points (model)
model.Ntest = 75;                   %number of test points (data)
noise = 2;                          %Noise level of the data measuremets 
model.flagm = 1;                    %1:includes model in training; 0 excludes the model
model.flagd = 1;                    %1:includes data in training; 0 excludes the data
model.k = 0.1;
%% ------------- Truth ---------------
N1 = 101;
N2 = 101;
L = 1;
x1 = linspace(0,L,N1); dx1 = x1(2) -x1(1);
x2 = linspace(0,L,N2); dx2 = x2(2) -x2(1);

d2x1 = diag(1/dx1^2*ones(1,N1-1),-1) + diag(-2/dx1^2*ones(1,N1)) + diag(1/dx1^2*ones(1,N1-1),1);
d2x2 = diag(1/dx2^2*ones(1,N2-1),-1) + diag(-2/dx2^2*ones(1,N2)) + diag(1/dx2^2*ones(1,N2-1),1);

D2x1 = sparse(kron(d2x1,eye(N1)));
D2x2 = sparse(kron(eye(N2),d2x2));

[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];

Lap = D2x1 + D2x2;
BDRY.left  = find(X1==0);
BDRY.right = find(X1==L);
BDRY.bot   = find(X2==0);
BDRY.top   = find(X2==L);

BDRY.I= [BDRY.left; BDRY.right; BDRY.bot; BDRY.top];
Lap(BDRY.I,:) = 0; Lap(BDRY.I,BDRY.I) = eye(length(BDRY.I));
Q = -q([X1(:) X2(:)]);
r = Q(:); r(BDRY.I)=0; 
T = Lap\r/model.k; 
T = reshape(T,N2,N1);
%surf(X1,X2,T); colormap(jet); hold on
%% ----------- Training Data ------------
IB = randperm(length(BDRY.I),model.Ndb)'; IB=BDRY.I(IB);
I = [IB;randperm(N1*N2,model.Nd-model.Ndb+model.Ntest)'];
I_train = I(1:model.Nd);

model.x = [X1(I_train), X2(I_train)]';
model.y = T(I_train)'+noise*randn(1,model.Nd);
model.xstar = X';
model.xstar_size= [N1 N2];
model.ystar_t = T;

%% ----------- Test Data ------------
I_test  = I(model.Nd+1:model.Nd+model.Ntest);
model.xtest = [X1(I_test), X2(I_test)]';
model.ytest = T(I_test)';

model.iteration   = [];
model.train_error = [];
model.test_error  = [];

%% ----------- Training Data: Model ------------ 
model.xm = [L*rand(model.Nm,1) L*rand(model.Nm,1)]';

w = randn(model.np,1);
[l,g]=loss(w);

check_grad;

options = optimoptions(...
    'fminunc', ...
    'GradObj', 'on', ...
    'Display', 'iter', ...
    'Algorithm', 'trust-region',...
    'Diagnostics', 'on', ...
    'MaxIterations', 2000, ...
    'FunctionTolerance', 1e-8, ...
    'DerivativeCheck', 'off' ...
);
model.w = fminunc(@loss,w,options);
