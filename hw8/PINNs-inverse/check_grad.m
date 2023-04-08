function check_grad
global model

W1 = model.layer(1).W; % jk
W2 = model.layer(2).W; % ij
b1 = model.layer(1).b;
b2 = model.layer(2).b;

M = size(W1, 1); % hidden units (j)
D = size(W1, 2); % input units  (k)
N = 1;           % batch index  (l)

x0 = randn(D, N); % kl
e1 = ones(M, N);  % jl
e2 = ones(1, M);  % ij

h   = @(x) tanh(x);
hp  = @(x) 1 - h(x).^2;
hpp = @(x) -2*tanh(x).*hp(x);

%loss = @(x) sum(W1*x+b1); % jk,kl->jl
%grad = @(x) W1'*e1;       % kj,jl->kl

%loss = @(x) sum(h(W1*x+b1), "all"); % jk,kl->jl
%grad = @(x) W1'*hp(W1*x+b1);        % kj,jl->kl

%loss = @(x) sum(W2*h(W1*x+b1), "all"); % ij,jk,kl->il
%grad = @(x) W2.*W1'*hp(W1*x+b1);       % ij,kj,jl->kl

%loss = @(x) sum(forward_pass(x), "all");
%grad = @(x) Dy(x);

%loss = @(x) sum(Dy(x), "all");
%grad = @(x) sum(D2y(x), 1)';

%x0 = randn(1, M);
%loss = @(W2) sum(LW2(W2), "all");
%grad = @(W2) DLW2(W2);

%x0 = randn(M, 1);
%loss = @(b1) sum(Lb1(b1), "all");
%grad = @(b1) DLb1(b1);

%x0 = randn(M, D);
%loss = @(W1) sum(LW1(W1), "all");
%grad = @(W1) DLW1(W1);

%x0 = randn(1+M+M+D*M, 1);
%loss = @(W) sum(LW(W), "all");
%grad = @(W) DLW(W);

x0 = randn(1,1).^2;
loss = @loss_k;
grad = @grad_k;

% NUMERICAL GRADIENT CHECK
relative_error = zeros(size(x0));
step_size = 1e-6;
tolerance = 1e-5;
for i=1:length(x0)
    u = zeros(size(x0));
    u(i) = 1;

    G_ana = sum(grad(x0).*u, "all");

    L_num1 = loss(x0 + step_size*u);
    L_num2 = loss(x0 - step_size*u);
    G_num = (L_num1 - L_num2) / (2*step_size);

    relative_error(i, 1) = abs(G_ana - G_num) / (abs(G_num) + 1e-12);
end
mean_relative_error = mean(relative_error, "all")

if mean_relative_error < tolerance
    result = "Gradient check passed"
else
    result = "Gradient check failed"
end
