function [loss, grad]=my_func(x)
global model

D = size(model.x, 1);
N = size(model.x, 2);
m = length(model.layer(1).b);

W1 = model.layer(1).W;
W2 = model.layer(2).W;
b1 = model.layer(1).b;
b2 = model.layer(2).b;

size(W1); % jk (m, D)
size(b1); % j1 (m, 1)

size(W2); % ij (1, m)
size(b2); % i1 (1, 1)

% first derivative
%y = forward_pass(x);
%loss = sum(y.^2, "all");
%grad = 2*y.*Dy(x); % il,kl -> kl

% second derivative
%loss = sum(Dy(x), "all"); % kl
%grad = sum(D2y(x), 1); % kml

% second layer bias
%model.layer(2).b = x;
%s = D2y(model.x); % kml
%loss = sum(s(1,1,:) + s(2,2,:), "all");
%grad = zeros(1, 1);

% second layer weights

a1   = W1*model.x+b1; % jk,kl-> jl
z1pp = hpp(a1); % jl
z1p3 = hp3(a1); % jl
W12  = W1.*W1;  % jk

model.layer(2).W = x;
s = D2y(model.x); % kml
loss = sum(s(1,1,:) + s(2,2,:), "all");

grad = zeros(m, N); % jl
for j=1:m
    for l=1:N
        % jl,jk,jk->jl
        grad(j,l) = sum(z1pp(j,l).*W12(j,:));
    end
end
grad = sum(grad, 2)';
