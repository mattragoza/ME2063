function G=DLW2(W2)
global model

x = model.x;
D = size(x, 1);

W1 = model.layer(1).W;
%W2 = model.layer(2).W;
b1 = model.layer(1).b;
b2 = model.layer(2).b;
M = size(W1, 1);

a1   = W1*x+b1; % jk,kl -> jl
z1pp = hpp(a1); % jl
W1_2 = W1 .^ 2; % jn

G = zeros(1, M);
for n=1:D
    % jn,jl->jl ->ij
    G = G + sum(W1_2(:,n).*z1pp, 2)';
end
