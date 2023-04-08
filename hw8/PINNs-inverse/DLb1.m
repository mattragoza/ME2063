function G=DLb1(b1)
global model

x = model.x;
D = size(x, 1);

W1 = model.layer(1).W;
W2 = model.layer(2).W;
%b1 = model.layer(1).b;
b2 = model.layer(2).b;
M = size(W1, 1);

a1   = W1*x+b1;  % jk,kl -> jl
z1p3 = hp3(a1);  % jl
W1_2 = W1 .^ 2;  % jn

G = zeros(M, 1); % ji
for n=1:D
    % ji,jn,jl->ji
    G = G + W2' .* W1_2(:,n) .* sum(z1p3, 2);
end
size(G)
