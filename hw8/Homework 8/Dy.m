function d=Dy(x)

global model
W1 = model.layer(1).W;
W2 = model.layer(2).W;
b1 = model.layer(1).b;
b2 = model.layer(2).b;

a1  = W1*x+b1;   % jk,kl->jl
z1p = hp(a1);    % jl
d = W2.*W1'*z1p; % ij,kj,jl->kl
