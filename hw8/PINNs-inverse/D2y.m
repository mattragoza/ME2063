function H=D2y(x)
global model

W1 = model.layer(1).W;
W2 = model.layer(2).W;
b1 = model.layer(1).b;
b2 = model.layer(2).b;

a1   = W1*x+b1; % jk,kl -> jl
z1pp = hpp(a1); % jl

D = size(x, 1);
N = size(x, 2);

H = zeros(D, D, N); % Hessian
for m=1:D
    % ij,(jk,jm->jk)',jl->kml
    H(:,m,:) = W2.*(W1.*W1(:,m))'*z1pp;
end
