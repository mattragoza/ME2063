function L=LW(W)
global model

x = model.x;

D = size(x, 1);
N = size(x, 2);
M = size(model.layer(1).b, 1);

W1 = reshape(W(1:M*D), M, D);
b1 = reshape(W(M*D+1:M*D+M), M, 1);
W2 = reshape(W(M*D+M+1:M*D+2*M), 1, M);
b2 = reshape(W(M*D+2*M+1), 1, 1);

a1   = W1*x+b1; % jk,kl -> jl
z1pp = hpp(a1); % jl

H = zeros(D, D, N); % Hessian
for m=1:D
    % ij,(jk,jm->jk)',jl->kml
    H(:,m,:) = W2.*(W1.*W1(:,m))'*z1pp;
end

L = squeeze(H(1,1,:) + H(2,2,:))'; % Laplacian
