function L=LW1(W1)
global model

x = model.x;
%W1 = model.layer(1).W;
W2 = model.layer(2).W;
b1 = model.layer(1).b;
b2 = model.layer(2).b;

a1   = W1*x+b1; % jk,kl -> jl
z1pp = hpp(a1); % jl

D = size(x, 1);
N = size(x, 2);
M = size(b1, 1);

H = zeros(D, D, N); % Hessian
for m=1:D
    % ij,(jk,jm->jk)',jl->kml
    H(:,m,:) = W2.*(W1.*W1(:,m))'*z1pp;
end

L = squeeze(H(1,1,:) + H(2,2,:))'; % Laplacian

% DEBUGGING
%z1pp = reshape(1:M*N, M, N);
%W1 = reshape(1:M*D, M, D);

% kj,jl->kl
%L = W1'*z1pp;
%L = (W1'.^2)*z1pp;
%L = W2.*(W1'.^2)*z1pp;
