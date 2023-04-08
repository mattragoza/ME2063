function G=DLW1(W1)
global model

x = model.x;

%W1 = model.layer(1).W;
W2 = model.layer(2).W;
b1 = model.layer(1).b;
b2 = model.layer(2).b;

a1   = W1*x+b1; % jk,kl -> jl
z1pp = hpp(a1); % jl
z1p3 = hp3(a1); % jl
W1_2 = W1 .^ 2; % jn

D = size(x, 1);
N = size(x, 2);
M = size(b1, 1);

% DEBUGGING
%z1pp = reshape(1:M*N, M, N);
%W1 = reshape(1:M*D, M, D);

G = zeros(M, D);
for k=1:D
    %term1 = z1pp*ones(N, 1); % jl,lk->jk
    %term1 = 2*sum(W1(:,k).*z1pp, 2); % jk,jl->jk
    term1 = 2*sum(W2'.*W1(:,k).*z1pp, 2); % ji,jk,jl->jk

    %term2 = sum(W1.*(z1p3*x(k,:)'), 2); % jn,jl,lk->jk
    %term2 = sum((W1.^2).*(z1p3*x(k,:)'), 2); % jn,jl,lk->jk
    term2 = sum(W2'.*(W1.^2).*(z1p3*x(k,:)'), 2); % ji,jn,jl,lk->jk

    G(:,k) = term1 + term2;
end
