function G=DLW(W)
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
z1p3 = hp3(a1); % jl
W1_2 = W1 .^ 2; % jn

G_b2 = zeros(1, 1);

G_W2 = zeros(M, 1);
for n=1:D
    % jn,jl->jl ->ij
    G_W2 = G_W2 + sum(W1_2(:,n).*z1pp, 2);
end

G_b1 = zeros(M, 1);
for n=1:D
    % ji,jn,jl->ji
    G_b1 = G_b1 + sum(W2'.*W1_2(:,n).*z1p3, 2);
end

G_W1 = zeros(M, D);
for k=1:D
    %term1 = z1pp*ones(N, 1); % jl,lk->jk
    %term1 = 2*sum(W1(:,k).*z1pp, 2); % jk,jl->jk
    term1 = 2*sum(W2'.*W1(:,k).*z1pp, 2); % ji,jk,jl->jk

    %term2 = sum(W1.*(z1p3*x(k,:)'), 2); % jn,jl,lk->jk
    %term2 = sum((W1.^2).*(z1p3*x(k,:)'), 2); % jn,jl,lk->jk
    term2 = sum(W2'.*(W1.^2).*(z1p3*x(k,:)'), 2); % ji,jn,jl,lk->jk

    G_W1(:,k) = term1 + term2;
end
G_W1 = reshape(G_W1, M*D, 1);

G = [G_W1;G_b1;G_W2;G_b2];
