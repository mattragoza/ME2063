function G=D2yDw(x)
global model

W1 = model.layer(1).W;
W2 = model.layer(2).W;
b1 = model.layer(1).b;
b2 = model.layer(2).b;

D = size(x, 1);
N = size(x, 2);
M = length(b1);

a1   = W1*x+b1; % jk,kl -> jl
z1pp = hpp(a1); % jl
z1p3 = hp3(a1); % jl
W1_2 = W1 .^ 2; % jn

% second layer bias
G_b2 = zeros(1, N); % il

% second layer weights
G_W2 = zeros(M, N); % jl
for n=1:D
    % jn,jl->jl
    G_W2 = G_W2 + W1_2(:,n).*z1pp;
end

% first layer bias
G_b1 = zeros(M, N); % jl
for n=1:D
    % ji,jn,jl->jl
    G_b1 = G_b1 + W2'.*W1_2(:,n).*z1p3;
end

% first layer weights
G_W1 = zeros(M, D, N); % jkl
for k=1:D
    term1 = 2*W2'.*W1(:,k).*z1pp; % ji,jk,jl->jkl

    term2 = 0;
    for n=1:D % ji,jn,jl,kl->jkl
        term2 = term2 + W2'.*W1_2(:,n).*z1p3.*x(k,:);
    end
    G_W1(:,k,:) = term1 + term2;
end
G_W1 = reshape(G_W1, M*D, N);

G = [G_W1;G_b1;G_W2;G_b2];
