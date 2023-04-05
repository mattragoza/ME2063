function d=Dy(x)

global model
W1 = model.layer(1).W;
W2 = model.layer(2).W;
b1 = model.layer(1).b;
b2 = model.layer(2).b;

size(x);  % kl
size(W1); % jk
size(b1); % j1

size(W2); % ij
size(b2); % i1

a1  = W1*x+b1; % jk,kl -> jl
z1p = hp(a1);  % jl

D = size(x, 1);
N = size(x, 2);
d = zeros(size(x)); % kl
for k=1:D
    for l=1:N
        % ij,jl,jk -> kl
        d(k,l) = sum(W2'.*z1p(:,l).*W1(:,k));
    end
end
