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

d = zeros(size(x)); % kl
for j=1:length(b1)
    % ij,jl,jk -> kl
    d = d + W2(:,j).*z1p(j,:).*W1(j,:)';
end
