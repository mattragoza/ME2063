function s=D2y(x)
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

a1   = W1*x+b1; % jk,kl -> jl
z1pp = hpp(a1); % jl
W12  = W1.*W1;  % jk

D = size(x, 1);
N = size(x, 2);
s = zeros(D, D, N); % kml
for k=1:D
    for m=1:D
        for l=1:N
            % ij,jl,jk,jm -> kml
            s(k,m,l) = sum(W2'.*z1pp(:,l).*W1(:,k).*W1(:,m));
        end
    end
end

% convert Hessian to Laplacian
s = reshape(s(1,1,:) + s(2,2,:), 1, N);
