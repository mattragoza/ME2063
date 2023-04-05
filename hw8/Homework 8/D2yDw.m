function D2yDw=D2yDw(x)
global model

W1 = model.layer(1).W;
W2 = model.layer(2).W;
b1 = model.layer(1).b;
b2 = model.layer(2).b;

D = size(x, 1);
N = size(x, 2);
m = length(b1);

size(x);  % kl
size(W1); % jk
size(b1); % j1

size(W2); % ij
size(b2); % i1

a1   = W1*x+b1; % jk,kl-> jl
z1pp = hpp(a1); % jl
z1p3 = hp3(a1); % jl
W12  = W1.*W1;  % jk

% second layer bias
G_b2 = zeros(1, N); % il

% second layer weights
G_W2 = zeros(m, N); % jl
for j=1:m
    for l=1:N
        % jl,jk -> jl
        G_W2(j,l) = sum(z1pp(j,l).*W12(j,:));
    end
end

% first layer bias
G_b1 = zeros(m, N); % jl
for j=1:m
    for l=1:N
        % jl,jk -> jl
        G_b1(j,l) = sum(W2(:,j).*z1p3(j,l).*W12(j,:));
    end
end

% first layer weights
G_W1 = []; % kjl
for k=1:D
    G_W1_k = zeros(m, N); % jl
    for j=1:m
        for l=1:N

            % jl,jk->kjl
            term1 = sum(2*W2(:,j).*z1pp(j,l).*W1(j,k), "all");
            
            % jl,jk,kl->kjl
            term2 = sum(W2(:,j).*z1p3(j,l).*W12(j,:).*model.x(k,l), "all");

            G_W1_k(j,l) = term1 + term2;
        end
    end
    G_W1 = [G_W1; G_W1_k]; % stack vertically
end

size(G_W1); % 80 x 300
size(G_b1); % 40 x 300
size(G_W2); % 40 x 300
size(G_b2); %  1 x 300

D2yDw = [G_W1;G_b1;G_W2;G_b2];
