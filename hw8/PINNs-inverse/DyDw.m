function DyDW=DyDw(x)
global model 
W1 = model.layer(1).W;
W2 = model.layer(2).W;
b1 = model.layer(1).b;
b2 = model.layer(2).b;

D = model.D;

m = size(b1,1);
N = length(x);

em = ones(m,1);
eN = ones(1,N);

a1 = W1*x+b1;
G_W1 = [];
for i=1:D
    G_W1 = [G_W1; W2'.*hp(a1).*x(i,:)];
end

a1 = W1*x+b1;
z1 = h(a1);
G_b1 = W2'.*hp(a1);
G_W2 = z1;
G_b2 = 1*eN;



DyDW = [G_W1;G_b1;G_W2;G_b2];
