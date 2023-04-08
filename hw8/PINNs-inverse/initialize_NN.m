function model=initialize_NN(M,D)
model.W1 = randn(M,D);
model.b1 = randn(M,1);
model.W2 = randn(1,M);
model.b2 = randn(1,1);
model.kp = randn(1,1);

model.layers = [D M 1];

n = 1;
for i=1:length(model.layers)-1
    m1 = model.layers(i);
    m2 = model.layers(i+1);
    
    model.layer(i).W = randn(m2,m1);
    model.layer(i).b = randn(m2,1);
    model.layer(i).iw = [n n+m1*m2-1]; n=n+m1*m2;
    model.layer(i).ib = [n n+m2-1]; n=n+m2;
    model.layer(i).ws = [m2 m1];
    model.layer(i).bs = [m2 1]; 
end
model.np = n;
model.D = D;
