function y=forward_pass(x)
global model
W1 = model.layer(1).W;
W2 = model.layer(2).W;
b1 = model.layer(1).b;
b2 = model.layer(2).b;

z1 = h(W1*x+b1);
y = W2*z1+b2;
