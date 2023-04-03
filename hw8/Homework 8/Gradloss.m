function G = Gradloss
global model 
W1 = model.layer(1).W;
W2 = model.layer(2).W;
b1 = model.layer(1).b;
b2 = model.layer(2).b;

y = forward_pass(model.x);
Gp = DyDw(model.x);
G = model.flagd*Gp*(y-model.y)'/model.Nd;

%% ----------------- Adding Physics-informed gradient ---------------
    




