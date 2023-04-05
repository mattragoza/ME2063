function G = Gradloss
global model 
W1 = model.layer(1).W;
W2 = model.layer(2).W;
b1 = model.layer(1).b;
b2 = model.layer(2).b;

G = 0;
if model.flagd > 0
    y_pred = forward_pass(model.x);
    Gy = DyDw(model.x);
    Gd = Gy*(y_pred-model.y)'/model.Nd;
    G = G + model.flagd * Gd;
end

%% ----------------- Adding Physics-informed gradient ---------------

if model.flagm > 0
    L = D2y(model.xm); % Laplacian
    pde_res = model.k * L + q(model.xm')';
    GL = D2yDw(model.xm);
    Gm = model.k*GL*pde_res'/model.Nm;
    G = G + model.flagm * Gm;
end
