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
    Gd_T = Gy*(y_pred-model.y)'/model.Nd;
    Gd_k = zeros(1, 1);
    Gd = [Gd_T; Gd_k];
    G = G + model.flagd * Gd;
end

%% ----------------- Adding Physics-informed gradient ---------------

if model.flagm > 0
    H = D2y(model.xm); % Hessian
    L = squeeze(H(1,1,:) + H(2,2,:))'; % Laplacian
    pde_res = model.kp * L + q(model.xm')';
    GL = D2yDw(model.xm);
    Gm_T = model.kp*GL*pde_res'/model.Nm;
    Gm_k = L*pde_res'/model.Nm;
    Gm = [Gm_T; Gm_k];
    G = G + model.flagm * Gm;
end
