function loss=loss_k(k)
global model

H = D2y(model.xm); % Hessian
L = squeeze(H(1,1,:) + H(2,2,:))'; % Laplacian
pde_res = k * L + q(model.xm')';
loss = sum(pde_res.^2, "all")/(2*model.Nm);
