function [model,FEM_M,FEM_K,FEM_F] = GetFEMMatmodel(xc,yc,model)
global EID
f = @(location,state)10000*exp(-((location.x - xc).^2+(location.y-yc).^2)/0.05);
specifyCoefficients(model,'m',0,'d',1,'c',1,'a',0,'f',f);
model_FEM_matrices = assembleFEMatrices(model);
FEM_M = model_FEM_matrices.M;
FEM_K = model_FEM_matrices.K;
FEM_F = model_FEM_matrices.F;
    
FEM_M(EID,:)= 0 ;
FEM_M(:,EID)= 0 ;
FEM_M(EID,EID)= eye(length(EID)) ;
