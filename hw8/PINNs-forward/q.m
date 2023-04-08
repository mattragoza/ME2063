function q=q(X)
global L 
q = 250*exp(-20*(X(:,1)-L/6).^2-20*(X(:,2)-L/2).^2);