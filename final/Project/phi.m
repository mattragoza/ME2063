function L = phi(x, P)
% Evaluate Legendre polynomials up to order P in the interval [a, b] at x
a = 0;
b = 180;
% Rescale x from [a, b] to [-1, 1]
x = 2*(x-a)/(b-a) - 1;

% Initialize array to hold Legendre polynomials
L = zeros(length(x), P);

% Evaluate Legendre polynomials up to order P
L(:,1) = 1;
L(:,2) = x;
for n = 2:P-1
    L(:,n+1) = ((2*n-1)*x.*L(:,n) - (n-1)*L(:,n-1))/n;
end

end
