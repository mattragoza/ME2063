function [loss, grad]=my_func(x)

%y = forward_pass(x);
%loss = sum(y.^2, "all");
%grad = 2*y.*Dy(x); % il,kl -> kl

loss = sum(Dy(x), "all"); % kl
grad = sum(D2y(x), 1); % kml
