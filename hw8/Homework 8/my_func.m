function [loss, grad]=my_func(x)

y = forward_pass(x);
loss = sum(y.^2, "all");
grad = 2*y.*Dy(x); % il,kl -> kl

%d = Dy(x); % kl
%loss = sum(d.^2, "all");
%grad = 2*d.*D2y(x); % kl,kl -> kl

%y = hpp(x);
%loss = sum(y.^2, "all");
%grad = 2*y.*hp3(x);
