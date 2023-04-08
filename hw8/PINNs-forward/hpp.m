function hpp=hpp(x)
% hpp = 1-tanh(x).^2;
hpp = -2*tanh(x).*hp(x);
% hp = ones(size(x));