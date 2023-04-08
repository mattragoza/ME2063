function hp3=hp3(x)
hp3 = (-2 + 6*tanh(x).^2) .* hp(x);
