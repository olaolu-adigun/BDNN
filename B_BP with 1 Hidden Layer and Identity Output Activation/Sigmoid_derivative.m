function derive = Sigmoid_derivative(x)
%
%d = Sigmoid(x);
%derive = (d + 1).*(1 -d);
d = Sigmoid(x) + 1;
derive = 2* 0.5 * d.*(2-d);
end

