function [net, Delta_1,Delta_2, delb_1, delb_2] = Backpropagation(x, y, t,net)

% Backpropagate error from the output to the Hidden layer
delta_2 = (y' - t)* net.ah';
Delta_2 = delta_2;
delb_2 = (y' - t);

% Backpropagate from the Hidden layer to the Inpuut layer
del1 = Sigmoid_derivative(net.oh);
delta_1 = del1.*((y - t') * net.W2)';
Delta_1 = delta_1 * x;
delb_1 = delta_1;

end

