function [t, net] = Feedforward(x,net)

% Feed Forward Propagation. 
net.oh = (net.W1*x') + net.bias_W1;
net.ah = Sigmoid(net.oh);

% Output Layers
net.ok = (net.W2*net.ah) + net.bias_W2;
net.ak = net.ok;
t = net.ak;
end

