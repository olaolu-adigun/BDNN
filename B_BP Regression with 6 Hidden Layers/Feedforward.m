function [t, net] = Feedforward(x,net)

% Feed Forward Propagation.
% Hidden Layer 1
net.oh1 = (net.W1 * x') + net.bias2;
net.ah1 = Sigmoid(net.oh1);

% Hidden Layer 2
net.oh2 = (net.W2 * net.ah1) + net.bias3;
net.ah2 = Sigmoid(net.oh2);

% Hidden Layer 3
net.oh3 = (net.W3 * net.ah2) + net.bias4;
net.ah3 = Sigmoid(net.oh3);

% Hidden Layers 4 
net.oh4 = (net.W4 * net.ah3) + net.bias5;
net.ah4 = Sigmoid(net.oh4);

% Hidden Layers 5 
net.oh5 = (net.W5 * net.ah4) + net.bias6;
net.ah5 = Sigmoid(net.oh5);

% Hiidden Layers 6
net.oh6 = (net.W6 * net.ah5) + net.bias7;
net.ah6 = Sigmoid(net.oh6);

% Output Layer
net.ok = (net.W7 * net.ah6) + net.bias8;
net.ak = net.ok;
t = net.ak;
end

