function [net, Del_1, Del_2, Del_3, Del_4, Del_5, Del_6, Del_7, del_2, del_3,del_4, del_5, del_6, del_7, del_8] = Backpropagation(x, y, t,net)

% Hidden Layer 6
del_8 = (y' - t);
Del_7 = del_8 * net.ah6';

% Hidden layer 5
d7 = Sigmoid_derivative(net.oh6);
del_7 = d7.*(del_8 * net.W7');
Del_6 = (net.ah5 * del_7')';

% Hidden Layer 4
d6 = Sigmoid_derivative(net.oh5);
del_6 = d6.*(net.W6'* del_7);
Del_5 = (net.ah4* del_6')';


% Hidden Layer 3
d5 = Sigmoid_derivative(net.oh4);
del_5 = d5.* (net.W5' * del_6);
Del_4 = (net.ah3* del_5')';

% Hidden Layer 2
d4 = Sigmoid_derivative(net.oh3);
del_4 = d4.* (net.W4' * del_5);
Del_3 = (net.ah2* del_4')';

% Hidden Layer 1
d3 = Sigmoid_derivative(net.oh2);
del_3 = d3.* (net.W3' * del_4);
Del_2 = (net.ah1* del_3')';

% Input Layer
d2 = Sigmoid_derivative(net.oh1);
del_2 = d2.* (net.W2'* del_3 );
Del_1 =  (x * del_2')';

end

