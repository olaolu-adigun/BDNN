function [mse,obs] = Compute_Backward_Error(X,Y,net)
%
N = size(X,1);
W1 = net.W2';
W2 = net.W1';

net.W1 = W1;
net.W2 = W2;
error = zeros(size(Y));
obs = zeros(size(Y));
for i = 1:1:N
    x = X(i,:);
    [ob, net] = Feedforward(x,net);
    error(i,:) = (Y(i,:) - ob').^2;
    obs(i,:) = ob';
end
%error = error(:,2:size(Y,2));
mse = mean(mean(error));
end

