function [test_MSE, test_MSE_b] = main()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% BIDIRECTIONAL BACKPROPAGATION %%%%%%%

%%%%%%%%%%%%%% TRAINING APPROACH %%%%%%%%%%%%%%
%%%%Forward First - Backward Second %%%%%%%%%%%
%%%%Forward First, Update, Compute Error %%%%%%
%%%%Backward Second, Update, Compute Error %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Define Dataset
clear;
load Data;

% Training set
train_x = H(1:18000,1:5);
train_y = H(1:18000,6:10);

test_x = H(18001:20000,1:5);
test_y = H(18001:20000, 6:10);

%% Initialize trianng parameters and weights  
% Define the optimization parameter
opts.numepochs = 180;
opts.batch = 100;
opts.learning = 0.1;
assert(size(train_x,1)==size(train_y, 1), 'Check the data set.');

% Define the Neural Network
nn.layers = {
    struct('type', 'input', 'size', size(train_x,2))
    struct('type', 'hidden', 'size', 100)
    struct('type', 'output','size', size(train_y,2))
    };

%% Weight and bias random range
e = 1.0;
b = -e;
opts.e = e;

% Initialize the weights 
nn.W1 = unifrnd(b, e,nn.layers{2}.size,size(train_x,2));
nn.W2 = unifrnd(b, e,nn.layers{3}.size,  nn.layers{2}.size);

nn.bias_W2 = unifrnd(b,e,nn.layers{3}.size,1); 
nn.bias_W1 = unifrnd(b,e, nn.layers{2}.size, 1);

nnb.bias_W1 = nn.bias_W1;
nnb.bias_W2 = unifrnd(b,e,size(train_x,2),1);

%% Training

train_MSE = zeros((opts.numepochs),1);
kk = randperm(size(train_x,1));

for iter = 1:1:opts.numepochs
    
    opts.learning = opts.learning * (0.9999^iter);
    
    ind = randi(18000,100,1);
    ind  = sort(ind);
    batch_x = train_x(ind,:);
    batch_y = train_y(ind,:);
      
    del1 = zeros(size(nn.W1));
    del2 = zeros(size(nn.W2));
     
    del_b2 = zeros(size(nn.bias_W2));
  
    del_b1 = zeros(size(nn.bias_W1));
    
    %% Forward 
    for i = 1:1:opts.batch
        
        x = batch_x(i, :);
        y = batch_y(i, :);
        
        [o, nn] = Feedforward(x,nn);
        [nn, Delta_1, Delta_2, delb_1, delb_2] = Backpropagation(x, y, o, nn);
        
        nnb.W1 = (nn.W2)';
        nnb.W2 = (nn.W1)';
        
        del1 = del1 + Delta_1;
        del2 = del2 + Delta_2;
        
        del_b2 = del_b2 + delb_2;
        del_b1 = del_b1 + delb_1;
    end
    
    %% Update
    nn.W1 = nn.W1 + ((1 / opts.batch) * opts.learning * del1);
    nn.W2 = nn.W2 + ((1 / opts.batch) * opts.learning * del2);
    
    nn.bias_W1 = nn.bias_W1 + ((1/opts.batch)*opts.learning * del_b1);
    nn.bias_W2 = nn.bias_W2 + ((1/opts.batch)*opts.learning * del_b2);
    
    nnb.bias_W1 = nn.bias_W1;
    nnb.W1 = (nn.W2)';
    nnb.W2 = (nn.W1)';
    
    %% Compute Error
    % Compute the Training Error 
    error = zeros(opts.batch, nn.layers{3}.size);
    error_b = zeros(opts.batch, nn.layers{1}.size);
    
    observed = zeros(opts.batch, size(batch_y,2));
    obser    = zeros(opts.batch, size(batch_y,2));
    
    for j = 1:1:opts.batch
        [observed(j,:), nn] = Feedforward(batch_x(j, :),nn);
        error(j, :) = observed(j,:) - batch_y(j, :);
        
        [obser(j,:) , nnb] = Feedforward(batch_y(j,:),nnb);
        error_b(j,:) = obser(j,:) - batch_x(j,:); 
    end
    
    train_MSE(iter,1) = mean(mean(error.^2));
    
    
    %% Backward Training
    del1 = zeros(size(nn.W1));
    del2 = zeros(size(nn.W2)); 
    
    delb_b2 = zeros(size(nnb.bias_W2));
    del_b1 = zeros(size(nn.bias_W1));
    
    % Backward 
    for i = 1:1:opts.batch
        
      x = batch_x(i, :);
      y = batch_y(i, :);
         
      [t ,nnb] = Feedforward(y,nnb);
      [nnb,D1,D2,d1,d2] = Backpropagation(y,x,t,nnb);
        
      del1 = del1 + D1;
      del2 = del2 + D2;
        
      delb_b2 = delb_b2 + d2;
      del_b1 = del_b1 + d1;
    end
    
    %% Update
    nnb.W1 = nnb.W1 + ((1 / opts.batch) * opts.learning * del1);
    nnb.W2 = nnb.W2 + ((1 / opts.batch) * opts.learning * del2);
    
    nnb.bias_W1 = nn.bias_W1 + ((1/opts.batch)*opts.learning * del_b1);
    nnb.bias_W2 = nnb.bias_W2 + ((1/opts.batch)*opts.learning * delb_b2);
    
    nn.bias_W1 = nnb.bias_W1;
    nn.W1 = (nnb.W2)';
    nn.W2 = (nnb.W1)';
    
    %% Compute Error
    % Compute the Training Error 
    error = zeros(opts.batch, nn.layers{3}.size);
    error_b = zeros(opts.batch, nn.layers{1}.size);
    
    observed = zeros(opts.batch, size(batch_y,2));
    obser    = zeros(opts.batch, size(batch_y,2));
    
    for j = 1:1:opts.batch
        [observed(j,:), nn] = Feedforward(batch_x(j, :),nn);
        error(j, :) = observed(j,:) - batch_y(j, :);
        
        [obser(j,:) , nnb] = Feedforward(batch_y(j,:),nnb);
        error_b(j,:) = obser(j,:) - batch_x(j,:); 
    end
    
    train_MSE(iter,2) = mean(mean(error_b.^2));
end

figure(1);
hold on;
title('Training Error (MSE) vs. Iteration');
plot(train_MSE(:,1),'b');
hold on;
plot(train_MSE(:,2),'r');
grid on;
xlabel('Iteration');
ylabel('Training Error (MSE)');
        
%% COMPUTE THE TEST ERROR 
% Compute the Forward Error
err = zeros(size(test_y));
obs = zeros(size(test_y));
for q = 1:1:size(test_y,1)
    [ob, nn] = Feedforward(test_x(q, :),nn);
    obs(q,:) = ob';
    err(q, :) = ob' - test_y(q,:);
end
%ab = sum(sum(sign(obs) == test_y));
%disp(ab);
test_MSE = mean(mean(err.^2));
fprintf('Test set MSE Forward error: %f\n',test_MSE);
    
% Compute the Backward Error
err = zeros(size(test_y));
obs = zeros(size(test_y));
for q = 1:1:size(test_y,1)
    [ob, nnb] = Feedforward(test_y(q, :),nnb);
    obs(q,:) = ob';
    err(q, :) = ob' - test_x(q,:);
end
%ab = sum(sum(sign(obs) == test_x));
%disp(ab);
test_MSE_b = mean(mean(err.^2));
fprintf('Test set MSE Backward error: %f\n',test_MSE);
    
end