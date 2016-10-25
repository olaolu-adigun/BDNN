function [test_MSE, err_b] = main()

% [train_MSE, test_MSE, err_b, forw_Y, back_X]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%TRAINING APPROACH(Regression)%%%%%%%%%
%%%%%%%Forward First - Backward Second%%%%%%%%%
%%%%%Forward First, Update, Compute Error%%%%%%
%%%%Backward Second, Update, Compute Error%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Define Dataset

clear;


load Train_data;
load Test_data;
% Training set
train_x = LH(1:40000,2);
train_y = LH(1:40000,1);

% Test set
test_x = TH(1:40000,2);
test_y = TH(1:40000,1);


%% Initialize trianng parameters and weights  

%---Define the optimization parameter
opts.numepochs = 200;
opts.batch = 200;
opts.learning = 0.1;
assert(size(train_x,1) == size(train_y, 1), 'Check the data set.');

% Define the Neural Network
nn.layers = {
    struct('type', 'input', 'size', size(train_x,2))
    struct('type', 'hidden1', 'size', 10)
    struct('type', 'hidden2', 'size', 10)
    struct('type', 'hidden3', 'size', 10)
    struct('type', 'hidden4', 'size', 10)
    struct('type', 'hidden5', 'size', 10)
     struct('type', 'hidden6', 'size',10)
    struct('type', 'output','size', size(train_y,2))
    };

%% Weight and bias random range
e = 1.5;
b = -e;
opts.e = e;

%---Initialize the weights 
nn.W1 = unifrnd(b, e, nn.layers{2}.size, size(train_x,2));
nn.W2 = unifrnd(b, e, nn.layers{3}.size, nn.layers{2}.size);
nn.W3 = unifrnd(b, e, nn.layers{4}.size, nn.layers{3}.size);
nn.W4 = unifrnd(b, e, nn.layers{5}.size, nn.layers{4}.size);
nn.W5 = unifrnd(b, e, nn.layers{6}.size, nn.layers{5}.size);
nn.W6 = unifrnd(b, e, nn.layers{7}.size, nn.layers{6}.size);
nn.W7 = unifrnd(b, e, nn.layers{8}.size, nn.layers{7}.size);


nn.bias1 = unifrnd(b,e, nn.layers{1}.size, 1);
nn.bias2 = unifrnd(b,e, nn.layers{2}.size, 1);
nn.bias3 = unifrnd(b,e, nn.layers{3}.size, 1); 
nn.bias4 = unifrnd(b,e, nn.layers{4}.size, 1);
nn.bias5 = unifrnd(b,e, nn.layers{5}.size, 1);
nn.bias6 = unifrnd(b,e, nn.layers{6}.size, 1);
nn.bias7 = unifrnd(b,e, nn.layers{7}.size, 1);
nn.bias8 = unifrnd(b,e, nn.layers{8}.size, 1);

%% TRAINING

train_MSE = zeros(opts.numepochs,1);

for iter = 1:1:opts.numepochs
    
    opts.learning = opts.learning * (0.9999^iter);

    ind = sort(randi(40000,200,1));
    batch_x = train_x(ind,:);
    batch_y = train_y(ind,:);
      
    delW1 = zeros(size(nn.W1));
    delW2 = zeros(size(nn.W2));
    delW3 = zeros(size(nn.W3));
    delW4 = zeros(size(nn.W4));
    delW5 = zeros(size(nn.W5));
    delW6 = zeros(size(nn.W6));
    delW7 = zeros(size(nn.W7));
  
    %del_b1 = zeros(size(nn.bias1));
    del_b2 = zeros(size(nn.bias2));
    del_b3 = zeros(size(nn.bias3));
    del_b4 = zeros(size(nn.bias4));
    del_b5 = zeros(size(nn.bias5));
    del_b6 = zeros(size(nn.bias6));
    del_b7 = zeros(size(nn.bias7));
    del_b8 = zeros(size(nn.bias8));
    
    %---Forward Training
    for i = 1:1:opts.batch
        
        x = batch_x(i, :);
        y = batch_y(i, :);
        
        [o, nn] = Feedforward(x,nn);
        [nn, Del_1, Del_2, Del_3, Del_4, Del_5, Del_6, Del_7, del_2, del_3, del_4, del_5, del_6, del_7, del_8] = Backpropagation(x, y, o, nn);
        
        delW1 = delW1 + Del_1;
        delW2 = delW2 + Del_2;
        delW3 = delW3 + Del_3;
        delW4 = delW4 + Del_4;
        delW5 = delW5 + Del_5;
        delW6 = delW6 + Del_6;
        delW7 = delW7 + Del_7;
                
        
        %del_b1 = del_b1 + (d1);
        del_b2 = del_b2 + (del_2);
        del_b3 = del_b3 + (del_3);
        del_b4 = del_b4 + (del_4);
        del_b5 = del_b5 + (del_5);
        del_b6 = del_b6 + (del_6);
        del_b7 = del_b7 + (del_7);
        del_b8 = del_b8 + (del_8);
    end
    
    %% Update Weights and Bias
    nn.W1 = nn.W1 + ((1 / opts.batch) * opts.learning * delW1);
    nn.W2 = nn.W2 + ((1 / opts.batch) * opts.learning * delW2);
    nn.W3 = nn.W3 + ((1 / opts.batch) * opts.learning * delW3);
    nn.W4 = nn.W4 + ((1 / opts.batch) * opts.learning * delW4);
    nn.W5 = nn.W5 + ((1 / opts.batch) * opts.learning * delW5);
    nn.W6 = nn.W6 + ((1 / opts.batch) * opts.learning * delW6);
    nn.W7 = nn.W7 + ((1 / opts.batch) * opts.learning * delW7);
    
    %nn.bias1 = nn.bias1 + ((1/opts.batch)*opts.learning * del_b1);
    nn.bias2 = nn.bias2 + ((1/opts.batch)*opts.learning * del_b2);
    nn.bias3 = nn.bias3 + ((1/opts.batch)*opts.learning * del_b3);
    nn.bias4 = nn.bias4 + ((1/opts.batch)*opts.learning * del_b4);
    nn.bias5 = nn.bias5 + ((1/opts.batch)*opts.learning * del_b5);
    nn.bias6 = nn.bias6 + ((1/opts.batch)*opts.learning * del_b6);
    nn.bias7 = nn.bias7 + ((1/opts.batch)*opts.learning * del_b7);
    nn.bias8 = nn.bias8 + ((1/opts.batch)*opts.learning * del_b8);
    
    nnb.W1 = (nn.W7)';
    nnb.W2 = (nn.W6)';
    nnb.W3 = (nn.W5)';
    nnb.W4 = (nn.W4)';
    nnb.W5 = (nn.W3)';
    nnb.W6 = (nn.W2)';
    nnb.W7 = (nn.W1)';
    
    nnb.bias1 = nn.bias8;
    nnb.bias2 = nn.bias7;
    nnb.bias3 = nn.bias6;
    nnb.bias4 = nn.bias5;
    nnb.bias5 = nn.bias4;
    nnb.bias6 = nn.bias3;
    nnb.bias7 = nn.bias2;
    nnb.bias8 = nn.bias1;
    
    %---Compute the Forward Training Error 
    error = zeros(opts.batch, nn.layers{5}.size);
    observed = zeros(size(batch_y));
  
    
    for j = 1:1:opts.batch
        [observed(j,:), nn] = Feedforward(batch_x(j, :),nn);
        error(j, :) = observed(j,:) - batch_y(j, :);
    end
    
    % Forward Error
    train_MSE(iter) = mean(mean(error.^2));
    

    %% Backward Training 
    
    delW1 = zeros(size(nn.W1));
    delW2 = zeros(size(nn.W2));
    delW3 = zeros(size(nn.W3));
    delW4 = zeros(size(nn.W4));
    delW5 = zeros(size(nn.W5));
    delW6 = zeros(size(nn.W6));
    delW7 = zeros(size(nn.W7));
  
     
    del_b1 = zeros(size(nn.bias1));
    del_b2 = zeros(size(nn.bias2));
    del_b3 = zeros(size(nn.bias3));
    del_b4 = zeros(size(nn.bias4));
    del_b5 = zeros(size(nn.bias5));
    del_b6 = zeros(size(nn.bias6));
    del_b7 = zeros(size(nn.bias7));
    del_b8 = zeros(size(nn.bias8));
    
    for i = 1:1:opts.batch
        
        x = batch_x(i, :);
        y = batch_y(i, :);
     
        nnb.W1 = (nn.W7)';
        nnb.W2 = (nn.W6)';
        nnb.W3 = (nn.W5)';
        nnb.W4 = (nn.W4)';
        nnb.W5 = (nn.W3)';
        nnb.W6 = (nn.W2)';
        nnb.W7 = (nn.W1)';
        
        nnb.bias1 = nn.bias8;
        nnb.bias2 = nn.bias7;
        nnb.bias3 = nn.bias6;
        nnb.bias4 = nn.bias5;
        nnb.bias5 = nn.bias4;
        nnb.bias6 = nn.bias3;
        nnb.bias7 = nn.bias2;
        nnb.bias8 = nn.bias1;
        
        [t ,nnb] = Feedforward(y,nnb);
        [nnb,D7,D6,D5, D4,D3,D2,D1,d7,d6,d5,d4,d3,d2,d1] = Backpropagation(y,x,t,nnb);
        
        delW1 = delW1 + D1';
        delW2 = delW2 + D2';
        delW3 = delW3 + D3';
        delW4 = delW4 + D4';
        delW5 = delW5 + D5';
        delW6 = delW6 + D6';
        delW7 = delW7 + D7';
                
        del_b1 = del_b1 + (d1);
        del_b2 = del_b2 + (d2);
        del_b3 = del_b3 + (d3);
        del_b4 = del_b4 + (d4);
        del_b5 = del_b5 + (d5);
        del_b6 = del_b6 + (d6);
        del_b7 = del_b7 + (d7);
        % del_b8 = del_b8 + (del_8);
    end
    
    % Update Weights and Bias
    nn.W1 = nn.W1 + ((1 / opts.batch) * opts.learning * delW1);
    nn.W2 = nn.W2 + ((1 / opts.batch) * opts.learning * delW2);
    nn.W3 = nn.W3 + ((1 / opts.batch) * opts.learning * delW3);
    nn.W4 = nn.W4 + ((1 / opts.batch) * opts.learning * delW4);
    nn.W5 = nn.W5 + ((1 / opts.batch) * opts.learning * delW5);
    nn.W6 = nn.W6 + ((1 / opts.batch) * opts.learning * delW6);
    nn.W7 = nn.W7 + ((1 / opts.batch) * opts.learning * delW7);
    
    nn.bias1 = nn.bias1 + ((1/opts.batch)*opts.learning * del_b1);
    nn.bias2 = nn.bias2 + ((1/opts.batch)*opts.learning * del_b2);
    nn.bias3 = nn.bias3 + ((1/opts.batch)*opts.learning * del_b3);
    nn.bias4 = nn.bias4 + ((1/opts.batch)*opts.learning * del_b4);
    nn.bias5 = nn.bias5 + ((1/opts.batch)*opts.learning * del_b5);
    nn.bias6 = nn.bias6 + ((1/opts.batch)*opts.learning * del_b6);
    nn.bias7 = nn.bias7 + ((1/opts.batch)*opts.learning * del_b7);
    %nn.bias8 = nn.bias8 + ((1/opts.batch)*opts.learning * del_b8);
    
    nnb.W1 = (nn.W7)';
    nnb.W2 = (nn.W6)';
    nnb.W3 = (nn.W5)';
    nnb.W4 = (nn.W4)';
    nnb.W5 = (nn.W3)';
    nnb.W6 = (nn.W2)';
    nnb.W7 = (nn.W1)';
    
    nnb.bias1 = nn.bias8;
    nnb.bias2 = nn.bias7;
    nnb.bias3 = nn.bias6;
    nnb.bias4 = nn.bias5;
    nnb.bias5 = nn.bias4;
    nnb.bias6 = nn.bias3;
    nnb.bias7 = nn.bias2;
    nnb.bias8 = nn.bias1;
    
    % Compute Backward Training Error 
   
    error_b = zeros(opts.batch, nn.layers{1}.size);
    obser = zeros(size(batch_x));
    
    for j = 1:1:opts.batch
        [obser(j,:) , nnb] = Feedforward(batch_y(j,:),nnb);
        error_b(j,:) = obser(j,:) - batch_x(j,:); 
    end
   
    train_MSE(iter,2) = mean(mean(error_b.^2));
end
    figure(9);
    hold on;
    title('Training Error (MSE) vs. Iteration');
    plot(train_MSE(:,1),'r');
    hold on;
    plot(train_MSE(:,2),'b');
    grid on;
    
    xlabel('Iteration');
    ylabel('Training Error (MSE)');
    
    figure(10);
    title('Forward Mapping');
    plot(batch_x,batch_y,'b');
    hold on;
    plot(batch_x, observed,'r');
    
    figure(11);
    title('Inverse Mapping');
    plot(batch_y,batch_x,'b');
    hold on;
    plot(batch_y, obser,'r');
   
 % Compute the Test Error
    err = zeros(size(test_y));
    obs = zeros(size(test_y));
    for q = 1:1:size(test_y,1)
        [ob, nn] = Feedforward(test_x(q, :),nn);
        obs(q,:) = ob';
        err(q, :) = ob' - test_y(q,:);
    end
    test_MSE = mean(mean(err.^2));
    fprintf('Test set MSE Forward error: %f\n',test_MSE);
    %forw_Y = obs;
    
    %% Compute the feedbackward error with forward training only
    [err_b, back_X] = Compute_Backward_Error(test_y, test_x, nnb);
    fprintf('Test set MSE Backward error: %f\n',err_b);
end

