function [best_model, loss_history, train_acc_history, val_acc_history]...\
                    = train(X, y, X_val, y_val, ...\
                            model, loss_function, reg, ...\
                            lr, momentum, lr_decay, ...\
                            num_epochs, batch_size, acc_frequency, ...\
                            update, sample_batches)
%
% author: hgaolbb
% version: beta 0.01
%
% Test for run
% X = ones(100,3,4,4);model.W1 = 1e-2 * ones(5,3,3,3);model.b1 = ones(5,1);model.W2 = ones(10,5,2,2);model.b2 = ones(10,1);
% model.conv_param.pad = 1; model.conv_param.stride = 1;y = randint(100,1,[1,10]);
% model.pool_param.stride = 2;model.pool_param.width = 2; model.pool_param.height = 2;
% [best_model, loss_history, train_acc_history, val_acc_history] = train(X, y, X, y, model, @two_layer_convnet);

% Test for correct
% X = rand(100,3,4,4);model.W1 = 1e-3 * random('norm',0,1,[5,1,3,3]);model.b1 = zeros(5,1);
% model.W2 = 1e-3 * random('norm',0,1,[10,5,2,2]);model.b2 = zeros(10,1);
% model.conv_param.pad = 1; model.conv_param.stride = 1;y = randint(100,1,[1,10]);
% model.pool_param.stride = 2;model.pool_param.width = 2; model.pool_param.height = 2;
% [best_model, loss_history, train_acc_history, val_acc_history] = train(X, y, X, y, model, @two_layer_convnet,0,0.015,1,0.95,1000);

%% argin
if nargin < 15
    sample_batches = true;
end
if nargin < 14
    update = 'sgd';
end
if nargin < 13
    acc_frequency = 0;
end
if nargin < 12
    batch_size = 100;
end
if nargin < 11
    num_epochs = 30;
end
if nargin < 10
    lr_decay = 0.95;
end
if nargin < 9
    momentum = 0;
end
if nargin < 8
    lr = 1e-2;
end
if nargin < 7
    reg = 0;
end
if nargin < 6
    fprintf('the number of args should at least be 6.');
end

%% 
N = size(X, 1);
if sample_batches
    iterations_per_epoch = N ./ batch_size;
else
    iterations_per_epoch = N;
end

num_iters = num_epochs * iterations_per_epoch;
epoch = 0;
best_val_acc = 0;
best_model = model;
loss_history = [];
train_acc_history = [];
val_acc_history = [];
step_cache = struct;

% TODO，将X，y随机化
for it = 1:num_iters
%     if mod(it, 10) == 0
%         fprintf('starting iteration %d\n', it);
%     end
    
    % 这里随机？顺序选取？还是选之前先打乱，然后顺序吧
    if sample_batches
%         X_batch = X(mod((it-1)*batch_size+1,N+1): mod(it*batch_size,N+1), :, :, :);
        X_batch = X(1:10,:,:,:);
        y_batch = y(1:10);
%         y_batch = y((it-1)*batch_size+1:it*batch_size, :);
    end
    
    [loss, grads] = loss_function(X_batch, model, y_batch, reg);
    fprintf('iter: %d\tloss: %d\n', it, loss);
    loss_history = [loss_history; loss];
    
    % updata
    param_name = fieldnames(grads);
    for i = 1:size(param_name)
        dname = param_name(i);
        dname = dname{1,1};
        if strcmp(update, 'sgd')
            dx = -lr .* grads.(dname);
        elseif strcmp(update, 'momentum')
            if ~isfield(step_cache, dname)
                step_cache.(dname) = zeros(size(grads.(dname)));
            end
            dx = momentum .* step_cache.(dname) - lr .* grads.(dname);
        elseif strcmp(update, 'rmsprop')
            ;
        else
            fprintf('Unrecogized update type "%s"\n', update);
        end
        
        model.(dname) = model.(dname) + dx;
    end
    
    first_it = (it == 1);
    epoch_end = mod(it, iterations_per_epoch) == 0;
    acc_check = (acc_frequency ~= 0 && mod(it, acc_frequency) == 0);
    if (first_it || epoch_end || acc_check)
        if (it > 1 && epoch_end)
            lr = lr * lr_decay;
            epoch = epoch + 1;
        end
        
        % evaluate train accuracy
        % random pick 1000 or less samples to calc acc
        if N > 1000
            % TODO
            X_train_subset = X(1:10,:,:,:);
            y_train_subset = y(1:10);
        else
            X_train_subset = X;
            y_train_subset = y;
        end
        scores_train = loss_function(X_train_subset, model);
        [~, y_pred_train] = max(scores_train, [], 2);
        train_acc = y_pred_train == y_train_subset;
        train_acc = mean(train_acc(:));
        train_acc_history = [train_acc_history; train_acc];
        
        % evaluate val accuracy
        scores_val = loss_function(X_val, model);
        [~, y_pred_val] = max(scores_val, [], 2);
        val_acc = (y_pred_val == y_val);
        val_acc = mean(val_acc(:));
        val_acc_history = [val_acc_history; val_acc];
        
        % keep track of the best model based on val acc
        if val_acc > best_val_acc
            best_val_acc = val_acc;
            for i = 1:size(param_name)
                dname = param_name(i);
                dname = dname{1,1};
                best_model.(dname) = model.(dname);
            end
        end
    end
end
