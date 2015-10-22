function [best_model, loss_history, train_acc_history, val_acc_history]...\
                    = train(X, y, X_val, y_val, ...\
                            model, options)
%
% author: hgaolbb
% version: beta 0.01
% brief: train cnn
%

%% argin
if ~exist('options', 'var')
    options.loss_function = @two_layer_convnet;
    options.reg = 0;
    options.lr = 1e-2;
    options.momentum = 0.95;
    options.lr_decay = 0.99;
    options.num_epochs = 500;
    options.batch_size = 100;
    options.update = 'momentum';
    options.sample_batches = true;
    options.acc_frequency = 100;
else
    if ~isfield(options, 'loss_function')
        options.loss_function = @two_layer_convnet;
    end
    if ~isfield(options, 'reg')
        options.reg = 0;
    end
    if ~isfield(options, 'lr')
        options.lr = 1e-2;
    end
    if ~isfield(options, 'momentum')
        options.momentum = 0.95;
    end
    if ~isfield(options, 'lr_decay')
        options.lr_decay = 0.99;
    end
    if ~isfield(options, 'num_epochs')
        options.num_epochs = 30;
    end
    if ~isfield(options, 'batch_size')
        options.batch_size = 100;
    end
    if ~isfield(options, 'update')
        options.update = 'momentum';
    end
    if ~isfield(options, 'sample_batches')
        options.sample_batches= true;
    end
    if ~isfield(options, 'acc_frequency')
        options.acc_frequency = 100;
    end
end

%% 
N = size(X, 1);
% batch size per iteration
if options.sample_batches
    iterations_per_epoch = N ./ options.batch_size;
else
    iterations_per_epoch = N;
end

num_iters = options.num_epochs * iterations_per_epoch;
epoch = 0;
best_val_acc = 0;
best_model = model;
loss_history = [];
train_acc_history = [];
val_acc_history = [];
step_cache = struct;

% shuffle data
idx = randperm(size(X,1));
X = X(idx,:,:,:);
y = y(idx,:);
for it = 1:num_iters

    if options.sample_batches
        X_batch = X(mod((it-1)*options.batch_size+1,N): mod((it-1)*options.batch_size,N)+options.batch_size, :, :, :);
        y_batch = y(mod((it-1)*options.batch_size+1,N): mod((it-1)*options.batch_size,N)+options.batch_size, :, :, :);
    end
    
    [loss, grads] = options.loss_function(X_batch, model, y_batch, options.reg);

    loss_history = [loss_history; loss];
    
    % updata
    param_name = fieldnames(grads);
    for i = 1:size(param_name)
        dname = param_name(i);
        dname = dname{1,1};
        if strcmp(options.update, 'sgd')
            dx = -lr .* grads.(dname);
        elseif strcmp(options.update, 'momentum')
            if ~isfield(step_cache, dname)
                step_cache.(dname) = zeros(size(grads.(dname)));
            end
            dx = options.momentum .* step_cache.(dname) - options.lr .* grads.(dname);
        elseif strcmp(options.update, 'rmsprop')
            decay_rate = 0.99; % change it yourself
            if ~isfield(step_cache, dname)
                step_cache.(dname) = zeros(size(grads.(dname)));
            end
            step_cache.(dname) = decay_rate .* step_cache.(dname) + (1 - decay_rate) .* pow2(grads.(dname),2);
            dx = -options.lr .* grads.(dname) ./ sqrt(step_cache.(dname) + 1e-8);
        elseif strcmp(options.update, 'adagrad')
            if ~isfield(step_cache, dname)
                step_cache.(dname) = zeros(size(grads.(dname)));
            end
            step_cache.(dname) = pow2(grads.(dname), 2);
            dx = -options.lr .* grads.(dname) ./ sqrt(step_cache.(dname) + 1e-8);
        else
            fprintf('Unrecogized update type "%s"\n', options.update);
        end
        
        model.(dname) = model.(dname) + dx;
    end

%% Evaluate
    first_it = (it == 1);
    epoch_end = mod(it, iterations_per_epoch) == 0;
    acc_check = (options.acc_frequency ~= 0 && mod(it, options.acc_frequency) == 0);
    if (first_it || epoch_end || acc_check)
    % if 1 == 0
        if (it > 1 && epoch_end)
            options.lr = options.lr * options.lr_decay;
            epoch = epoch + 1;
        end
        
        % evaluate train accuracy
        % random pick 1000 or less samples to calc acc
        if N > 1000
            % pick shuffle 1000
            idx_train = randperm(size(X_batch,1));
            X_train_subset = X_batch(idx_train(1:100),:,:,:);
            y_train_subset = y_batch(idx_train(1:100),:);
        else
            X_train_subset = X;
            y_train_subset = y;
        end
        scores_train = options.loss_function(X_train_subset, model);
        [~, y_pred_train] = max(scores_train, [], 2);
        train_acc = y_pred_train == y_train_subset;
        train_acc = mean(train_acc(:));
        train_acc_history = [train_acc_history; train_acc];
        
        % evaluate val accuracy
        scores_val = options.loss_function(X_val, model);
        [~, y_pred_val] = max(scores_val, [], 2);
        val_acc = (y_pred_val == y_val);
        val_acc = mean(val_acc(:));
        val_acc_history = [val_acc_history; val_acc];
        fprintf('iter: %d\tloss: %d\ttrain_acc: %0.3f%%\tval_acc: %0.3f%%\tlr: %d\n',...\
                    it, loss, train_acc * 100, val_acc*100, options.lr);
        
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
