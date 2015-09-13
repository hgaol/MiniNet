addpath  ..\ ..\Layers
%% Overfit small data
X = 1e2 * randn(100,3,4,4);
model.W1 = 1e-2 * randn([5,3,3,3]);
model.b1 = zeros(5,1);
model.W2 = 1e-2 * randn([10,5,2,2]);
model.b2 = zeros(10,1);
model.conv_param.pad = 1; model.conv_param.stride = 1;y = randi([1,10],100,1);
model.pool_param.stride = 2;model.pool_param.width = 2; model.pool_param.height = 2;

% % net options
% options.loss_function = @two_layer_convnet;
% options.reg = 0;
% options.lr = 1e-3;
% options.momentum = 0.95;
% options.lr_decay = 0.99;
% options.num_epochs = 10;
% options.batch_size = 10;
options.update = 'adagrad';
% options.sample_batches = true;
% options.acc_frequency = 100;

[best_model, loss_history, train_acc_history, val_acc_history] = train(X, y, X, y, model);