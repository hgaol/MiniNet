%
% author: hgaolbb
% version: beta 0.01
%
addpath ..\..\ ..\..\Layers
%% Mnist
% % net options
% options.loss_function = @two_layer_convnet;
% options.reg = 0;
options.lr = 1e-1;
% options.momentum = 0.95;
options.lr_decay = 0.98;
options.num_epochs = 500;
options.batch_size = 100;
% options.update = 'momentum';
% options.sample_batches = true;
options.acc_frequency = 5;

images = loadMNISTImages('t10k-images.idx3-ubyte');
labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10
images = permute(images, [2,1]);
images = reshape(images, [size(images,1),1,28,28]);
model.W1 = 1e-2 * randn([5,1,3,3]);
model.b1 = zeros(10,1);
model.W2 = 1e-2 * randn([10,5,14,14]);
model.b2 = zeros(10,1);
model.conv_param.pad = 1; model.conv_param.stride = 1;y = randi([1,10],100,1);
model.pool_param.stride = 2;model.pool_param.width = 2; model.pool_param.height = 2;

[best_model, loss_history, train_acc_history, val_acc_history] = ...\
                train(images(1:5000,:,:,:),labels(1:5000),images(1:1000,:,:,:),labels(1:1000),model,options);