%
% author: hgaolbb
% version: beta 0.01
%
addpath ..\Layers\
clear all;
%% Test function EvalNumGradientMat
a = reshape(1:9,3,3);
x = reshape(11:19,3,3);
dy = ones(3,3);

da_num = EvalNumGradientMat(@(a)fmat(a,x), a, dy);
dx_num = EvalNumGradientMat(@(x)fmat(a,x), x, dy);
fprintf('Testing EvalNumGradientMat:\n');
fprintf('diffence: %e\n', rel_error(da_num, x));
fprintf('diffence: %e\n', rel_error(dx_num, a));
%% Test function EvalNumGradient
a = 1:9;
x = (1:9)';
da_num = EvalNumGradient(@(a)fscalar(a,x), a);
dx_num = EvalNumGradient(@(x)fscalar(a,x), x);
fprintf('\nTesting EvalNumGradient:\n');
fprintf('diffence: %e\n', rel_error(da_num, x));
fprintf('diffence: %e\n', rel_error(dx_num, a));
%% Test SoftmaxLossLayer
addpath Layers;
num_classes = 10;
num_input = 50;
x = 1e-2 .* randn(num_input, num_classes);
y = randi([1,10], num_input, 1);
dx_num = EvalNumGradient(@(x)SoftmaxLossLayer(x,y), x);
[loss, dx] = SoftmaxLossLayer(x,y);
fprintf('\nTesting SoftmaxLossLayer:\n');
fprintf('loss: %d\n', loss);
fprintf('diffence: %e\n', rel_error(dx_num, dx));
%% Test SVM Layer
dx_num = EvalNumGradient(@(x)SVMLossLayer(x,y), x);
[loss, dx] = SVMLossLayer(x,y);
fprintf('\nTesting SVM Layer:\n');
fprintf('loss: %d\n', loss);
fprintf('diffence: %e\n', rel_error(dx_num, dx));
%% Test Affine Layer
x = randn(10, 2, 3);
w = randn(5, 6);
b = randn(5, 1);
dy = randn(10, 5);

dx_num = EvalNumGradientMat(@(x)AffineForward(x,w,b), x, dy);
dw_num = EvalNumGradientMat(@(w)AffineForward(x,w,b), w, dy);
db_num = EvalNumGradientMat(@(b)AffineForward(x,w,b), b, dy);
[~, cache] = AffineForward(x,w,b);
[dx, dw, db] = AffineBackward(dy, cache);
fprintf('\nTesting Affine Layer:\n');
fprintf('diffence dx: %e\n', rel_error(dx_num, dx));
fprintf('diffence dw: %e\n', rel_error(dw_num, dw));
fprintf('diffence db: %e\n', rel_error(db_num, db));
%% Test Relu Layer
x = randn(10, 2, 3);
dy = randn(size(x));

dx_num = EvalNumGradientMat(@(x)ReluLayerForward(x), x, dy);
[~, cache] = ReluLayerForward(x);
[dx] = ReluLayerBackward(dy, cache);
fprintf('\nTesting ReLU Layer:\n');
fprintf('diffence dx: %e\n', rel_error(dx_num, dx));
%% Convolution Layer
x = randn(4, 3, 5, 5);
w = randn(2, 3, 3, 3);
b = randn(2, 1);
dy = randn(4, 2, 5, 5);
conv_param.pad = 1; conv_param.stride = 1;

dx_num = EvalNumGradientMat(@(x)ConvForwardNaive(x,w,b,conv_param), x, dy);
dw_num = EvalNumGradientMat(@(w)ConvForwardNaive(x,w,b,conv_param), w, dy);
db_num = EvalNumGradientMat(@(b)ConvForwardNaive(x,w,b,conv_param), b, dy);
[~, cache] = ConvForwardNaive(x,w,b,conv_param);
[dx, dw, db] = ConvBackwardNaive(dy, cache);
fprintf('\nTesting CONV Layer:\n');
fprintf('diffence dx: %e\n', rel_error(dx_num, dx));
fprintf('diffence dw: %e\n', rel_error(dw_num, dw));
fprintf('diffence db: %e\n', rel_error(db_num, db));
%% Pooling Layer
x = randn(3, 2, 8, 8);
dy = randn(3, 2, 4, 4);
pool_param.height = 2;pool_param.width = 2;pool_param.stride = 2;
dx_num = EvalNumGradientMat(@(x)MaxPoolForwardNaive(x,pool_param), x, dy);
[~, cache] = MaxPoolForwardNaive(x,pool_param);
[dx] = MaxPoolBackwardNaive(dy, cache);
fprintf('\nTesting MaxPooling Layer:\n');
fprintf('diffence dx: %e\n', rel_error(dx_num, dx));
%% Dropout Layer
x = randn(8, 8);
dy = randn(8, 8);
dropout_param.mode = 'train';dropout_param.p = 0.7;
dx_num = EvalNumGradientMat(@(x)DropoutForward(x,dropout_param), x, dy);
[~, cache] = DropoutForward(x,dropout_param);
[dx] = DropoutBackward(dy, cache);
fprintf('\nTesting Dropout Layer:\n');
fprintf('diffence dx: %e\n', rel_error(dx_num, dx));
%% two layer conv net
x = randn(2, 2, 16, 16);
y = randi([1,10],2,1);
model.W1 = randn(5,2,3,3);
model.b1 = zeros(5,1);
model.W2 = randn(10,5,8,8);
model.b2 = zeros(10,1);
model.conv_param.pad = 1; model.conv_param.stride = 1;
model.pool_param.height = 2;model.pool_param.width = 2;model.pool_param.stride = 2;
[loss, grads] = two_layer_convnet(x,model,y,0);
[num_grads] = EvalNumGradientStruct(@(model)two_layer_convnet(x,model,y,0,'loss'), model);
disp('\nTesting two_layer_conv_net Layer:\n');
param_name = fieldnames(grads);
for i = 1:size(param_name)
    dname = param_name(i);
    dname = dname{1,1};
    e = rel_error(num_grads.(dname), grads.(dname));
    fprintf('diffence %s: %e\n', dname, e);
end


