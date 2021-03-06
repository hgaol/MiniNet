function [ loss, grads ] = two_layer_convnet( X, model, y, reg, ret_type)
%
% author: hgaolbb
% version: beta 0.01
%
% X: [N, C, Hx, Wx]
% model :
%       W1: []
%       b1:
%       W2:
%       b2:
%       conv_param:
%           pad
%           stride
%       pool_param:
%               height
%               width
%               stride
% conv->relu->pool->affine
% backward:
% example to test:
% X = randn(100,3,4,4);model.W1 = 1e-2 * randn(5,3,3,3);model.b1 = randn(5,1);
% model.W2 = 1e-2 * randn(10,5,2,2); model.b2 = randn(10,1);
% model.conv_param.pad = 1; model.conv_param.stride = 1;
% model.pool_param.stride = 2;model.pool_param.width = 2; model.pool_param.height = 2;
% y = randint(100,1,[1,10]);
% two_layer_convnet(X, model, y, 0);

if nargin == 2
    y = false;
    reg = 0;
    ret_type = 'scores';
end
if nargin == 3
    reg = 0;
    ret_type = 'go_back';
end
if nargin == 4
    ret_type = 'go_back';
end

%% Test code
% % test conv layer
% % X = randn(100,3,4,4);model.W1 = 1e-2 * ones(5,3,3,3);model.b1 = randn(5,1);model.W2 = rand(10,5,4,4);model.b2 = rand(10,1);
% % X = ones(100,3,4,4);model.W1 = 1e-2 * ones(5,3,3,3);model.b1 = ones(5,1);model.W2 = ones(10,5,2,2);model.b2 = ones(10,1);
% % model.conv_param.pad = 1; model.conv_param.stride = 1;y = randint(100,1,[1,10]);
% % model.pool_param.stride = 2;model.pool_param.width = 2; model.pool_param.height = 2;
% [a1, cache1] = conv_relu_pool_forward(X, model.W1, model.b1, model.conv_param, model.pool_param);
% [scores, cache2] = AffineForward(X, model.W2, model.b2);
% [data_loss, dscores] = SoftmaxLossLayer(scores, y);

% test affine layer
% model.a1 = rand(100,3,2,2);
% [scores, cache2] = AffineForward(model.a1, model.W2, model.b2);
% [data_loss, dscores] = SoftmaxLossLayer(scores, y);

%% Forward
[a1, cache1] = conv_relu_pool_forward(X, model.W1, model.b1, model.conv_param, model.pool_param);
[scores, cache2] = AffineForward(a1, model.W2, model.b2);

if strcmp(ret_type, 'scores')
    loss = scores;
    return;
end
[data_loss, dscores] = SoftmaxLossLayer(scores, y);
if strcmp(ret_type, 'loss')
    loss = data_loss;
    return;
end

%% Backward
[da1, dW2, db2] = AffineBackward(dscores, cache2);
[dX, dW1, db1] = conv_relu_pool_backward(da1, cache1);

% add regularization
dW1 = dW1 + reg .* model.W1;
dW2 = dW2 + reg .* model.W2;
reg_loss = 0.5 .* reg .* (sum(model.W1(:)) + sum(model.W2(:)));

loss = data_loss + reg_loss;
grads.W1 = dW1;
grads.W2 = dW2;
grads.b1 = db1;
grads.b2 = db2;

end

