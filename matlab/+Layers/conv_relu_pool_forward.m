function [ Y, cache ] = conv_relu_pool_forward( X, W, b, conv_param, pool_param )
%
% author: hgaolbb
% version: beta 0.01
%
% 把layer放在一起，组成大layer，不是基本的layer
% X: [N, C, Hx, Wx]
% W: [F, C, Hw, Ww]
% b: [F, 1]
% conv_param:
%           pad
%           stride
% pool_param:
%               height
%               width
%               stride
% example:
% x = rand(100,3,4,4);w = rand(5,3,3,3);b = rand(5,1);
% conv_param.pad = 1; conv_param.stride = 1;
% pool_param.stride = 2;pool_param.width = 2; pool_param.height = 2;
% [Y, cache] = conv_relu_pool_forward(x,w,b,conv_param,pool_param);

[a, conv_cache] = ConvForwardNaive(X, W, b, conv_param);
[s, relu_cache] = ReluLayerForward(a);
[Y, pool_cache] = MaxPoolForwardNaive(s, pool_param);
cache.conv_cache = conv_cache;
cache.relu_cache = relu_cache;
cache.pool_cache = pool_cache;

% % test
% [Y, conv_cache] = ConvForwardNaive(X, W, b, conv_param);
% cache.conv_cache = conv_cache;

end

