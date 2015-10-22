function [ Y, cache ] = conv_relu_pool_forward( X, W, b, conv_param, pool_param )
%
% author: hgaolbb
% version: beta 0.01
%
% 把layers放在一起，组成大layer
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
%

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

