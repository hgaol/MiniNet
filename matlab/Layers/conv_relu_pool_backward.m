function [ dX, dW, db ] = conv_relu_pool_backward( dY, cache )
%
% author: hgaolbb
% version: beta 0.01
%
% dY: [N, F, Hy, Wy]
% cache:
%    conv_cache:
%       X: [N, C, Hx, Wx] ---- before conv
%       W: [F_conv, C, Hw, Ww]
%       b: [F_conv, 1]
%       conv_param: pad, stride
%    relu_cache:
%       X: [N, F_conv, H_relu, WW_relu] ---- before relu, after conv
%    pool_cache:
%       X: [N, F_conv, H_relu, W_relu] ---- before pooling, after conv
%       pool_param: height, width, stride
% example:
%   first, run conv_relu_pool_forward and 
%   then conv_relu_pool_backward(Y, cache), this is just for testing the code.
conv_cache = cache.conv_cache;
relu_cache = cache.relu_cache;
pool_cache = cache.pool_cache;
ds = MaxPoolBackwardNaive(dY, pool_cache);
da = ReluLayerBackward(ds, relu_cache);
[dX, dW, db] = ConvBackwardNaive(da, conv_cache);

end

