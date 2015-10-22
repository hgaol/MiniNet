function [ Y, cache ] = ReluLayerForward( X )
%
% author: hgaolbb
% version: beta 0.01
%
% X: [N, C, HH, WW]
% Y: [N, C, HH, WW]
% because of affine layer, W and X must have the same HH and WW
% 

Y = max(0, X);

cache = X;

end

