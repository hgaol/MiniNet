function [ Y, cache ] = ReluLayerForward( X )
%
% author: hgaolbb
% version: beta 0.01
%
% X: [N, C, HH, WW]
% Y: [N, C, HH, WW]
% because of affine layer, W and X must have the same HH and WW
% example: X = random('norm',0,1,[100,3,2,2])
Y = max(0, X);

% test_X = permute(X, [3,4,2,1]);
% test_Y = max(0, test_X);
% test_Y = permute(test_Y, [4,3,1,2]);
% Y == test_Y;

cache = X;

end

