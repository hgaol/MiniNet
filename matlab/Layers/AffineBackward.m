function [ dX, dW, db ] = AffineBackward( dY, cache )
%
% author: hgaolbb
% version: beta 0.01
%
% dY: [N, F, 1, 1]
% cache: 
%        X: [N, C, HH, WW]
%        W: [F, C, HH, WW]
%        b: [F, 1]
% because of affine layer, W and X must have the same HH and WW
% output Y: [N, F, 1, 1]
% example: x = rand(100,3,2,2);w = rand(5,3,2,2);b = rand(5,1);

size_X = size(cache.X);
size_W = size(cache.W);
X = reshape(cache.X, size(cache.X, 1), []);
W = reshape(cache.W, size(cache.W, 1), []);

dX = reshape(dY * W, size_X);
dW = reshape(dY' * X, size_W);
db = sum(dY, 1)';

end

