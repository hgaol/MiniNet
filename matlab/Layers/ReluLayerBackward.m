function [ dX ] = ReluLayerBackward( dY, X )
%
% author: hgaolbb
% version: beta 0.01
%
% dY: [N, C, HH, WW]
% X: [N, C, HH, WW]
% dX: [N, C, HH, WW]
% because of affine layer, W and X must have the same HH and WW
% example: X = random('norm',0,1,[100,3,2,2])
dX = dY .* (X > 0);

end

