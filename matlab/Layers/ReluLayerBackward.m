function [ dX ] = ReluLayerBackward( dY, X )
%
% author: hgaolbb
% version: beta 0.01
%
% dY: [N, C, HH, WW]
% X: [N, C, HH, WW]
% dX: [N, C, HH, WW]
% because of affine layer, W and X must have the same HH and WW
%

dX = dY .* (X > 0);

end

