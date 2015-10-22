function [dX] = DropoutBackward( dY, cache )
%
% author: hgaolbb
% version: beta 0.01
%
% X: [N, C, HH, WW]
% Y: [N, C, HH, WW]
% dropout_param:
%               mode: train/test
%               p
% 

if (strcmp(cache.dropout_param.mode, 'train'))
    dX = dY .* cache.mask;
end
if (strcmp(cache.dropout_param.mode, 'test'))
    dX = dY;
end

end

