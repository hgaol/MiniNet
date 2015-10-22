function [ dX ] = MaxPoolBackwardNaive( dY, cache )
%
% author: hgaolbb
% version: beta 0.01
%
% dY: [N, C, Hy, Wy]
% cache:
%       X: [N, C, Hx, Wx]
%       pool_param: stride, height, width
% dX: [N, C, Hx, Wx]
% 

[N, C, Hy, Wy] = size(dY);
stride = cache.pool_param.stride;
pool_height = cache.pool_param.height;
pool_width = cache.pool_param.width;
X = cache.X;
% change X to [Hx, Wx, C, N]
X = permute(X, [3, 4, 2, 1]);

Hx = (Hy - 1) * stride + pool_height;
Wx = (Wy - 1) * stride + pool_width;

dX = zeros([Hx, Wx, C, N]);

for i = 1:N
    for c = 1:C
        for hy = 1:Hy
            for wy = 1:Wy
                window = X(1+(hy-1)*stride:pool_height+(hy-1)*stride, 1+(wy-1)*stride:pool_width+(wy-1)*stride, c, i);
                max_val = max(window(:));
                dX(1+(hy-1)*stride:pool_height+(hy-1)*stride, 1+(wy-1)*stride:pool_width+(wy-1)*stride, c, i) = ...\
                        (window == max_val) .* dY(i,c,hy,wy);
            end
        end
    end
end
% [N, C, Hx, Wx]
dX = permute(dX, [4, 3, 1, 2]);
end

