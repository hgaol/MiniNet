function [ Y, cache ] = MaxPoolForwardNaive( X, pool_param )
%
% author: hgaolbb
% version: beta 0.01
%
% X: [N, C, Hx, Wx]
% pool_param:
%               height
%               width
%               stride
% Y: [N, F, Hy, Wy]
% example:
% x = rand(100,3,4,4);pool_param.stride = 2;pool_param.width = 2; pool_param.height = 2
[N, C, Hx, Wx] = size(X);
cache.X = X;
cache.pool_param = pool_param;

pool_height = pool_param.height;
pool_width = pool_param.width;
stride = pool_param.stride;

% change X to [HH, WW, C, N]
X = permute(X, [3, 4, 2, 1]);

Hy = (Hx - pool_height) / stride + 1;
Wy = (Wx - pool_width) / stride + 1;
% [Hy, Wy, F, N]
Y = zeros([Hy, Wy, C, N]);
% conv
for i = 1:N
    for c = 1:C
        for hy = 1:Hy
            for wy = 1:Wy
                window = X(1+(hy-1)*stride:pool_height+(hy-1)*stride, 1+(wy-1)*stride:pool_width+(wy-1)*stride, c, i);
                Y(hy, wy, c, i) = max(window(:));
            end
        end
    end
end

% [N, C, Hy, Wy]
Y = permute(Y, [4, 3, 1, 2]);

end

