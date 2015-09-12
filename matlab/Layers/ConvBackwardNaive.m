function [ dX, dW, db ] = ConvBackwardNaive( dY, cache )
%
% author: hgaolbb
% version: beta 0.01
%
% X: [N, C, Hx, Wx]
% W: [F, C, Hw, Ww]
% b: [F, 1]
% conv_param:
%               pad
%               stride
% dY: [N, F, Hy, Wy]
% example: 
% cache.X = rand(100,3,4,4);cache.W = rand(5,3,3,3);cache.b = rand(5,1);
% dY = rand(100,5,4,4)
% cache.conv_param.pad = 1; cache.conv_param.stride = 1;

X = cache.X;
W = cache.W;
b = cache.b;
pad = cache.conv_param.pad;
stride = cache.conv_param.stride;
% [N, C, Hx, Wx] = size(X);
% [F, C, Hw, Ww] = size(W);
% pad_hw = (Hw - 1) / 2;
% pad_ww = (Ww - 1) / 2;
[~, ~, Hx, Wx] = size(X);
[~, C, ~, ~] = size(W);
[N, F, Hy, Wy] = size(dY);

% change X to [HH, WW, C, N]
X = permute(X, [3, 4, 2, 1]);
padded_X = padarray(X, [1,1]);
% change W to [HH, WW, C, F]
W = permute(W, [3, 4, 2, 1]);
% % change Y to [HH, WW, C, F]
% dY = permute(dY, [3, 4, 2, 1]);

% dX = zeros(size(X));
dW = zeros(size(W));
db = zeros(size(b));
padded_dX = zeros([Hx+pad*2, Wx+pad*2, C, N]);

for i = 1:N
    for f = 1:F
       for hy = 1:Hy
           for wy = 1:Wy
               % dy(i,j) <==> padded_x(1+pad+(i-1)*stride, 1+pad+(i-1)*stride);
               window = padded_X(1+(hy-1)*stride:1+2*pad+(hy-1)*stride, 1+(wy-1)*stride:1+2*pad+(wy-1)*stride, :, i);
               db(f) = db(f) + dY(i,f,hy,wy);
               dW(:,:,:,f) = dW(:,:,:,f) + dY(i,f,hy,wy) .* window;
               padded_dX(1+(hy-1)*stride:1+2*pad+(hy-1)*stride, 1+(wy-1)*stride:1+2*pad+(wy-1)*stride, :, i) = ...\
                       padded_dX(1+(hy-1)*stride:1+2*pad+(hy-1)*stride, 1+(wy-1)*stride:1+2*pad+(wy-1)*stride, :, i) + ...\
                       dY(i,f,hy,wy) .* W(:,:,:,f);
           end
       end
    end
end
dX = permute(padded_dX(2:Hx+1, 2:Wx+1, :, :), [4, 3, 1, 2]);
dW = permute(dW, [4, 3, 1, 2]);

end

