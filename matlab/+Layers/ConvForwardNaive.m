function [ Y, cache ] = ConvForwardNaive( X, W, b, conv_param )
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
% Y: [N, F, Hy, Wy]
% example: x = rand(100,3,4,4);w = rand(5,3,3,3);b = rand(5,1);
% conv_param.pad = 1; conv_param.stride = 1;

[N, C, Hx, Wx] = size(X);
[F, C, Hw, Ww] = size(W);
cache.X = X;
cache.W = W;
cache.b = b;
cache.conv_param = conv_param;

% change X to [HH, WW, C, N]
X = permute(X, [3, 4, 2, 1]);
% change W to [HH, WW, C, F]
W = permute(W, [3, 4, 2, 1]);

% zero padding
pad = conv_param.pad;
X = padarray(X, [pad, pad]);

Hy = (Hx + pad * 2 - Hw) / conv_param.stride + 1;
Wy = (Wx + pad * 2 - Ww) / conv_param.stride + 1;

% [Hy, Wy, F, N]
Y = zeros([Hy, Wy, F, N]);
% conv
for i = 1:N
    for f = 1:F
        conv_ans = zeros([Hy, Wy]);
        for c = 1:C
            conv_ans = conv_ans + conv2(X(:,:,c,i), W(:,:,c,f), 'valid');
        end
        Y(:,:,f,i) = conv_ans + b(f, 1);
    end
end
% [N, C, Hy, Wy]
Y = permute(Y, [4, 3, 1, 2]);

% pad_a = pad * 2 + 1;
% Y = zeros([N, F, Hy, Wy]);
% for i = 1:N
%     for f = 1:F
%         for hy = 1:Hy
%             for wy = 1:Wy
%                 window = X(1+(hy-1)*conv_param.stride:pad_a+(hy-1)*conv_param.stride,...\
%                                     1+(wy-1)*conv_param.stride:pad_a+(wy-1)*conv_param.stride, :, i);
%                 Y(i,f,hy,wy) = sum(sum(sum((window .* W(:,:,:,f))))) + b(f);
%             end
%         end
%     end
% end
% 
% testY = permute(Y, [3,4,2,1]);
end

