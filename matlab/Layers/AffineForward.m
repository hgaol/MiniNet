function [ Y, cache ] = AffineForward( X, W, b )
%
% author: hgaolbb
% version: beta 0.01
%
% X: [N, C, Hxw, Wxw]
% W: [F, C, Hxw, Wxw]
% b: [F, 1]
% because of affine layer, W and X must have the same HH and WW
% output Y: [N, F, 1, 1]
% example: x = rand(100,3,2,2);w = rand(5,3,2,2);b = rand(5,1);

cache.X = X;
cache.W = W;
cache.b = b;
N= size(X,1);
F = size(W,1);
% X = permute(X, [3,4,2,1,]);
% W = permute(W, [3,4,2,1,]);
% Y = zeros([N,F]);
% for i = 1:N
%     for f = 1:F
%         Y(i,f) = sum(sum(sum(X(:,:,:,i) .* W(:,:,:,f)))) + b(f);
%     end
% end
X = reshape(X, N, []);
W = reshape(W, F, []);
Y = X * W' + repmat(b', [N, 1]);

end

