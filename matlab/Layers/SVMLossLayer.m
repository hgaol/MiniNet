function [ loss, dx ] = SVMLossLayer( X, y )
%
% author: hgaolbb
% version: beta 0.01
%
% X: [N, C]
% Y: [N, C], where the ground truth is 1, others 0.
% \Delta is set to 1 in this svm loss layer.
% for more details, please visit ==> http://hgaolbb.github.io/%5Bcs231n%5D%5B7%5DLayers%E7%9A%84forward&backward.html
% for test, you can use 
% x = random('norm', 0, 1, [100, 10]);y = zeros([100,10]); y(:,5) = 1;
% answer should near N * \Delta;

Y = zeros(size(X));
for i = 1:size(X, 1)
    Y(i,y(i)) = 1;
end

dx = zeros(size(X));
delta = 1;
%% forward
[N, C] = size(X);
right_score = repmat(sum(X .* Y, 2), [1, C]);
loss_mat = X - right_score + delta;
loss_mat(Y == 1) = 0;

loss = sum(sum(max(0, loss_mat))) ./ N;
%% backward
dx(loss_mat > 0) = 1;
dx_yi = -repmat(sum(loss_mat > 0, 2), [1, C]) .* Y;
dx = (dx + dx_yi) ./ N;
end

