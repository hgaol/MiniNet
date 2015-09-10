function [loss, dx] = SoftmaxLossLayer( X, y )
%
% author: hgaolbb
% version: beta 0.01
%
% X: [N, C]
% Y: [N, C], where the ground truth is 1, others 0.
% for more details, please visit
% for test, you can use 
% x = random('norm', 0, 1, [100, 10]);y = zeros([100,10]); y(:,5) = 1;
% loss should near -log(0.1) = 2.3026;

Y = zeros(size(X));
for i = 1:size(X, 1)
    Y(i,y(i)) = 1;
end
%% forward
[N, C] = size(X);
X = X - repmat(max(X, [], 2), [1, C]);
sum_row = repmat(sum(exp(X), 2), [1, C]);

% prob: [N, C]
prob = -log( exp(X) ./ sum_row);

% calc loss
loss = sum(sum(prob .* Y)) ./ N;

%% backward
dx = (prob - Y) ./ N;

end

