function [num_grid] = EvalNumGradientMat(Func, x, dy, epsilon)
%
% author: hgaolbb
% version: beta 0.01
%
% x = randn(10,2,3);
% w = randn(6,5);
% dout = randn(10,5);

if ~exist('epsilon', 'var')
    epsilon = 1e-6;
end

[N, C, H, W] = size(x);
num_grid = zeros(size(x));

for i = 1:N
    for c = 1:C
        for h = 1:H
            for w = 1:W
                old_val = x(i,c,h,w);
                x(i,c,h,w) = old_val + epsilon;
                fp = Func(x);
                x(i,c,h,w) = old_val - epsilon;
                fm = Func(x);
                x(i,c,h,w) = old_val;
                num_grid(i,c,h,w) = sum(sum(sum(sum(((fp - fm) .* dy) ./ (2 .* epsilon)))));
            end
        end
    end
end

end

