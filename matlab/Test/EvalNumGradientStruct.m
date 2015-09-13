function [num_grid] = EvalNumGradientStruct(Func, x, epsilon)
%
% author: hgaolbb
% version: beta 0.01
%

if ~exist('epsilon', 'var')
    epsilon = 1e-6;
end

param_name = fieldnames(x);
% for num_name = 1:size(param_name)
for num_name = 1:4
    
dname = param_name(num_name);
dname = dname{1,1};
    
[N, C, H, W] = size(x.(dname));
num_grid.(dname) = zeros(size(x.(dname)));

for i = 1:N
    for c = 1:C
        for h = 1:H
            for w = 1:W
                old_val = x.(dname)(i,c,h,w);
                x.(dname)(i,c,h,w) = old_val + epsilon;
                fp = Func(x);
                x.(dname)(i,c,h,w) = old_val - epsilon;
                fm = Func(x);
                x.(dname)(i,c,h,w) = old_val;
                num_grid.(dname)(i,c,h,w) = (fp - fm) ./ (2 .* epsilon);
            end
        end
    end
end
end

end