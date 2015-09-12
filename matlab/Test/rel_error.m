function [err] = rel_error(x, y)
% 只要符号相反，就会得到1

err = max(abs(x(:) - y(:)) ./ (max(abs(x(:)) + abs(y(:)), 1e-8)));

end

