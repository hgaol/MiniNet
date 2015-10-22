function [ Y, cache ] = DropoutForward( X, dropout_param )
%
% author: hgaolbb
% version: beta 0.01
%
% X: [N, C, HH, WW]
% Y: [N, C, HH, WW]
% dropout_param:
%               mode: train/test
%               p: default 0.5
%               seed
% 

if ~exist('dropout_param', 'var')
    dropout_param.p = 0.5;
    dropout_param.mode = 'train';
    dropout_param.seed = 123;
end
if ~isfield(dropout_param, 'p')
    dropout_param.p = 0.5;
else
if ~isfield(dropout_param, 'seed')
    dropout_param.seed = 123;
end

if (strcmp(dropout_param.mode, 'train'))
    rand('seed', dropout_param.seed);
    mask = (rand(size(X)) < dropout_param.p) / dropout_param.p;
    Y = X .* mask;
end
if (strcmp(dropout_param.mode, 'test'))
    Y = X;
end

cache.mask = mask;
cache.dropout_param = dropout_param;

end

