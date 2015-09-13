

X = rand(100,3,4,4);
model.W1 = 1e-2 * random('norm',0,1,[10,3,4,4]);model.b1 = zeros(10,1);
y = randint(100,1,[1,10]);
% model.conv_param.pad = 1; model.conv_param.stride = 1;
% model.pool_param.stride = 2;model.pool_param.width = 2; model.pool_param.height = 2;
reg = 0;
lr = 100;
for i = 1:100000
    [scores, cache2] = AffineForward(X, model.W1, model.b1);
    [data_loss, dscores] = SoftmaxLossLayer(scores, y);
    
    if mod(i,100) == 0
        fprintf('iter: %d\tloss: %d\n', i, loss);
    end
    [a1, dW1, db1] = AffineBackward(dscores, cache2);
%     [dX, dW1, db1] = conv_relu_pool_backward(da1, cache1);

    % add regularization
    dW1 = dW1 + reg .* model.W1;
%     dW2 = dW2 + reg .* model.W2;
    reg_loss = 0.5 .* reg .* (sum(model.W1(:)));

    if mod(i, 1000)
        lr = lr * 0.99;
    end
    loss = data_loss + reg_loss;
    model.W1 = model.W1 - lr * dW1;
    model.b1 = model.b1 - lr * db1;
end