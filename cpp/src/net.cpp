/*!
*  Copyright (c) 2015 by hgaolbb
* \file net.cpp
* \brief net
*/

#include "../include/net.hpp"

namespace mini_net {

void Net::sampleNet(double reg) {
    /*! forward */
    int n = layers_type_.size();
    for (int i = 0; i < n-1; ++i) {
        std::string lname = layers_type_[i];
        shared_ptr<Blob> out;
        if (lname == "Conv")
            ConvLayer::forward(data_[layers_[i]], out, params_[layers_[i]]);
        if (lname == "Pool")
            PoolLayer::forward(data_[layers_[i]], out, params_[layers_[i]]);
        if (lname == "Fc")
            AffineLayer::forward(data_[layers_[i]], out);
        if (lname == "Relu")
            ReluLayer::forward(data_[layers_[i]], out);
        data_[layers_[i+1]][0] = out;
    }

    /*! calc loss */
    std::string loss_type = layers_type_.back();
    shared_ptr<Blob> dout;
    if (loss_type == "SVM")
        SVMLossLayer::go(data_[layers_.back()], loss_, dout);
    if (loss_type == "Softmax")
        SoftmaxLossLayer::go(data_[layers_.back()], loss_, dout);
    grads_[layers_.back()].push_back(dout);

    /*! backward */
    for (int i = n-2; i >= 0; --i) {
        std::string lname = layers_type_[i];
        if (lname == "Conv")
            ConvLayer::backward(grads_[layers_[i+1]][0], data_[lname], grads_[layers_[i]], params_[layers_[i]]);
        if (lname == "Pool")
            PoolLayer::backward(grads_[layers_[i+1]][0], data_[lname], grads_[layers_[i]], params_[layers_[i]]);
        if (lname == "Fc")
            AffineLayer::backward(grads_[layers_[i+1]][0], data_[lname], grads_[layers_[i]]);
        if (lname == "Relu")
            ReluLayer::backward(grads_[layers_[i+1]][0], data_[lname], grads_[layers_[i]]);
    }

    /*! regularition */
    double reg_loss = 0;
    for (auto i : layers_) {
        if (grads_[i][1]) {
            /* it's ok? */
            (*grads_[i][1]) = (*grads_[i][1]) + reg * (*data_[i][1]);
            reg_loss += data_[i][1]->sum();
        }
    }
    reg_loss *= reg * 0.5;
    loss_ = loss_ + reg_loss;

    return;
}

void Net::sampleTestNet() {
    
}

void Net::sampleInitNet() {

    layers_.push_back("conv1");
    layers_.push_back("relu1");
    layers_.push_back("pool1");
    layers_.push_back("fc1");
    layers_type_.push_back("Conv");
    layers_type_.push_back("Relu");
    layers_type_.push_back("Pool");
    layers_type_.push_back("Fc");
    layers_type_.push_back("SoftmaxLoss");

    for (auto i : layers_) {
        data_[i] = vector<shared_ptr<Blob>>(3);
    }
    /*! y */
    mat aa = randi<mat>(2, 1, distr_param(0, 9));
    mat bb(10, 10, fill::zeros);
    for (int i = 0; i < 10; ++i) {
        bb(i, (uword)aa(i, 0)) = 1;
    }
    mat2Blob(bb, ground_truth_, 10, 1, 1);
    /*! conv1 layer data */
    /*! x */
    data_["conv1"][0].reset(new Blob(2,2,16,16,TRANDN));
    /*! w1 */
    data_["conv1"][1].reset(new Blob(5,2,3,3,TRANDN));
    /*! b1 */
    data_["conv1"][2].reset(new Blob(5,1,1,1,TZEROS));

    /*! affine layer */
    /*! w1 */
    data_["fc1"][1].reset(new Blob(10,5,8,8,TRANDN));
    /*! b1 */
    data_["fc1"][2].reset(new Blob(10,1,1,1,TZEROS));

    /*! params */
    params_["conv1"].setConvParam(1,1);
    params_["pool1"].setPoolParam(2, 2, 2);

    return;
}

} //namespace mini_net
