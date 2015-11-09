/*!
*  Copyright (c) 2015 by hgaolbb
* \file net.hpp
* \brief 
*/

#ifndef MINI_NET_NET_HPP_
#define MINI_NET_NET_HPP_

#include "blob.hpp"
#include "layer.hpp"
#include <unordered_map>

using std::unordered_map;
using std::shared_ptr;

namespace mini_net {

/*
struct NetParam {
    vector<std::string> layers_;
    unordered_map<std::string, Param> params_;

    NetParam() {
        layers_.push_back("conv1");
        params_["conv1"].setConvParam(1,1);
        layers_.push_back("relu1");
        layers_.push_back("pool1");
        params_["conv1"].setPoolParam(2,2,2);
        layers_.push_back("fc1");
    }
};
*/

class Net {

public:
    Net(){}
    void sampleNet(double reg);
    void sampleTestNet();
    void sampleInitNet();

    vector<std::string> layers_;
    vector<std::string> layers_type_;
    double loss_;
    unordered_map<std::string, vector<shared_ptr<Blob>>> data_;
    shared_ptr<Blob> ground_truth_;
    unordered_map<std::string, vector<shared_ptr<Blob>>> grads_;
    unordered_map<std::string, vector<shared_ptr<Blob>>> num_grads_;
    unordered_map<std::string, Param> params_;

}; // class Net

} // namespace mini_net

#endif