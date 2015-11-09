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

class Net {

public:
    void sampleNet();

    vector<std::string> layers_;
    unordered_map<std::string, vector<shared_ptr<Blob>>> data_;
    unordered_map<std::string, vector<shared_ptr<Blob>>> grads_;
    unordered_map<std::string, vector<shared_ptr<Blob>>> num_grads_;
    unordered_map<std::string, unordered_map<std::string, Param>> param_;

}; // class Net

} // namespace mini_net

#endif