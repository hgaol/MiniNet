#ifndef MINI_NET_NET_HPP_
#define MINI_NET_NET_HPP_

#include "layer.h"

namespace mini_net {

class Net {

public:
    explicit Net(const NetParam& net_param);
    explicit Net();
    explicit ~Net();
    void Train();
    void ForwardInit(const NetParam& net_param);
    Eigen::MatrixXf ForwardAll();
};

struct NetParam {
    int num_epochs;
    int reg;
    int momentum;
    int lr;
    int lr_decay;
    // TODO
};

}   // namespace mini_net

#endif
