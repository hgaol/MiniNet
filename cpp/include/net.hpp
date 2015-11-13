/*!
*  Copyright (c) 2015 by hgaolbb
* \file net.hpp
* \brief 
*/

#ifndef MINI_NET_NET_HPP_
#define MINI_NET_NET_HPP_

#include "blob.hpp"
#include "layer.hpp"
#include "test.hpp"
#include <unordered_map>

using std::unordered_map;
using std::shared_ptr;

namespace mini_net {

struct TrainParam {
    /*
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
    */
    
};

struct NetParam {
    /*! methods of update net parameters, sgd/momentum/... */
    std::string update;
    /*! learning rate */
    double lr;
    double lr_decay;
    /*! momentum parameter */
    double momentum;
    int num_epochs;
    /*! whether use batch size */
    bool use_batch;
    int batch_size;
    /*! regulazation parameter */
    double reg;
    /*! whether test layers */
    bool test_net;
    /*! \brief acc_frequence, how many iterations to check val_acc and train_acc */
    int acc_frequence;

    vector<std::string> layers;
    vector<std::string> ltypes;
    unordered_map<std::string, Param> params;
};

class Net {

public:
    Net(){}

    /*! \brief forward and backward */
    void sampleNet(shared_ptr<Blob>& X, 
                   shared_ptr<Blob>& Y,
                   NetParam& param,
                   std::string mode = "fb");

    /*! \brief test if all layers are right, be careful set reg to 0 */
    void sampleTestNet(NetParam& param);

    /*! \brief set input data and ground truth */
    void sampleInitNet(NetParam& param,
                       shared_ptr<Blob>& X,
                       shared_ptr<Blob>& Y);

    /*! \brief train the net */
    void sampleTrain(NetParam& param);

    //void sampleInitData();

    /*! test num_grads of lnum th layer */
    void testLayer(NetParam& param, int lnum);

private:
    void _test_fc_layer(vector<shared_ptr<Blob>>& in,
                        vector<shared_ptr<Blob>>& grads,
                        shared_ptr<Blob>& dout); 
    void _test_conv_layer(vector<shared_ptr<Blob>>& in,
                         vector<shared_ptr<Blob>>& grads,
                         shared_ptr<Blob>& dout,
                         Param& param); 
    void _test_pool_layer(vector<shared_ptr<Blob>>& in,
                         vector<shared_ptr<Blob>>& grads,
                         shared_ptr<Blob>& dout,
                         Param& param); 
    void _test_relu_layer(vector<shared_ptr<Blob>>& in,
                         vector<shared_ptr<Blob>>& grads,
                         shared_ptr<Blob>& dout); 
    void _test_dropout_layer(vector<shared_ptr<Blob>>& in,
                         vector<shared_ptr<Blob>>& grads,
                         shared_ptr<Blob>& dout,
                         Param& param); 
    void _test_svm_layer(vector<shared_ptr<Blob>>& in,
                         shared_ptr<Blob>& dout); 
    void _test_softmax_layer(vector<shared_ptr<Blob>>& in,
                         shared_ptr<Blob>& dout); 

    vector<std::string> layers_;
    vector<std::string> ltype_;
    double loss_;
    // train data set
    shared_ptr<Blob> X_;
    shared_ptr<Blob> Y_;
    // val data set
    shared_ptr<Blob> X_val_;
    shared_ptr<Blob> Y_val_;
    
    unordered_map<std::string, vector<shared_ptr<Blob>>> data_;
    unordered_map<std::string, vector<shared_ptr<Blob>>> grads_;
    unordered_map<std::string, vector<shared_ptr<Blob>>> num_grads_;

    /*! train or test */
    std::string type_;
    /*! train result */
    vector<double> loss_history_;
    vector<double> train_acc_history_;
    vector<double> val_acc_history_;
    /*! step cache */
    unordered_map<std::string, vector<shared_ptr<Blob>>> step_cache_;

}; // class Net

} // namespace mini_net

#endif