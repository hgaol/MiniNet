/*!
* \file mnist.cpp
* \brief for mnist example
*/

#include "../include/mini_net.hpp"
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;
using namespace mini_net;
using namespace arma;

int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return ((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void ReadMnistLabel(string path, shared_ptr<Blob>& label) {
    ifstream file(path, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        for (int i=0; i<number_of_images; ++i)
        {
            unsigned char temp=0;
            file.read((char*)&temp, sizeof(temp));
            (*label)[i](0, 0, (int)temp) = 1;
        }
    }
}

void ReadMnistData(string path, shared_ptr<Blob>& image) {
    ifstream file(path, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for (int i=0; i<number_of_images; ++i)
        {
            for (int r=0; r<n_rows; ++r)
            {
                for (int c=0; c<n_cols; ++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp, sizeof(temp));
                    (*image)[i](r, c, 0) = (double)temp / 255;
                }
            }
        }
    }
}

void trainMnist(shared_ptr<Blob>& X, shared_ptr<Blob>& Y) {
    //NetParam param;
    //param.batch_size = 100;
    //param.lr = 0.1;
    //// momentum=0.9, lr_decay=0.99, lr=0.05
    //param.momentum = 0.9;
    //param.num_epochs = 500;
    ///*! when testing num_gradiets, reg must set to 0 */
    //param.reg = 0;
    //param.update = "momentum";
    //param.use_batch = true;
    //param.acc_frequence = 1;
    //param.lr_decay = 0.99;
    //param.acc_update_lr = true;

    //param.layers.push_back("conv1");
    ////param.params["conv1"].conv_width = 3;
    ////param.params["conv1"].conv_height = 3;
    ////param.params["conv1"].conv_pad = 1;
    ////param.params["conv1"].conv_stride = 1;
    ////param.params["conv1"].conv_kernels = 5;
    //param.params["conv1"].setConvParam(1,1,3,3,5);
    //param.layers.push_back("relu1");
    //param.layers.push_back("pool1");
    //param.params["pool1"].setPoolParam(2, 2, 2);
    //param.layers.push_back("fc1");
    //param.params["fc1"].fc_kernels = 10;
    //param.layers.push_back("softmax");
    //param.ltypes.push_back("Conv");
    //param.ltypes.push_back("Relu");
    //param.ltypes.push_back("Pool");
    //param.ltypes.push_back("Fc");
    //param.ltypes.push_back("Softmax");
    NetParam param;
    param.readNetParam("example//mnist.json");

    shared_ptr<Blob> X_train(new Blob(X->subBlob(0, 9000)));
    shared_ptr<Blob> Y_train(new Blob(Y->subBlob(0, 9000)));
    shared_ptr<Blob> X_val(new Blob(X->subBlob(9000, 9100)));
    shared_ptr<Blob> Y_val(new Blob(Y->subBlob(9000, 9100)));
    vector<shared_ptr<Blob>> XX{X_train, X_val};
    vector<shared_ptr<Blob>> YY{Y_train, Y_val};
    
    Net inst;
    inst.initNet(param, XX, YY);
    //inst.testNet(param);
    inst.train(param);
}

int main()
{
    shared_ptr<Blob> images(new Blob(10000,1,28,28));
    shared_ptr<Blob> labels(new Blob(10000,10,1,1,TZEROS));
    ReadMnistData("example//t10k-images.idx3-ubyte", images);
    ReadMnistLabel("example//t10k-labels.idx1-ubyte", labels);
    trainMnist(images, labels);

    return 0;
}