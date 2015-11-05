/*!
*  Copyright (c) 2015 by hgaolbb
* \file layer.hpp
* \brief all layers implement
*/

#ifndef MINI_NET_LAYER_HPP_
#define MINI_NET_LAYER_HPP_

#include "blob.hpp"
#include <map>

using std::vector;
//using std::map;

namespace mini_net {

/**
 Affine Layer
 */
class AffineLayer {
public:
    AffineLayer() {}
    ~AffineLayer() {}
    static void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
    static void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*>& grads, const Param* param = NULL);
};
//
///**
// ReLU Layer
// */
//class ReluLayer {
//public:
//  explicit ReluLayer() {}
//  virtual ~ReluLayer() {}
//  static void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
//  static void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL);
//};
//
///**
// Softmax Loss Layer
// */
//class SoftmaxLossLayer : public Layer {
//public:
//  explicit SoftmaxLossLayer() {}
//  virtual ~SoftmaxLossLayer() {}
//  virtual void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
//  virtual void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL);
//};
//
///**
// SVM Loss Layer
// */
//class SVMLossLayer : public Layer {
//public:
//  explicit SVMLossLayer() {}
//  virtual ~SVMLossLayer() {}
//  virtual void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
//  virtual void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL);
//};
//
///**
// Convolution Layer
// */
//class ConvolutionLayer : public Layer {
//public:
//  explicit ConvolutionLayer(const std::vector<std::map<std::string, int> > param_in);
//  virtual ~ConvolutionLayer() {}
//  virtual void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
//  virtual void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL);
//};
//
///**
// Pooling Layer
// */
//class PoolingLayer : public Layer {
//public:
//  explicit PoolingLayer(const std::vector<std::map<std::string, int> > param_in);
//  virtual ~PoolingLayer() {}
//  virtual void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
//  virtual void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL);
//};
//
///**
// Dropout Layer
// */
//class DropoutLayer : public Layer {
//public:
//  explicit DropoutLayer() {}
//  virtual ~DropoutLayer() {}
//  virtual void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
//  virtual void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL);
//};
//
} // namespace mini_net

#endif // MINI_NET_LAYER_
