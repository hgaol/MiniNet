/*!
*  Copyright (c) 2015 by hgaolbb
* \file layer.hpp
* \brief all layers implement
*/

#ifndef MINI_NET_LAYER_HPP_
#define MINI_NET_LAYER_HPP_

#include "blob.hpp"
using std::vector;

namespace mini_net {

/*!
 * \brief Affine Layer
 */
class AffineLayer {
public:
    AffineLayer() {}
    ~AffineLayer() {}
    /*!
    * \brief forward
    *             X:        [N, C, Hx, Wx]
    *             weight:   [F, C, Hw, Ww]
    *             bias:     [F, 1, 1, 1]
    *             out:      [N, F, 1, 1]
    * \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:weights, in[2]:bias
    * \param[in]  const Param* param            params
    * \param[out] Blob& out                     Y
    */
    static void forward(const vector<Blob*>& in, Blob** out);

    /*!
    * \brief backward
    *             in:       [N, C, Hx, Wx]
    *             weight:   [F, C, Hw, Ww]
    *             bias:     [F, 1, 1, 1]
    *             dout:     [N, F, 1, 1]
    * \param[in]  const Blob* dout              dout
    * \param[in]  const vector<Blob*>& cache    cache[0]:X, cache[1]:weights, cache[2]:bias
    * \param[out] vector<Blob*>& grads          grads[0]:dX, grads[1]:dW, grads[2]:db
    */
    static void backward(Blob* dout, const vector<Blob*>& cache, vector<Blob*>& grads);
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
