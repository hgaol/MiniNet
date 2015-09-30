#ifndef MINI_NET_LAYER_HPP_
#define MINI_NET_LAYER_HPP_

#include "blob.hpp"
#include <opencv2/core/core.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include <map>

using std::vector;
using std::map;
using cv::Mat;

namespace mini_net {

/**
 Layer base class
 */
class Layer {
public:
	explicit Layer() {}
	virtual ~Layer() {}

	//@brief forward
	//@param[in]  const vector<Blob*>& in		in[0]:X, in[1]:weights, in[2]:bias 
	//@param[out] Blob* out						Y 
	//@param[in]  const Param* param			params		
	virtual void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL) = 0;

	//@brief backward
	//@param[in]  const Blob* dout				dout, backpro value 
	//@param[in]  const vector<Blob*>& cache 	just use vector<Blob*>& in when forward
	//@param[out] vector<Blob*> grads			grads{X, weights, bias}	
	//@param[in]  const Param* param			param
	virtual void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL) = 0;
};

/**
 Affine Layer
 */
class AffineLayer : public Layer {
public:
	explicit AffineLayer() {}
	virtual ~AffineLayer() {}
	virtual void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
	virtual void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL);
};

/**
 ReLU Layer
 */
class ReluLayer : public Layer {
public:
	explicit ReluLayer() {}
	virtual ~ReluLayer() {}
	virtual void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
	virtual void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL);
};

/**
 Softmax Loss Layer
 */
class SoftmaxLossLayer : public Layer {
public:
	explicit SoftmaxLossLayer() {}
	virtual ~SoftmaxLossLayer() {}
	virtual void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
	virtual void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL);
};

/**
 SVM Loss Layer
 */
class SVMLossLayer : public Layer {
public:
	explicit SVMLossLayer() {}
	virtual ~SVMLossLayer() {}
	virtual void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
	virtual void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL);
};

/**
 Convolution Layer
 */
class ConvolutionLayer : public Layer {
public:
	explicit ConvolutionLayer(const std::vector<std::map<std::string, int> > param_in);
	virtual ~ConvolutionLayer() {}
	virtual void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
	virtual void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL);
};

/**
 Pooling Layer
 */
class PoolingLayer : public Layer {
public:
	explicit PoolingLayer(const std::vector<std::map<std::string, int> > param_in);
	virtual ~PoolingLayer() {}
	virtual void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
	virtual void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL);
};

/**
 Dropout Layer
 */
class DropoutLayer : public Layer {
public:
	explicit DropoutLayer() {}
	virtual ~DropoutLayer() {}
	virtual void forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL);
	virtual void backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL);
};

} // namespace mini_net

#endif // MINI_NET_LAYER_