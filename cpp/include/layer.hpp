#ifndef MINI_NET_LAYER_HPP_
#define MINI_NET_LAYER_HPP_

#include <Eigen/Dense>
#include <algorithm>
#include <string>
#include <vector>
#include <map>

namespace mini_net {

/**
 Layer base class
 */
class Layer {
public:
	explicit Layer() {}
	virtual ~Layer() {}
	virtual void forward(const std::vector<const Eigen::MatrixXf&> in, const std::vector<Eigen::MatrixXf&> out) = 0;
	virtual void backward(const std::vector<const Eigen::MatrixXf&> in, const std::vector<Eigen::MatrixXf&> out) = 0;
};

/**
 Affine Layer
 */
class AffineLayer : public Layer {
public:
	explicit AffineLayer() {}
	virtual ~AffineLayer() {}
	virtual void forward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
	virtual void backward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
};

/**
 ReLU Layer
 */
class ReluLayer : public Layer {
public:
	explicit ReluLayer() {}
	virtual ~ReluLayer() {}
	virtual void forward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
	virtual void backward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
};

/**
 Softmax Loss Layer
 */
class SoftmaxLossLayer : public Layer {
public:
	explicit SoftmaxLossLayer() {}
	virtual ~SoftmaxLossLayer() {}
	virtual void forward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
	virtual void backward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
};

/**
 SVM Loss Layer
 */
class SVMLossLayer : public Layer {
public:
	explicit SVMLossLayer() {}
	virtual ~SVMLossLayer() {}
	virtual void forward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
	virtual void backward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
};

/**
 Convolution Layer
 */
class ConvolutionLayer : public Layer {
public:
	explicit ConvolutionLayer(const std::vector<std::map<std::string, int> > param_in);
	virtual ~ConvolutionLayer() {}
	virtual void forward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
	virtual void backward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
};

/**
 Pooling Layer
 */
class PoolingLayer : public Layer {
public:
	explicit PoolingLayer(const std::vector<std::map<std::string, int> > param_in);
	virtual ~PoolingLayer() {}
	virtual void forward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
	virtual void backward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
};

/**
 Dropout Layer
 */
class DropoutLayer : public Layer {
public:
	explicit DropoutLayer() {}
	virtual ~DropoutLayer() {}
	virtual void forward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
	virtual void backward(const std::vector<const Eigen::MatrixXf&>& in, std::vector<Eigen::MatrixXf&>& out);
};

} // namespace mini_net

#endif // MINI_NET_LAYER_
