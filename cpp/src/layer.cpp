#include "layer.hpp"

using Eigen::MatrixXf;

namespace mini_net {

	void AffineLayer::forward(const MatrixXf& x, const MatrixXf& w, MatrixXf& y) {
		y = x * w;
		return ;
	}

	void AffineLayer::backward(const Eigen::MatrixXf& dy, const Eigen::MatrixXf& x, const Eigen::MatrixXf& w,
		Eigen::MatrixXf& dx, Eigen::MatrixXf& dw) {
		dx = dy * w.transpose();
		dw = x.transpose() * dy;
		return ;
	}
	
	void SoftmaxLossLayer::forward(const Eigen::MatrixXf& x, const Eigen::MatrixXf& w, Eigen::MatrixXf& y) {

	}

//TODO

} //namespace mini_net
