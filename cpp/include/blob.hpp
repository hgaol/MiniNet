#ifndef MINI_NET_BLOB_HPP_
#define MINI_NET_BLOB_HPP_

#include <armadillo>
#include <vector>

using std::vector;
using namespace arma;

namespace mini_net {

class Blob {

public:
	Blob() : N_(0), C_(0), H_(0), W_(0), data_(NULL) {}
	template<typename fill_type>
	explicit Blob(const int n, const int c, const int h, const int w,
					const fill::fill_class<fill_type>& ftype = fill::none, const double eps = 1e-3);
	template<typename fill_type>
	explicit Blob(const vector<int>& shape, 
					const fill::fill_class<fill_type>& ftype = fill::none, const double eps = 1e-3);
	~Blob();

	cube& operator[] (int i);

	// return [N,C,H,W]
	vector<int> get_shape_vec();

	inline int get_N() {
		return N_;
	}
	inline int get_C() {
		return C_;
	}
	inline int get_H() {
		return H_;
	}
	inline int get_W() {
		return W_;
	}

	// return data_
	vector<cube>* get_data();

	//@brief: reshape [N,C,H,W] to [N,C*H*W], it will increase efficiency. [TODO]
	cube& reshape();

private:
	int N_;
	int C_;
	int H_;
	int W_;
	vector<cube> *data_;
};

// struct Param
struct Param {
	// conv param
	int stride;
	int pad;
	int width;
	int height;
};

} // namespace MiniNet

#endif // MINI_NET_BLOB_