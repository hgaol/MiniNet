#ifndef MINI_NET_BLOB_HPP_
#define MINI_NET_BLOB_HPP_

#include <opencv2/core/core.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include <map>

using std::vector;
using std::map;
using cv::Mat;

namespace mini_net {

class Blob;
//to use blob * a
Blob* operator+ (Blob& A, const double a);
Blob* operator+ (const double a, Blob& A);
// to use blob_A * blob_B
Blob* operator+ (Blob& A, Blob& B);

//to use blob * a
Blob* operator- (Blob& A, const double a);
Blob* operator- (const double a, Blob& A);
// to use blob_A * blob_B
Blob* operator- (Blob& A, Blob& B);

//to use blob * a
Blob* operator* (Blob& A, const double a);
Blob* operator* (const double a, Blob& A);
// to use blob_A * blob_B
Blob* operator* (Blob& A, Blob& B);

//to use blob * a
Blob* operator/ (Blob& A, const double a);
Blob* operator/ (const double a, Blob& A);
// to use blob_A * blob_B
Blob* operator/ (Blob& A, Blob& B);

// sum(A[a_th] + B[b_th])
double sum_blob_ith(Blob* A, int a, Blob* B, int b);

class Blob {

public:
	Blob() : N_(0), C_(0), H_(0), W_(0), data_(NULL) {}
	explicit Blob(const int n, const int c, const int h, const int w, const double val=0);
	explicit Blob(const vector<int>& shape, const double val=0);
	~Blob();

	//to use blob * a
	friend Blob* operator+ (const Blob& A, const double a);
	friend Blob* operator+ (const double a, const Blob& A);
	// to use blob_A * blob_B
	friend Blob* operator+ (const Blob& A, const Blob& B);

	//to use blob * a
	friend Blob* operator- (const Blob& A, const double a);
	friend Blob* operator- (const double a, const Blob& A);
	// to use blob_A * blob_B
	friend Blob* operator- (const Blob& A, const Blob& B);

	//to use blob * a
	friend Blob* operator* (const Blob& A, const double a);
	friend Blob* operator* (const double a, const Blob& A);
	// to use blob_A * blob_B
	friend Blob* operator* (const Blob& A, const Blob& B);

	//to use blob * a
	friend Blob* operator/ (const Blob& A, const double a);
	friend Blob* operator/ (const double a, const Blob& A);
	// to use blob_A * blob_B
	friend Blob* operator/ (const Blob& A, const Blob& B);
	
	// to use blob[i][j] operation
	vector<Mat>& operator[] (int i);

	// ------- get member variable -------
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
	vector<vector<Mat> >* get_data();

	//@brief: reshape [N,C,H,W] to [N,C*H*W], it will increase efficiency. [TODO]
	Mat& reshape();

private:
	int N_;
	int C_;
	int H_;
	int W_;
	vector<vector<Mat> > *data_;
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