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
	
	// ------- overide operator ----------
	// to use blob[i][j] operation
	vector<Mat>& operator[] (int i);

	// ------- get member variable -------
	// return [N,C,H,W]
	vector<int> get_shape_vec();

	// return data_
	vector<vector<Mat> >* get_data();

	//@brief: reshape [N,C,H,W] to [N,C*H*W] 
	Mat& reshape();

private:
	int N_;
	int C_;
	int H_;
	int W_;
	vector<vector<Mat> > *data_;
};

} // namespace MiniNet

#endif // MINI_NET_BLOB_