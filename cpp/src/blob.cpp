#include "blob.hpp"

namespace mini_net {

//@biref element multiply with scalar
Blob* operator* (Blob& A, const double a) {
	if (!A.get_data()) {
		return NULL;
	}
	vector<int> shape = A.get_shape_vec();
	Blob *pdst = new Blob(shape);
	for (int i = 0; i < shape[0]; ++i) {
		for (int j = 0; j < shape[1]; ++j) {
			(*pdst)[i][j] = A[i][j] * a;
		}
	}
	return pdst;
}

//@brief element multiply
Blob* operator* (const double a, Blob& A) {
	if (!A.get_data()) {
		return NULL;
	}
	vector<int> shape = A.get_shape_vec();
	Blob *pdst = new Blob(shape);
	for (int i = 0; i < shape[0]; ++i) {
		for (int j = 0; j < shape[1]; ++j) {
			(*pdst)[i][j] = A[i][j] * a;
		}
	}
	return pdst;
}

//@brief: element multiply with Blob
Blob* operator* (Blob& A, Blob& B) {
	if (!A.get_data() || !B.get_data()) {
		return NULL;
	}
	std::vector<int> shape_A = A.get_shape_vec();
	std::vector<int> shape_B = B.get_shape_vec();
	for (int i = 0; i < 4; ++i) {
		if (shape_A[i] != shape_B[i]) {
			printf("Blob size should be equal!\n");
			return NULL;
		}
	}
	Blob *pdst = new Blob(shape_A);
	for (int i = 0; i < shape_A[0]; ++i) {
		for (int j = 0; j < shape_A[1]; ++j) {
			(*pdst)[i][j] = A[i][j] * B[i][j];
		}
	}
	return pdst;
}

//@biref element multiply with scalar
Blob* operator+ (Blob& A, const double a) {
	if (!A.get_data()) {
		return NULL;
	}
	vector<int> shape = A.get_shape_vec();
	Blob *pdst = new Blob(shape);
	for (int i = 0; i < shape[0]; ++i) {
		for (int j = 0; j < shape[1]; ++j) {
			(*pdst)[i][j] = A[i][j] + a;
		}
	}
	return pdst;
}

//@brief element multiply
Blob* operator+ (const double a, Blob& A) {
	if (!A.get_data()) {
		return NULL;
	}
	vector<int> shape = A.get_shape_vec();
	Blob *pdst = new Blob(shape);
	for (int i = 0; i < shape[0]; ++i) {
		for (int j = 0; j < shape[1]; ++j) {
			(*pdst)[i][j] = A[i][j] + a;
		}
	}
	return pdst;
}

//@brief: element multiply with Blob
Blob* operator+ (Blob& A, Blob& B) {
	if (!A.get_data() || !B.get_data()) {
		return NULL;
	}
	std::vector<int> shape_A = A.get_shape_vec();
	std::vector<int> shape_B = B.get_shape_vec();
	for (int i = 0; i < 4; ++i) {
		if (shape_A[i] != shape_B[i]) {
			printf("Blob<Dtype> size should be equal!\n");
			return NULL;
		}
	}
	Blob *pdst = new Blob(shape_A);
	for (int i = 0; i < shape_A[0]; ++i) {
		for (int j = 0; j < shape_A[1]; ++j) {
			(*pdst)[i][j] = A[i][j] + B[i][j];
		}
	}
	return pdst;
}

//@biref element multiply with scalar
Blob* operator- (Blob& A, const double a) {
	if (!A.get_data()) {
		return NULL;
	}
	vector<int> shape = A.get_shape_vec();
	Blob *pdst = new Blob(shape);
	for (int i = 0; i < shape[0]; ++i) {
		for (int j = 0; j < shape[1]; ++j) {
			(*pdst)[i][j] = A[i][j] - a;
		}
	}
	return pdst;
}

//@brief element multiply
Blob* operator- (const double a, Blob& A) {
	if (!A.get_data()) {
		return NULL;
	}
	vector<int> shape = A.get_shape_vec();
	Blob *pdst = new Blob(shape);
	for (int i = 0; i < shape[0]; ++i) {
		for (int j = 0; j < shape[1]; ++j) {
			(*pdst)[i][j] = a - A[i][j];
		}
	}
	return pdst;
}

//@brief: element multiply with Blob
Blob* operator- (Blob& A, Blob& B) {
	if (!A.get_data() || !B.get_data()) {
		return NULL;
	}
	std::vector<int> shape_A = A.get_shape_vec();
	std::vector<int> shape_B = B.get_shape_vec();
	for (int i = 0; i < 4; ++i) {
		if (shape_A[i] != shape_B[i]) {
			printf("Blob<Dtype> size should be equal!\n");
			return NULL;
		}
	}
	Blob *pdst = new Blob(shape_A);
	for (int i = 0; i < shape_A[0]; ++i) {
		for (int j = 0; j < shape_A[1]; ++j) {
			(*pdst)[i][j] = A[i][j] - B[i][j];
		}
	}
	return pdst;
}

//@biref element multiply with scalar
Blob* operator/ (Blob& A, const double a) {
	if (!A.get_data()) {
		return NULL;
	}
	vector<int> shape = A.get_shape_vec();
	Blob *pdst = new Blob(shape);
	for (int i = 0; i < shape[0]; ++i) {
		for (int j = 0; j < shape[1]; ++j) {
			(*pdst)[i][j] = A[i][j] / a;
		}
	}
	return pdst;
}

//@brief element multiply
Blob* operator/ (const double a, Blob& A) {
	if (!A.get_data()) {
		return NULL;
	}
	vector<int> shape = A.get_shape_vec();
	Blob *pdst = new Blob(shape);
	for (int i = 0; i < shape[0]; ++i) {
		for (int j = 0; j < shape[1]; ++j) {
			(*pdst)[i][j] = a / A[i][j];
		}
	}
	return pdst;
}

//@brief: element multiply with Blob
Blob* operator/ (Blob& A, Blob& B) {
	if (!A.get_data() || !B.get_data()) {
		return NULL;
	}
	std::vector<int> shape_A = A.get_shape_vec();
	std::vector<int> shape_B = B.get_shape_vec();
	for (int i = 0; i < 4; ++i) {
		if (shape_A[i] != shape_B[i]) {
			printf("Blob<Dtype> size should be equal!\n");
			return NULL;
		}
	}
	Blob *pdst = new Blob(shape_A);
	for (int i = 0; i < shape_A[0]; ++i) {
		for (int j = 0; j < shape_A[1]; ++j) {
			(*pdst)[i][j] = A[i][j] / B[i][j];
		}
	}
	return pdst;
}

double sum_blob_ith(Blob* A, int a, Blob* B, int b) {
	int C = A->get_C();
	double ret = 0;
	for (int c = 0; c < C; ++c) {
		ret += sum((*B)[b][c] + (*A)[a][c])[0];
	}
	return ret;
}

// -----------Blob space-------------

Blob::Blob(const int n, const int c, const int h, const int w, const double val) :
		N_(n), C_(c), H_(h), W_(w) {
	data_ = new vector<vector<Mat> >(N_,vector<Mat>(C_,Mat(H_, W_, CV_32F, cvScalar(val))));
	return;
}

Blob::Blob(const vector<int>& shape, const double val) :
	N_(shape[0]), C_(shape[1]), H_(shape[2]), W_(shape[3]) {
	data_ = new vector<vector<Mat> >(N_,vector<Mat>(C_,Mat(H_, W_, CV_32F, cvScalar(val))));
	return;
}

Blob::~Blob() {
	if (!data_) {
		delete data_;
	}
	return;
}

vector<Mat>& Blob::operator[] (int i) {
	return (*data_)[i];
}

vector<int> Blob::get_shape_vec() {
	vector<int> shape{ N_, C_, H_, W_ };
	return shape;	
}

vector<vector<Mat> >* Blob::get_data() {
	return data_;
}

} // namespace mini_net
