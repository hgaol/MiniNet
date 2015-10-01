#include "../include/blob.hpp"

namespace mini_net {

Blob* operator+(Blob& A, const double num) {
	Blob *pB = new Blob(A.get_shape_vec());
	int N = A.get_N();
	for (int i = 0; i < N; ++i) {
		(*pB)[i] = A[i] + num;
	}
	return pB;
}
Blob* operator+(const double num, Blob& A) {
	Blob *pB = new Blob(A.get_shape_vec());
	int N = A.get_N();
	for (int i = 0; i < N; ++i) {
		(*pB)[i] = A[i] + num;
	}
	return pB;
}
Blob* operator+(Blob& A, Blob& B) {
	vector<int> sz_A = A.get_shape_vec();
	vector<int> sz_B = B.get_shape_vec();
	for (int i = 0; i < 4; ++i) {
		if (sz_A[i] != sz_B[i])
			return NULL;
	}
	Blob *pC = new Blob(A.get_shape_vec());
	int N = A.get_N();
	for (int i = 0; i < N; ++i) {
		(*pC)[i] = A[i] + B[i];
	}
	return pC;
}
Blob* operator-(Blob& A, const double num) {
	Blob *pB = new Blob(A.get_shape_vec());
	int N = A.get_N();
	for (int i = 0; i < N; ++i) {
		(*pB)[i] = A[i] - num;
	}
	return pB;
}
Blob* operator-(const double num, Blob& A) {
	Blob *pB = new Blob(A.get_shape_vec());
	int N = A.get_N();
	for (int i = 0; i < N; ++i) {
		(*pB)[i] = num - A[i] ;
	}
	return pB;
}
Blob* operator-(Blob& A, Blob& B) {
	vector<int> sz_A = A.get_shape_vec();
	vector<int> sz_B = B.get_shape_vec();
	for (int i = 0; i < 4; ++i) {
		if (sz_A[i] != sz_B[i])
			return NULL;
	}
	Blob *pC = new Blob(A.get_shape_vec());
	int N = A.get_N();
	for (int i = 0; i < N; ++i) {
		(*pC)[i] = A[i] - B[i];
	}
	return pC;
}
Blob* operator*(Blob& A, const double num) {
	Blob *pB = new Blob(A.get_shape_vec());
	int N = A.get_N();
	for (int i = 0; i < N; ++i) {
		(*pB)[i] = A[i] * num;
	}
	return pB;
}
Blob* operator*(const double num, Blob& A) {
	Blob *pB = new Blob(A.get_shape_vec());
	int N = A.get_N();
	for (int i = 0; i < N; ++i) {
		(*pB)[i] = A[i] * num;
	}
	return pB;
}
Blob* operator*(Blob& A, Blob& B) {
	vector<int> sz_A = A.get_shape_vec();
	vector<int> sz_B = B.get_shape_vec();
	for (int i = 0; i < 4; ++i) {
		if (sz_A[i] != sz_B[i])
			return NULL;
	}
	Blob *pC = new Blob(A.get_shape_vec());
	int N = A.get_N();
	for (int i = 0; i < N; ++i) {
		(*pC)[i] = A[i] % B[i];
	}
	return pC;
}
Blob* operator/(Blob& A, const double num) {
	Blob *pB = new Blob(A.get_shape_vec());
	int N = A.get_N();
	for (int i = 0; i < N; ++i) {
		(*pB)[i] = A[i] / num;
	}
	return pB;
}
Blob* operator/(const double num, Blob& A) {
	Blob *pB = new Blob(A.get_shape_vec());
	int N = A.get_N();
	for (int i = 0; i < N; ++i) {
		(*pB)[i] = num / A[i];
	}
	return pB;
}
Blob* operator/(Blob& A, Blob& B) {
	vector<int> sz_A = A.get_shape_vec();
	vector<int> sz_B = B.get_shape_vec();
	for (int i = 0; i < 4; ++i) {
		if (sz_A[i] != sz_B[i])
			return NULL;
	}
	Blob *pC = new Blob(A.get_shape_vec());
	int N = A.get_N();
	for (int i = 0; i < N; ++i) {
		(*pC)[i] = A[i] / B[i];
	}
	return pC;
}
//---Blob---
	
Blob::Blob(const int n, const int c, const int h, const int w, int type) :
		N_(n), C_(c), H_(h), W_(w) {
	if (type == TNONE)	data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::none));
	if (type == TONES)	data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::ones));
	if (type == TZEROS)	data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::zeros));
	if (type == TRANDU)	data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::randu));
	if (type == TRANDN)	data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::randn));
	return;
}
Blob::Blob(const int n, const int c, const int h, const int w, const double eps):
		N_(n), C_(c), H_(h), W_(w) {
	data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::randn) * eps);
	return;
}
Blob::Blob(const vector<int>& shape) :
		N_(shape[0]), C_(shape[1]), H_(shape[2]), W_(shape[3]) {
	data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::none));
	return;
}
Blob::Blob(const vector<int>& shape, const double eps) :
		N_(shape[0]), C_(shape[1]), H_(shape[2]), W_(shape[3]) {
	data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::randn) * eps);
	return;
}

Blob::~Blob() {
	if (data_) {
		delete data_;
	}
	return;
}

cube& Blob::operator[] (int i) {
	return (*data_)[i];
}

vector<int> Blob::get_shape_vec() {
	vector<int> shape{ N_, C_, H_, W_ };
	return shape;	
}

vector<cube>* Blob::get_data() {
	return data_;
}

} // namespace mini_net
