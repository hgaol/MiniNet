#include "../include/blob.hpp"

namespace mini_net {

//---Blob---
template<typename fill_type>
Blob::Blob(const int n, const int c, const int h, const int w,
		const fill::fill_class<fill_type>& ftype, const double eps):
		N_(n), C_(c), H_(h), W_(w) {
	if (ftype == fill::none) {
		data_ = new vector<cube>(N_, cube(H_, W_, C_, ftype));
	}
	else {
		data_ = new vector<cube>(N_, cube(H_, W_, C_, ftype) * eps);
	}
	return;
}

template<typename fill_type>
Blob::Blob(const vector<int>& shape,
		const fill::fill_class<fill_type>& ftype, const double eps) :
		N_(shape[0]), C_(shape[1]), H_(shape[2]), W_(shape[3]) {
	if (ftype == fill::none) {
		data_ = new vector<cube>(N_, cube(H_, W_, C_, ftype));
	}
	else {
		data_ = new vector<cube>(N_, cube(H_, W_, C_, ftype) * eps);
	}
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
