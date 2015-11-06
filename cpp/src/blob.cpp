/*!
*  Copyright (c) 2015 by hgaolbb
* \file blob.cpp
* \brief all layers implement
*/

#include "../include/blob.hpp"

namespace mini_net {

Blob* operator+(Blob& A, const double num) {
    Blob *pB = new Blob(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        (*pB)[i] = A[i] + num;
    }
    return pB;
}
Blob* operator+(const double num, Blob& A) {
    Blob *pB = new Blob(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        (*pB)[i] = A[i] + num;
    }
    return pB;
}
Blob* operator+(Blob& A, Blob& B) {
    vector<int> sz_A = A.size();
    vector<int> sz_B = B.size();
    for (int i = 0; i < 4; ++i) {
        if (sz_A[i] != sz_B[i])
            return NULL;
    }
    Blob *pC = new Blob(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        (*pC)[i] = A[i] + B[i];
    }
    return pC;
}
Blob* operator-(Blob& A, const double num) {
    Blob *pB = new Blob(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        (*pB)[i] = A[i] - num;
    }
    return pB;
}
Blob* operator-(const double num, Blob& A) {
    Blob *pB = new Blob(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        (*pB)[i] = num - A[i] ;
    }
    return pB;
}
Blob* operator-(Blob& A, Blob& B) {
    vector<int> sz_A = A.size();
    vector<int> sz_B = B.size();
    for (int i = 0; i < 4; ++i) {
        if (sz_A[i] != sz_B[i])
            return NULL;
    }
    Blob *pC = new Blob(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        (*pC)[i] = A[i] - B[i];
    }
    return pC;
}
Blob* operator*(Blob& A, const double num) {
    Blob *pB = new Blob(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        (*pB)[i] = A[i] * num;
    }
    return pB;
}
Blob* operator*(const double num, Blob& A) {
    Blob *pB = new Blob(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        (*pB)[i] = A[i] * num;
    }
    return pB;
}
Blob* operator*(Blob& A, Blob& B) {
    vector<int> sz_A = A.size();
    vector<int> sz_B = B.size();
    for (int i = 0; i < 4; ++i) {
        if (sz_A[i] != sz_B[i])
            return NULL;
    }
    Blob *pC = new Blob(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        (*pC)[i] = A[i] % B[i];
    }
    return pC;
}
Blob* operator/(Blob& A, const double num) {
    Blob *pB = new Blob(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        (*pB)[i] = A[i] / num;
    }
    return pB;
}
Blob* operator/(const double num, Blob& A) {
    Blob *pB = new Blob(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        (*pB)[i] = num / A[i];
    }
    return pB;
}
Blob* operator/(Blob& A, Blob& B) {
    vector<int> sz_A = A.size();
    vector<int> sz_B = B.size();
    for (int i = 0; i < 4; ++i) {
        if (sz_A[i] != sz_B[i])
            return NULL;
    }
    Blob *pC = new Blob(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        (*pC)[i] = A[i] / B[i];
    }
    return pC;
}

// convertion
void mat2Blob(mat& mA, Blob** out, int c, int h, int w) {
    int n = mA.n_rows;
    assert(mA.n_cols == c*h*w);

    mA = mA.t();
    if (*out) {
        delete *out;
        *out = NULL;
    }
    *out = new Blob(n, c, h, w);
    for (int i = 0; i < n; ++i) {
        //mat a = arma::reshape(mA.row(i), h, w, c);
        //cube ca = cube(mA.colptr(i), h, w, c);
        //cout << ca << endl;
        (**out)[i] = cube(mA.colptr(i), h, w, c);
    }
    return;
}
void mat2Blob(mat& mA, Blob** out, const vector<int>& sz) {
    int n = mA.n_rows;
    int c = sz[1];
    int h = sz[2];
    int w = sz[3];
    assert(mA.n_cols == c*h*w);

    mA = mA.t();
    if (*out) {
        delete *out;
        *out = NULL;
    }
    *out = new Blob(n, c, h, w);
    for (int i = 0; i < n; ++i) {
        //mat a = arma::reshape(mA.row(i), h, w, c);
        //cube ca = cube(mA.colptr(i), h, w, c);
        //cout << ca << endl;
        (**out)[i] = cube(mA.colptr(i), h, w, c);
    }
    return;
}

// += -= *= /=
Blob& Blob::operator+=(const double num) {
    for (int i=0; i<N_; ++i) {
        (*data_)[i] = (*data_)[i] + num;
    }
    return *this;
}
Blob& Blob::operator-=(const double num) {
    for (int i=0; i<N_; ++i) {
        (*data_)[i] = (*data_)[i] - num;
    }
    return *this;
}
Blob& Blob::operator*=(const double num) {
    for (int i=0; i<N_; ++i) {
        (*data_)[i] = (*data_)[i] * num;
    }
    return *this;
}
Blob& Blob::operator/=(const double num) {
    for (int i=0; i<N_; ++i) {
        (*data_)[i] = (*data_)[i] / num;
    }
    return *this;
}

//---Blob---
Blob::Blob(const int n, const int c, const int h, const int w, int type) :
        N_(n), C_(c), H_(h), W_(w) {
    if (type == TNONE)  data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::none));
    if (type == TONES)  data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::ones));
    if (type == TZEROS) data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::zeros));
    if (type == TRANDU) data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::randu));
    if (type == TRANDN) data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::randn));
    if (type == TDEFAULT) data_ = new vector<cube>(N_, cube(H_, W_, C_));
    return;
}
Blob::Blob(const int n, const int c, const int h, const int w, const double eps):
        N_(n), C_(c), H_(h), W_(w) {
    data_ = new vector<cube>(N_, cube(H_, W_, C_, fill::randn) * eps);
    return;
}
Blob::Blob(const vector<int>& shape) :
        N_(shape[0]), C_(shape[1]), H_(shape[2]), W_(shape[3]) {
    data_ = new vector<cube>(N_, cube(H_, W_, C_));
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

vector<int> Blob::size() {
    vector<int> shape{ N_, C_, H_, W_ };
    return shape;   
}

vector<cube>* Blob::get_data() {
    return data_;
}

mat Blob::reshape() {
    cube dst;
    for (int i = 0; i < N_; ++i) {
        dst = join_slices(dst, (*data_)[i]);
    }
    return arma::reshape(vectorise(dst), N_, C_*H_*W_);
}

} // namespace mini_net
