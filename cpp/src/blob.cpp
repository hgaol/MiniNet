/*!
*  Copyright (c) 2015 by hgaolbb
* \file blob.cpp
* \brief all layers implement
*/

#include "../include/blob.hpp"

namespace mini_net {

Blob operator+(Blob& A, const double num) {
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] + num;
    }
    return out;
}
Blob operator+(const double num, Blob& A) {
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] + num;
    }
    return out;
}
Blob operator+(Blob& A, Blob& B) {
    vector<int> sz_A = A.size();
    vector<int> sz_B = B.size();
    for (int i = 0; i < 4; ++i) {
        assert(sz_A[i] == sz_B[i]);
    }
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] + B[i];
    }
    return out;
}
Blob operator-(Blob& A, const double num) {
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] - num;
    }
    return out;
}
Blob operator-(const double num, Blob& A) {
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = num - A[i] ;
    }
    return out;
}
Blob operator-(Blob& A, Blob& B) {
    vector<int> sz_A = A.size();
    vector<int> sz_B = B.size();
    for (int i = 0; i < 4; ++i) {
        assert(sz_A[i] == sz_B[i]);
    }
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] - B[i];
    }
    return out; 
}
Blob operator*(Blob& A, const double num) {
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] * num;
    }
    return out;
}
Blob operator*(const double num, Blob& A) {
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] * num;
    }
    return out;
}
Blob operator*(Blob& A, Blob& B) {
    vector<int> sz_A = A.size();
    vector<int> sz_B = B.size();
    for (int i = 0; i < 4; ++i) {
        assert(sz_A[i] == sz_B[i]);
    }
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] % B[i];
    }
    return out;
}
Blob operator/(Blob& A, const double num) {
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] / num;
    }
    return out;
}
Blob operator/(const double num, Blob& A) {
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = num / A[i];
    }
    return out;
}
Blob operator/(Blob& A, Blob& B) {
    vector<int> sz_A = A.size();
    vector<int> sz_B = B.size();
    for (int i = 0; i < 4; ++i) {
        assert(sz_A[i] == sz_B[i]);
    }
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] / B[i];
    }
    return out;
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
        data_[i] = data_[i] + num;
    }
    return *this;
}
Blob& Blob::operator-=(const double num) {
    for (int i=0; i<N_; ++i) {
        data_[i] = data_[i] - num;
    }
    return *this;
}
Blob& Blob::operator*=(const double num) {
    for (int i=0; i<N_; ++i) {
        data_[i] = data_[i] * num;
    }
    return *this;
}
Blob& Blob::operator/=(const double num) {
    for (int i=0; i<N_; ++i) {
        data_[i] = data_[i] / num;
    }
    return *this;
}

//---Blob---
void Blob::setShape(vector<int>& shape) {
    N_ = shape[0];
    C_ = shape[1];
    H_ = shape[2];
    W_ = shape[3];
    data_ = vector<cube>(N_, cube(H_, W_, C_));
    return;
}
Blob::Blob(const int n, const int c, const int h, const int w, int type) :
        N_(n), C_(c), H_(h), W_(w) {
    if (type == TNONE)  data_ = vector<cube>(N_, cube(H_, W_, C_, fill::none));
    if (type == TONES)  data_ = vector<cube>(N_, cube(H_, W_, C_, fill::ones));
    if (type == TZEROS) data_ = vector<cube>(N_, cube(H_, W_, C_, fill::zeros));
    if (type == TRANDU) data_ = vector<cube>(N_, cube(H_, W_, C_, fill::randu));
    if (type == TRANDN) data_ = vector<cube>(N_, cube(H_, W_, C_, fill::randn));
    if (type == TDEFAULT) data_ = vector<cube>(N_, cube(H_, W_, C_));
    return;
}
Blob::Blob(const int n, const int c, const int h, const int w, const double eps):
        N_(n), C_(c), H_(h), W_(w) {
    data_ = vector<cube>(N_, cube(H_, W_, C_, fill::randn) * eps);
    return;
}
Blob::Blob(const vector<int>& shape) :
        N_(shape[0]), C_(shape[1]), H_(shape[2]), W_(shape[3]) {
    data_ = vector<cube>(N_, cube(H_, W_, C_));
    return;
}
Blob::Blob(const vector<int>& shape, const double eps) :
        N_(shape[0]), C_(shape[1]), H_(shape[2]), W_(shape[3]) {
    data_ = vector<cube>(N_, cube(H_, W_, C_, fill::randn) * eps);
    return;
}

cube& Blob::operator[] (int i) {
    return data_[i];
}

vector<int> Blob::size() {
    vector<int> shape{ N_, C_, H_, W_ };
    return shape;   
}

vector<cube>& Blob::get_data() {
    return data_;
}

mat Blob::reshape() {
    cube dst;
    for (int i = 0; i < N_; ++i) {
        dst = join_slices(dst, data_[i]);
    }
    return arma::reshape(vectorise(dst), N_, C_*H_*W_);
}

double Blob::sum() {
    assert(!data_.empty());
    double ans = 0;
    for (int i = 0; i < N_; ++i) {
        ans += accu(data_[i]);
    }
    return ans;
}

double Blob::sumElement() {
    return N_ * C_ * H_ * W_;
}

Blob Blob::max(double val) {
    assert(!data_.empty());
    Blob out(*this);
    for (int i = 0; i < N_; ++i) {
        out[i].transform([](double ev) {return ev;});
    }
}

} // namespace mini_net
