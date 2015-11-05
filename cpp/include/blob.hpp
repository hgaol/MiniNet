/*!
*  Copyright (c) 2015 by hgaolbb
* \file blob.hpp
* \brief all layers implement
*/

#ifndef MINI_NET_BLOB_HPP_
#define MINI_NET_BLOB_HPP_

#include <armadillo>
#include <vector>

using std::vector;
using namespace arma;

namespace mini_net {

enum FillType {
    TNONE = 0,
    TONES = 1,
    TZEROS = 2,
    TRANDU = 3,
    TRANDN = 4,
    TDEFAULT = 5
};
class Blob;
Blob* operator+(Blob& A, const double num);
Blob* operator+(const double num, Blob& A);
Blob* operator+(Blob& A, Blob& B);
Blob* operator-(Blob& A, const double num);
Blob* operator-(const double num, Blob& A);
Blob* operator-(Blob& A, Blob& B);
Blob* operator*(Blob& A, const double num);
Blob* operator*(const double num, Blob& A);
Blob* operator*(Blob& A, Blob& B);
Blob* operator/(Blob& A, const double num);
Blob* operator/(const double num, Blob& A);
Blob* operator/(Blob& A, Blob& B);

class Blob {

public:
    Blob() : N_(0), C_(0), H_(0), W_(0), data_(NULL) {}
    explicit Blob(const int n, const int c, const int h, const int w, int type = TDEFAULT);
    explicit Blob(const int n, const int c, const int h, const int w, const double eps);
    explicit Blob(const vector<int>& shape);
    explicit Blob(const vector<int>& shape, const double eps);
    ~Blob();

    cube& operator[] (int i);
    friend Blob* operator+(Blob& A, const double num);
    friend Blob* operator+(const double num, Blob& A);
    friend Blob* operator+(Blob& A, Blob& B);
    friend Blob* operator-(Blob& A, const double num);
    friend Blob* operator-(const double num, Blob& A);
    friend Blob* operator-(Blob& A, Blob& B);
    friend Blob* operator*(Blob& A, const double num);
    friend Blob* operator*(const double num, Blob& A);
    friend Blob* operator*(Blob& A, Blob& B);
    friend Blob* operator/(Blob& A, const double num);
    friend Blob* operator/(const double num, Blob& A);
    friend Blob* operator/(Blob& A, Blob& B);

    Blob& operator+=(const double num);
    Blob& operator-=(const double num);
    Blob& operator*=(const double num);
    Blob& operator/=(const double num);

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
