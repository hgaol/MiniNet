/*!
*  Copyright (c) 2015 by hgaolbb
* \file blob.hpp
* \brief all layers implement
*/

#ifndef MINI_NET_BLOB_HPP_
#define MINI_NET_BLOB_HPP_

#include <armadillo>
#include <vector>
#include <assert.h>
#include <iostream>
#include <cstdio>

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
// operation
Blob operator+(Blob& A, const double num);
Blob operator+(const double num, Blob& A);
Blob operator+(Blob& A, Blob& B);
Blob operator-(Blob& A, const double num);
Blob operator-(const double num, Blob& A);
Blob operator-(Blob& A, Blob& B);
Blob operator*(Blob& A, const double num);
Blob operator*(const double num, Blob& A);
Blob operator*(Blob& A, Blob& B);
Blob operator/(Blob& A, const double num);
Blob operator/(const double num, Blob& A);
Blob operator/(Blob& A, Blob& B);
// convertion
void mat2Blob(mat& mA, Blob** out, int c, int h, int w);
void mat2Blob(mat& mA, Blob** out, const vector<int>& sz);

class Blob {

public:
    Blob() : N_(0), C_(0), H_(0), W_(0), data_(NULL) {}
    explicit Blob(const int n, const int c, const int h, const int w, int type = TDEFAULT);
    explicit Blob(const int n, const int c, const int h, const int w, const double eps);
    explicit Blob(const vector<int>& shape);
    explicit Blob(const vector<int>& shape, const double eps);
    ~Blob() {}

    // need set shape later sometimes, like in test.hpp func[calcNumGradientBlob]
    void setShape(vector<int>& shape);
    cube& operator[] (int i);
    friend Blob operator+(Blob& A, const double num);
    friend Blob operator+(const double num, Blob& A);
    friend Blob operator+(Blob& A, Blob& B);
    friend Blob operator-(Blob& A, const double num);
    friend Blob operator-(const double num, Blob& A);
    friend Blob operator-(Blob& A, Blob& B);
    friend Blob operator*(Blob& A, const double num);
    friend Blob operator*(const double num, Blob& A);
    friend Blob operator*(Blob& A, Blob& B);
    friend Blob operator/(Blob& A, const double num);
    friend Blob operator/(const double num, Blob& A);
    friend Blob operator/(Blob& A, Blob& B);

    Blob& operator+=(const double num);
    Blob& operator-=(const double num);
    Blob& operator*=(const double num);
    Blob& operator/=(const double num);

    // return [N,C,H,W]
    vector<int> size();

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
    vector<cube>& get_data();

    //@brief: reshape [N,C,H,W] to [N,C*H*W]
    mat reshape();

    /*! Element wise operation */
    // sum of all element in Blob
    double sum();
    /*! sum number of element*/
    double numElement();
    /*! element wise operation, if element is smaller than val, then set it equals to val*/
    Blob max(double val);
    /*! element wise operation, return absolute value*/
    Blob abs();

    /*! find the max value in the blob */
    double maxVal();
    /*! print Blob */
    void print(std::string s = "");

private:
    int N_;
    int C_;
    int H_;
    int W_;
    vector<cube> data_;
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
