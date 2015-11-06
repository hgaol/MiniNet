/*!
*  Copyright (c) 2015 by hgaolbb
* \file test.hpp
* \brief all layers implement
*/

#ifndef MINI_NET_TEST_HPP_
#define MINI_NET_TEST_HPP_

#include "blob.hpp"
#include "layer.hpp"

namespace mini_net {

enum TestType {
    TDX = 1,
    TDW = 2,
    TDB = 3
};

class Test {

public:
    Test() {}
    ~Test() {}

    template<typename _Tp>
    static mat calcNumGradientX(mat& x, _Tp func, mat& a, double eps = 1e-6) {
        int m = x.n_rows;
        int n = x.n_cols;
        mat num_grad(size(x));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                double old_val = x(i, j);
                x(i, j) = old_val + eps;
                double fp = func(a, x);
                x(i, j) = old_val - eps;
                double fm = func(a, x);
                x(i, j) = old_val;
                num_grad(i, j) = (fp - fm) / (2 * eps);
                int a=0;
                a == 0;
            }
        }
        return num_grad;
    }

    template<typename _Tp>
    static mat calcNumGradientA(mat& a, _Tp func, mat& x, double eps = 1e-6) {
        int m = a.n_rows;
        int n = a.n_cols;
        mat num_grad(size(a));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                double old_val = a(i, j);
                a(i, j) = old_val + eps;
                double fp = func(a, x);
                a(i, j) = old_val - eps;
                double fm = func(a, x);
                a(i, j) = old_val;
                num_grad(i, j) = (fp - fm) / (2 * eps);
                int a=0;
                a == 0;
            }
        }
        return num_grad;
    }

    template<typename _Tp>
    static Blob calcNumGradientBlob(vector<Blob*>& in, Blob* dout, _Tp func, mat& weight, TestType type, double eps = 1e-6) {
        
        if (type == TDX) Blob* num_grad(in[0]->size());
        if (type == TDW) Blob* num_grad(in[1]->size());
        if (type == TDB) Blob* num_grad(in[2]->size());

        int N = num_grad->get_N();
        int C = num_grad->get_C();
        int H = num_grad->get_H();
        int W = num_grad->get_W();

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        if (type == TDX) {
                            double old_val = (*in[0])[n](h, w, c);
                            (*in[0])[n](h, w, c) = old_val + eps;
                            Blob *fp
                            func(in, &fp);
                            (*in[0])[n](h, w, c) = old_val - eps;
                            Blob *fm
                            func(in, &fm);
                            (*in[0])[n](h, w, c) = old_val;
                            (*num_grad)[n](h, w, c) = 
                        }
                        if (type == TDW) {
                            double old_val = (*in[1])[n](h, w, c);
                        }
                        if (type == TDB) {
                            double old_val = (*in[2])[n](h, w, c);
                        }
                        x(i, j) = old_val + eps;
                        double fp = func(a, x);
                        x(i, j) = old_val - eps;
                        double fm = func(a, x);
                        x(i, j) = old_val;
                        num_grad(i, j) = (fp - fm) / (2 * eps);
                        int a=0;
                        a == 0;
                    }
                }
            }
        }
        return num_grad;
    }

    //template<typename _Tp> static void calcNumGradientMat(Blob *x, _Tp func);
    static double test_fcalar(mat& a, mat& x) {
        return as_scalar(a * x);
    }
};

} // namespace mini_net

#endif // MINI_NET_TEST_