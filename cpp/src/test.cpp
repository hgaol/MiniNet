/*!
*  Copyright (c) 2015 by hgaolbb
* \file test.cpp
* \brief all layers implement
*/

#include "../include/layer.hpp"
#include "../include/blob.hpp"
#include "../include/test.hpp"

namespace mini_net {

template<typename _Tp>
mat Test::calcNumGradient(mat& x, _Tp func, mat& a, double eps) {
    /*
    int m = x.n_rows;
    int n = x.n_cols;
    mat num_grad(size(x));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double old_val = x(i, j);
            x(i, j) = x(i, j) + eps;
            fp = func(a, x);
            x(i, j) = x(i, j) - eps;
            fm = func(a, x);
            x(i, j) = old_val;
            num_grad(i, j) = (fp - fm) / (2 * eps);
        }
    }
    return num_grad;
    */
    return x;
}

double Test::fcalar(mat& a, mat& x) {
    return as_scalar(a * x);
}

} //namespace mini_net