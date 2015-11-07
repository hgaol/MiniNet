/*!
*  Copyright (c) 2015 by hgaolbb
* \file layer.hpp
* \brief all layers implement
*/

#ifndef MINI_NET_LAYER_HPP_
#define MINI_NET_LAYER_HPP_

#include "blob.hpp"
using std::vector;

namespace mini_net {

/*! layer parameters */
struct Param {
    Param() : conv_stride(0), conv_pad(0) {}
    /*! conv param */
    int conv_stride;
    int conv_pad;
    inline void setConvParam(int s, int p) {
        conv_stride = s;
        conv_pad = p;
    }
};

/*!
 * \brief Affine Layer
 */
class AffineLayer {
public:
    AffineLayer() {}
    ~AffineLayer() {}
    /*!
    * \brief forward
    *             X:        [N, C, Hx, Wx]
    *             weight:   [F, C, Hw, Ww]
    *             bias:     [F, 1, 1, 1]
    *             out:      [N, F, 1, 1]
    * \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:weights, in[2]:bias
    * \param[out] Blob& out                     Y
    */
    static void forward(const vector<Blob*>& in, Blob** out);

    /*!
    * \brief backward
    *             in:       [N, C, Hx, Wx]
    *             weight:   [F, C, Hw, Ww]
    *             bias:     [F, 1, 1, 1]
    *             dout:     [N, F, 1, 1]
    * \param[in]  const Blob* dout              dout
    * \param[in]  const vector<Blob*>& cache    cache[0]:X, cache[1]:weights, cache[2]:bias
    * \param[out] vector<Blob*>& grads          grads[0]:dX, grads[1]:dW, grads[2]:db
    */
    static void backward(Blob* dout, const vector<Blob*>& cache, vector<Blob*>& grads);
};

/*!
* \brief Convolutional Layer
*/
class ConvLayer {
public:
    ConvLayer() {}
    ~ConvLayer() {}

    /*!
    * \brief forward
    *             X:        [N, C, Hx, Wx]
    *             weight:   [F, C, Hw, Ww]
    *             bias:     [F, 1, 1, 1]
    *             out:      [N, F, 1, 1]
    * \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:weights, in[2]:bias
    * \param[in]  const ConvParam* param        conv params
    * \param[out] Blob& out                     Y
    */
    static void forward(const vector<Blob*>& in, Blob** out, Param& param);

    /*!
    * \brief backward
    *             in:       [N, C, Hx, Wx]
    *             weight:   [F, C, Hw, Ww]
    *             bias:     [F, 1, 1, 1]
    *             dout:     [N, F, 1, 1]
    * \param[in]  const Blob* dout              dout
    * \param[in]  const vector<Blob*>& cache    cache[0]:X, cache[1]:weights, cache[2]:bias
    * \param[out] vector<Blob*>& grads          grads[0]:dX, grads[1]:dW, grads[2]:db
    */
    static void backward(Blob* dout, const vector<Blob*>& cache, vector<Blob*>& grads, Param& param);
};

} // namespace mini_net

#endif // MINI_NET_LAYER_
