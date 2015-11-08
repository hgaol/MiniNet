/*!
*  Copyright (c) 2015 by hgaolbb
* \file layer.cpp
* \brief all layers implement
*/

#include "../include/layer.hpp"

namespace mini_net {

/*!
* \brief forward
*             X:        [N, C, Hx, Wx]
*             weight:   [F, C, Hw, Ww]
*             bias:     [F, 1, 1, 1]
*             out:      [N, F, 1, 1]
* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:weights, in[2]:bias
* \param[in]  const Param* param            params
* \param[out] Blob& out                     Y
*/
void AffineLayer::forward(const vector<Blob*>& in, Blob** out) {
    if (*out) {
        delete *out;
        *out = NULL;
    }
    int N = in[0]->get_N();
    int F = in[1]->get_N();
    
    //*out = new Blob(N, F, 1, 1);

    mat x = (*in[0]).reshape();
    mat w = (*in[1]).reshape();
    mat b = (*in[2]).reshape();
    mat ans = x * w.t() + b.t();
    mat2Blob(ans, out, F, 1, 1);
    //(**out).print();
    //(*out)[0].print();
    /*
    for (int i = 0; i < N; ++i) {
        for (int f = 0; f < F; ++f) {
            //(*in[0])[0].print();
            //(*in[1])[0].print();
            //cout << accu((*in[0])[i] % (*in[1])[f]);// + (*in[2])[f];
            (**out)[i](0,0,f) = accu((*in[0])[i] % (*in[1])[f]) + (*in[2])[f](0,0,0);
            //cout << (**out)[i](0, 0, f) << endl;
            //(*out)[i].print("out[i] = \n");
        }
        //(*out)[i].print("###");
    }
    (**out).print();
    */

    return;
}

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
void AffineLayer::backward(Blob* dout, const vector<Blob*>& cache, vector<Blob*>& grads) {
    Blob *dX = NULL;
    Blob *dW = NULL;
    Blob *db = NULL;
    
    int n = dout->get_N();

    Blob *pX = cache[0];
    //(*pX).reshape().print("x\n");
    Blob *pW = cache[1];
    //(*pW).reshape().print("w\n");
    Blob *pb = cache[2];
    //(*pb).reshape().print("b\n");
    //(*dout).reshape().print("dy\n");

    // calc grads
    // dX
    mat mat_dx = (*dout).reshape() * (*pW).reshape();
    //mat_dx.print("dx\n");
    mat2Blob(mat_dx, &dX, (*pX).size());
    grads.push_back(dX);
    // dW
    mat mat_dw = (*dout).reshape().t() * (*pX).reshape();
    //mat_dw.print("dw\n");
    mat2Blob(mat_dw, &dW, (*pW).size());
    grads.push_back(dW);
    // db
    mat mat_db = (*dout).reshape().t() * mat(n, 1, fill::ones);
    //mat_db.print("db\n");
    mat2Blob(mat_db, &db, (*pb).size());
    grads.push_back(db);

    return;
}

/*!
* \brief convolutional layer forward
*             X:        [N, C, Hx, Wx]
*             weight:   [F, C, Hw, Ww]
*             bias:     [F, 1, 1, 1]
*             out:      [N, F, (Hx+pad*2-Hw)/stride+1, (Wx+pad*2-Ww)/stride+1]
* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:weights, in[2]:bias
* \param[in]  const ConvParam* param        conv params: stride, pad
* \param[out] Blob** out                    Y
*/
void ConvLayer::forward(const vector<Blob*>& in, Blob** out, Param& param) {
    if (*out) {
        delete *out;
        *out = NULL;
    }
    assert((*in[0]).get_C() == (*in[1]).get_C());
    int N = (*in[0]).get_N();
    int F = (*in[1]).get_N();
    int C = (*in[0]).get_C();
    int Hx = (*in[0]).get_H();
    int Wx = (*in[0]).get_W();
    int Hw = (*in[1]).get_H();
    int Ww = (*in[1]).get_W();

    // calc Hy, Wy
    int Hy = (Hx + param.conv_pad*2 -Hw) / param.conv_stride + 1;
    int Wy = (Wx + param.conv_pad*2 -Ww) / param.conv_stride + 1;
    
    *out = new Blob(N, F, Hy, Wy);
    Blob padX = (*in[0]).pad(param.conv_pad);

    for (int n = 0; n < N; ++n) {
        for (int f = 0; f < F; ++f) {
            for (int hh = 0; hh < Hy; ++hh) {
                for (int ww = 0; ww < Wy; ++ww) {
                    cube window = padX[n](span(hh * param.conv_stride, hh * param.conv_stride + Hw - 1),
                                            span(ww * param.conv_stride, ww * param.conv_stride + Ww - 1),
                                            span::all);
                    (**out)[n](hh, ww, f) = accu(window % (*in[1])[f]) + as_scalar((*in[2])[f]);
                    //cout << accu(window % (*in[1])[f]) << endl;
                }
            }
        }
    }
    return;
}

/*!
* \brief backward
*             in:       [N, C, Hx, Wx]
*             weight:   [F, C, Hw, Ww]
*             bias:     [F, 1, 1, 1]
*             out:      [N, F, (Hx+pad*2-Hw)/stride+1, (Wx+pad*2-Ww)/stride+1]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X, cache[1]:weights, cache[2]:bias
* \param[out] vector<Blob*>& grads          grads[0]:dX, grads[1]:dW, grads[2]:db
*/
void ConvLayer::backward(Blob* dout, const vector<Blob*>& cache, vector<Blob*>& grads, Param& param) {
    int N = (*cache[0]).get_N();
    int F = (*cache[1]).get_N();
    int C = (*cache[0]).get_C();
    int Hx = (*cache[0]).get_H();
    int Wx = (*cache[0]).get_W();
    int Hw = (*cache[1]).get_H();
    int Ww = (*cache[1]).get_W();
    int Hy = (*dout).get_H();
    int Wy = (*dout).get_W();
    assert(C == (*cache[1]).get_C());
    assert(F == (*cache[2]).get_N());
    if (!grads.empty()) {
        grads.clear();
    }
    Blob *dX = new Blob(cache[0]->size(), TZEROS);
    Blob *dW = new Blob(cache[1]->size(), TZEROS);
    Blob *db = new Blob(cache[2]->size(), TZEROS);

    Blob pad_dX(N, C, Hx + param.conv_pad*2, Wx + param.conv_pad*2, TZEROS);
    Blob pad_X = (*cache[0]).pad(1);

    for (int n = 0; n < N; ++n) {
        for (int f = 0; f < F; ++f) {
            for (int hh = 0; hh < Hy; ++hh) {
                for (int ww = 0; ww < Wy; ++ww) {
                    cube window = pad_X[n](span(hh * param.conv_stride,  hh * param.conv_stride + Hw - 1),
                                            span(ww * param.conv_stride, ww * param.conv_stride + Ww - 1),
                                            span::all);
                    //window.print("window:\n");
                    //cout << (*db)[f](0, 0, 0) << endl;
                    (*db)[f](0, 0, 0) += (*dout)[n](hh, ww, f);
                    //cout << (*db)[f](0, 0, 0) << endl;
                    //(*dW)[f].print("W before:\n");
                    (*dW)[f] += window * (*dout)[n](hh, ww, f);
                    //(*dW)[f].print("W after:\n");
                    //pad_dX[n](span(hh * param.stride, hh * param.stride + Hw - 1),
                    //    span(ww * param.stride, ww * param.stride + Ww - 1),
                    //    span::all).print("X before:\n");
                    pad_dX[n](span(hh * param.conv_stride, hh * param.conv_stride + Hw - 1),
                        span(ww * param.conv_stride, ww * param.conv_stride + Ww - 1),
                        span::all) += (*cache[1])[f] * (*dout)[n](hh, ww, f);
                    //pad_dX[n](span(hh * param.stride, hh * param.stride + Hw - 1),
                    //    span(ww * param.stride, ww * param.stride + Ww - 1),
                    //    span::all).print("X after:\n");
                }
            }
        }
    }
    *dX = pad_dX.dePad(param.conv_pad);
    grads.push_back(dX);
    grads.push_back(dW);
    grads.push_back(db);

    return;
}

/*!
* \brief forward
*             X:        [N, C, Hx, Wx]
*             out:      [N, C, Hx/2, Wx/2]
* \param[in]  const vector<Blob*>& in       in[0]:X
* \param[in]  const Param* param        conv params
* \param[out] Blob& out                     Y
*/
void PoolLayer::forward(const vector<Blob*>& in, Blob** out, Param& param) {
    if (*out) {
        delete *out;
        *out = NULL;
    }
    int N = (*in[0]).get_N();
    int C = (*in[0]).get_C();
    int Hx = (*in[0]).get_H();
    int Wx = (*in[0]).get_W();
    int height = param.pool_height;
    int width = param.pool_width;
    int stride = param.pool_stride;

    int Hy = (Hx - height) / stride + 1;
    int Wy = (Wx - width) / stride + 1;
    
    *out = new Blob(N, C, Hy, Wy);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int hh = 0; hh < Hy; ++hh) {
                for (int ww = 0; ww < Wy; ++ww) {
                    /*(*in[0])[n](span(hh * stride, hh * stride + height - 1),
                        span(ww * stride, ww * stride + width - 1),
                        span(c, c)).print();
                    cout << (*in[0])[n](span(hh * stride, hh * stride + height - 1),
                        span(ww * stride, ww * stride + width - 1),
                        span(c, c)).max() << endl;*/
                    (**out)[n](hh, ww, c) = (*in[0])[n](span(hh * stride, hh * stride + height - 1),
                                                        span(ww * stride, ww * stride + width - 1),
                                                        span(c, c)).max();
                }
            }
        }
    }
    return;
}

/*!
* \brief backward
*             cache:    [N, C, Hx, Wx]
*             dout:     [N, F, Hx/2, Wx/2]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X
* \param[out] vector<Blob*>& grads          grads[0]:dX
*/
void PoolLayer::backward(Blob* dout, const vector<Blob*>& cache, vector<Blob*>& grads, Param& param) {
    int N = (*cache[0]).get_N();
    int C = (*cache[0]).get_C();
    int Hx = (*cache[0]).get_H();
    int Wx = (*cache[0]).get_W();
    int Hy = (*dout).get_H();
    int Wy = (*dout).get_W();
    int height = param.pool_height;
    int width = param.pool_width;
    int stride = param.pool_stride;

    Blob *dX = new Blob((*cache[0]).size(), TZEROS);
    //(*dX).print("dX:\n");

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int hh = 0; hh < Hy; ++hh) {
                for (int ww = 0; ww < Wy; ++ww) {
                    mat window = (*cache[0])[n](span(hh * stride, hh * stride + height - 1),
                                    span(ww * stride, ww * stride + width - 1),
                                    span(c, c));
                    double maxv = window.max();
                    mat mask = conv_to<mat>::from(maxv == window);
                    //mask.print("mask\n");
                    //(mask%window).print("multi:\n");
                    //(*dX)[n](span(hh * stride, hh * stride + height - 1),
                    //    span(ww * stride, ww * stride + width - 1),
                    //    span(c, c)).print();
                    (*dX)[n](span(hh * stride, hh * stride + height - 1),
                            span(ww * stride, ww * stride + width - 1),
                            span(c, c)) += mask * (*dout)[n](hh, ww, c);
                    //(*dX)[n](span(hh * stride, hh * stride + height - 1),
                    //    span(ww * stride, ww * stride + width - 1),
                    //    span(c, c)).print();
                }
            }
        }
    }
    grads.push_back(dX);
    return;
}

/*!
* \brief forward, out = max(0, X)
*             X:        [N, C, Hx, Wx]
*             out:      [N, C, Hx, Wx]
* \param[in]  const vector<Blob*>& in       in[0]:X
* \param[out] Blob& out                     Y
*/
void ReluLayer::forward(const vector<Blob*>& in, Blob** out) {
    if (*out) {
        delete *out;
        *out = NULL;
    }
    *out = new Blob(*in[0]);
    (**out).maxIn(0);
    return;
}

/*!
* \brief backward, dX = dout .* (X > 0)
*             in:       [N, C, Hx, Wx]
*             dout:     [N, F, Hx, Wx]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X
* \param[out] vector<Blob*>& grads          grads[0]:dX
*/
void ReluLayer::backward(Blob* dout, const vector<Blob*>& cache, vector<Blob*>& grads) {
    Blob *dX = new Blob(*cache[0]);
    int N = cache[0]->get_N();
    for (int i = 0; i < N; ++i) {
        (*dX)[i].transform([](double e) {return e > 0 ? 1 : 0;});
    }
    (*dX) = (*dout) * (*dX);
    grads.push_back(dX);
    return;
}

/*!
* \brief forward
*             X:        [N, C, Hx, Wx]
*             out:      [N, C, Hx, Wx]
* \param[in]  const vector<Blob*>& in       in[0]:X
* \param[in]  Param& param                  int mode, double p, int seed, Blob *mask
* \param[out] Blob& out                     Y
*/
void DropoutLayer::forward(const vector<Blob*>& in, Blob** out, Param& param) {
    if (*out) {
        delete *out;
        *out = NULL;
    }
    int mode = param.drop_mode;
    double p = param.drop_p;
    assert(0 <= p && p <= 1);
    assert(0 <= mode && mode <= 3);
    int seed;
    /*! train mode */
    if ((mode & 1) == 1) {
        if ((mode & 2) == 2) {
            seed = param.drop_seed;
            arma_rng::set_seed(seed);
        }
        Blob *mask = new Blob(in[0]->size(), TRANDU);
        //(*mask).print("maks\n");
        (*mask).smallerIn(p);
        //(*mask).print("maks\n");
        //(*in[0]).print("X:\n");
        *out = new Blob(*in[0] * (*mask) / p);
        //(**out).print("out\n");
        if (param.drop_mask) {
            delete param.drop_mask;
        }
        param.drop_mask = mask;
    }
    else {
        /*! test mode */
        *out = new Blob(*in[0]);
    }
    return;
}

/*!
* \brief backward
*             in:       [N, C, Hx, Wx]
*             dout:     [N, F, Hx, Wx]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X
* \param[in]  Param& param                  int mode, double p, int seed, Blob *mask
* \param[out] vector<Blob*>& grads          grads[0]:dX
*/
void DropoutLayer::backward(Blob* dout, const vector<Blob*>& cache, vector<Blob*>& grads, Param& param) {
    Blob *dX = new Blob((*dout));
    int mode = param.drop_mode;
    assert(0 <= mode && mode <= 3);
    if ((mode & 1) == 1) {
        *dX = (*dX) * (*param.drop_mask) / param.drop_p;
    }
    grads.push_back(dX);
    return;
}

/*!
* \brief forward
*             X:        [N, C, 1, 1], usually the output of affine(fc) layer
*             Y:        [N, C, 1, 1], ground truth, with 1(true) or 0(false)
* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:Y
* \param[out] double& loss                  loss
* \param[out] Blob** out                    out: dX
*/
void SoftmaxLossLayer::go(const vector<Blob*>& in, double& loss, Blob** out, int mode) {
    Blob X(*in[0]);
    Blob Y(*in[1]);
    if (*out) {
        delete *out;
        *out = NULL;
    }
    int N = (*in[0]).get_N();
    int C = (*in[0]).get_C();
    int H = (*in[0]).get_H();
    int W = (*in[0]).get_W();
    assert(H == 1 && W == 1);

    mat mat_x = X.reshape();
    //mat_x.print();
    mat mat_y = Y.reshape();
    //mat_y.print();

    /*! forward */
    mat row_max = repmat(arma::max(mat_x, 1), 1, C);
    //row_max.print("row_max\n");
    mat_x = arma::exp(mat_x - row_max);
    mat row_sum = repmat(arma::sum(mat_x, 1), 1, C);
    mat e = mat_x / row_sum;
    mat prob = -arma::log(e);
    //prob.print("prob:\n");
    //mat_y.print("y:\n");
    /*! loss should near 2.3026 */
    loss = accu(prob % mat_y) / N;
    /*! only forward */
    if (mode == 1)
        return;

    /*! backward */
    mat dx = e - mat_y;
    dx /= N;
    mat2Blob(dx, out, (*in[0]).size());
    return;
}

/*!
* \brief forward
*             X:        [N, C, 1, 1], usually the output of affine(fc) layer
*             Y:        [N, C, 1, 1], ground truth, with 1(true) or 0(false)
* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:Y
* \param[out] double& loss                  loss
* \param[out] Blob** out                    out: dX
* \param[in]  int mode                      1: only forward, 0:forward and backward
*/
void SVMLossLayer::go(const vector<Blob*>& in, double& loss, Blob** out, int mode) {
    if (*out) {
        delete *out;
        *out = NULL;
    }
    /*! let delta equals to 1 */
    double delta = 0.2;
    int N = in[0]->get_N();
    int C = in[0]->get_C();
    mat mat_x = (*in[0]).reshape();
    //mat_x.print("mat_x;\n");
    mat mat_y = (*in[1]).reshape();
    //mat_y.print("mat_y;\n");

    /*! forward */
    mat good_x = repmat(arma::sum(mat_x % mat_y, 1), 1, C);
    //good_x.print("good_x:\n");
    mat mat_loss = (mat_x - good_x + delta);
    //mat_loss.print("mat_loss\n");
    mat_loss.transform([](double e) {return e > 0 ? e : 0;});
    //mat_loss.print("mat_loss > 0:\n");
    mat_y.transform([](double e) {return e ? 0 : 1;});
    ////mat_y.print();
    mat_loss %= mat_y;
    loss = accu(mat_loss) / N;
    if (mode == 1)
        return;

    /*! backward */
    mat dx(mat_loss);
    dx.transform([](double e) {return e ? 1 : 0;});
    //dx.print("dx:\n");
    mat_y.transform([](double e) {return e ? 0 : 1;});
    mat sum_x = repmat(arma::sum(dx, 1), 1, C) % mat_y;
    //sum_x.print("sum_x:\n");
    dx = (dx - sum_x) / N;
    mat2Blob(dx, out, in[0]->size());
    return;
}

} //namespace mini_net
