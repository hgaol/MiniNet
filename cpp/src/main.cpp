/*!
*  Copyright (c) 2015 by hgaolbb
* \file main.cpp
* \brief for test
*/
#include <iostream>
#include <armadillo>
#include "../include/blob.hpp"
#include "../include/layer.hpp"
#include "../include/test.hpp"

using namespace arma;
using namespace mini_net;
using std::shared_ptr;

//
//void testSharedptr() {
//    shared_ptr<int> pp(new int(1));
//    cout << *pp << endl;
//    //pp.reset();
//    //cout << *pp << endl;
//}
//
//void testArma() {
//    /*! test randn/randu seed */
//    //arma_rng::set_seed(123);
//    arma_rng::set_seed_random();
//    cube cca(5,5,1,fill::randn);
//    cca.print();
//    //arma_rng::set_seed(123);
//    arma_rng::set_seed_random();
//    cube ccb(5,5,1,fill::randn);
//    ccb.print();
//    /*! test cude/vec/mat to blob */
//    vec va = linspace(1, 12, 12);
//    va.print();
//    vectorise(va).print();
//    mat mma = reshape(va, 2,6);
//    mma.print();
//    // A * B, run!
//    mat A = randu<mat>(5, 10);
//    mat B = randu<mat>(10, 5);
//    mat C = A * B;
//    C.print();
//    // mat2blob
//    mat mc(2,8,fill::randu);
//    mc.print();
//    //mc = mc.t();
//    //mc.print();
//    //cube cb = cube(mc.colptr(0), 2, 2, 3);
//    //cb.print();
//    Blob *bc = NULL;
//    //mat2Blob(mc, &bc, 2,2,2);
//    // vectorisze
//    cube ca(2,2,2,fill::ones);
//    ca *= 2;
//    ca.print();
//    mat mb = reshape(vectorise(ca), 4, 2);
//    mb.print();
//
//    mat ma(4,2,fill::ones);
//    ma *= 0.5;
//    //cout << vectorise(ma) << endl;
//    cout << accu(vectorise(ma) % vectorise(ca)) << endl;
//   
//    //
//    cube a(2,2,2);
//    a.print();
//    cout << a(0,0,0) << endl;
//    a(0,0,1) = 1;
//    cout << a(0,0,1);
//}
//
//void testBlob() {
//    /*! random problem */
//    Blob aaa(4,2,2,2,TRANDN);
//    aaa.reshape().print();
//    Blob aab(4,2,2,2,TRANDU);
//    aab.reshape().print();
//    /*! test pad() */
//    Blob oa(2,2,2,2,TONES);
//    oa.pad(1, 0).print();
//
//    /*! test sum() abs()*/
//    oa.print();
//    oa.max(0.5).print();
//    Blob oc(2,2,2,2,TRANDN);
//    oc.print();
//    oc.abs().print();
//    cout << oc.maxVal() << endl;
//
//    /*! test operation */
//    Blob ob(2,2,2,2,TONES);
//    (oa - ob)[0].print();
//    (oa + ob)[0].print();
//    (oa * ob)[0].print();
//    (oa / ob)[0].print();
//
//    // check sum()
//    Blob aa(2,2,2,2,TONES);
//    cout << aa.sum();
//
//    // reshape
//    Blob pc(5,2,2,2,TONES);
//    mat mc = pc.reshape();
//    cout << "row: " << mc.n_rows << "\t" << "col: " << mc.n_cols << endl;
//    mc.print();
//
//    // ptr
//    Blob *pa = new Blob(5,2,2,1,TONES);
//    (*pa)[0].print();
//    Blob *pb = new Blob(5,2,2,2,TDEFAULT);
//    (*pb)[0].print();
//
//    // += -= *= /+
//    Blob a(2,3,2,2,TONES);
//    a[0].print("a[0] = \n");
//    a += 0.5;
//    a[0].print("a[0] = \n");
//    a -= 0.5;
//    a[0].print("a[0] = \n");
//    a *= 2;
//    a[0].print("a[0] = \n");
//    a /= 2;
//    a[0].print("a[0] = \n");
//
//    return;
//}

void testAffineLayer() {
    shared_ptr<Blob> a(new Blob(5,3,2,2,TONES));
    shared_ptr<Blob> b(new Blob(10,3,2,2,TONES));
    shared_ptr<Blob> c(new Blob(10,1,1,1,TONES));
    shared_ptr<Blob> dout(new Blob(5,10,1,1,TRANDN));
    vector<shared_ptr<Blob>> in{a, b, c};
    shared_ptr<Blob> out;
    vector<shared_ptr<Blob>> grads;
    AffineLayer::backward(dout, in, grads);
    auto nfunc =[in](shared_ptr<Blob>& e) {return AffineLayer::forward(in, e);};
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
    Blob num_dw = Test::calcNumGradientBlob(in[1], dout, nfunc);
    Blob num_db = Test::calcNumGradientBlob(in[2], dout, nfunc);

    cout << Test::relError(num_dx, *grads[0]) << endl;
    cout << Test::relError(num_dw, *grads[1]) << endl;
    cout << Test::relError(num_db, *grads[2]) << endl;

    return;
}

void testConvLayer() {
    shared_ptr<Blob> x(new Blob(4,3,5,5,TRANDN));
    shared_ptr<Blob> w(new Blob(2,3,3,3,TRANDN));
    shared_ptr<Blob> b(new Blob(2,1,1,1,TRANDN));
    shared_ptr<Blob> dout(new Blob(4,2,5,5,TRANDN));
    Param param;
    param.setConvParam(1,1);
    vector<shared_ptr<Blob>> in{x, w, b};
    vector<shared_ptr<Blob>> grads;
    ConvLayer::backward(dout, in, grads, param);

    /*! test num_grads */
    auto nfunc =[in, &param](shared_ptr<Blob>& e) {return ConvLayer::forward(in, e, param);};
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
    Blob num_dw = Test::calcNumGradientBlob(in[1], dout, nfunc);
    Blob num_db = Test::calcNumGradientBlob(in[2], dout, nfunc);

    cout << Test::relError(num_dx, *grads[0]) << endl;
    cout << Test::relError(num_dw, *grads[1]) << endl;
    cout << Test::relError(num_db, *grads[2]) << endl;

    return;
}

void testPoolLayer() {
    shared_ptr<Blob> x(new Blob(1,1,8,8,TRANDN));
    shared_ptr<Blob> dout(new Blob(1,1,4,4,TRANDN));
    vector<shared_ptr<Blob>> in{x};
    shared_ptr<Blob> out;
    Param param;
    param.setPoolParam(2,2,2);
    PoolLayer::forward(in, out, param);
    vector<shared_ptr<Blob>> grads;
    PoolLayer::backward(dout, in, grads, param);
    auto nfunc =[in, &param](shared_ptr<Blob>& e) {return PoolLayer::forward(in, e, param);};
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
    cout << Test::relError(num_dx, *grads[0]) << endl;
}

void testRelu() {
    shared_ptr<Blob> x(new Blob(1,1,5,5,TRANDN));
    shared_ptr<Blob> dout(new Blob(1,1,5,5,TRANDN));
    vector<shared_ptr<Blob>> in{x};
    shared_ptr<Blob> out;
    vector<shared_ptr<Blob>> grads;
    ReluLayer::backward(dout, in, grads);
    auto nfunc =[in](shared_ptr<Blob>& e) {return ReluLayer::forward(in, e);};
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
    cout << Test::relError(num_dx, *grads[0]) << endl;
}

void testDropout() {
    shared_ptr<Blob> x(new Blob(1,1,5,5,TRANDN));
    shared_ptr<Blob> dout(new Blob(1,1,5,5,TRANDN));
    vector<shared_ptr<Blob>> in{x};
    shared_ptr<Blob> out;
    Param param;
    param.setDropoutpParam(3, 0.5, 123);
    vector<shared_ptr<Blob>> grads;
    shared_ptr<Blob> dummy_out;
    auto nfunc =[in, &param](shared_ptr<Blob>& e) {return DropoutLayer::forward(in, e, param);};
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
    DropoutLayer::backward(dout, in, grads, param);
    cout << Test::relError(num_dx, *grads[0]) << endl;
}

void testSoftmax() {
    shared_ptr<Blob> x(new Blob(10,8,1,1,TRANDU));
    mat aa = randi<mat>(10, 1, distr_param(0,7));
    mat bb(10,8,fill::zeros);
    for (int i = 0; i < 10; ++i) {
        bb(i, (uword)aa(i,0)) = 1;
    }
    shared_ptr<Blob> y;
    mat2Blob(bb, y, x->size());
    vector<shared_ptr<Blob>> in{x, y};
    double loss;
    shared_ptr<Blob> out;
    shared_ptr<Blob> dummy_out;
    auto nfunc =[in, &dummy_out](double& e) {return SoftmaxLossLayer::go(in, e, dummy_out, 1);};
    SoftmaxLossLayer::go(in, loss, out);
    Blob num_dx = Test::calcNumGradientBlobLoss(in[0], nfunc);
    cout << Test::relError(num_dx, *out) << endl;
}

void testSVM() {
    shared_ptr<Blob> x(new Blob(10, 8, 1, 1, TRANDU));
    mat aa = randi<mat>(10, 1, distr_param(0, 7));
    mat bb(10, 8, fill::zeros);
    for (int i = 0; i < 10; ++i) {
        bb(i, (uword)aa(i, 0)) = 1;
    }
    shared_ptr<Blob> y;
    mat2Blob(bb, y, x->size());
    vector<shared_ptr<Blob>> in{x, y};
    double loss;
    shared_ptr<Blob> out;
    SVMLossLayer::go(in, loss, out);
    shared_ptr<Blob> dummy_out;
    auto nfunc =[in, &dummy_out](double& e) {return SVMLossLayer::go(in, e, dummy_out, 1);};
    Blob num_dx = Test::calcNumGradientBlobLoss(in[0], nfunc);
    cout << Test::relError(num_dx, *out) << endl;
}

///*
//void testTest() {
//    /*
//    // check gradient
//    mat x = linspace<mat>(0, 9, 10);
//    mat a(1, 10, fill::ones);
//    mat num_dx = Test::calcNumGradientX(x, Test::test_fcalar, a);
//    x.print();
//    num_dx.print();
//    mat num_da = Test::calcNumGradientA(a, Test::test_fcalar, x);
//    a.print();
//    num_da.print();
//    */
//
//    /*! check gradient mat */
//    // affine layer
//    Blob x(4,2,2,2,TRANDN);
//    Blob w(2,2,2,2,TRANDN);
//    Blob b(2,1,1,1,TRANDN);
//    Blob dout(4,2,1,1,TRANDN);
//    vector<Blob*> in{&x, &w, &b};
//    //Blob *oo = NULL;
//    //w.print();
//    //AffineLayer::forward(in, &oo);
//    Blob num_dx = Test::calcNumGradientBlob(in, &dout, AffineLayer::forward, TDX);
//    Blob num_dw = Test::calcNumGradientBlob(in, &dout, AffineLayer::forward, TDW);
//    Blob num_db = Test::calcNumGradientBlob(in, &dout, AffineLayer::forward, TDB);
//    vector<Blob*> grads;
//    AffineLayer::backward(&dout, in, grads);
//    num_db.print("num_dw:\n");
//    (*grads[2]).print("dw:\n");
//
//    cout << Test::relError(num_dx, *grads[0]) << endl;
//    cout << Test::relError(num_dw, *grads[1]) << endl;
//    cout << Test::relError(num_db, *grads[2]) << endl;
//
//    return;
//}

int main()
{
    //testSharedptr();
    //testArma();
    //testBlob();
    testAffineLayer();
    //testConvLayer();
    //testPoolLayer();
    //testRelu();
    //testDropout();
    //testSoftmax();
    //testSVM();
    //testTest();
    //int n = 3;
    //while (n--) {
    //    mat a(5,5,fill::randn);
    //    a.print("a\n");
    //}
    //Blob b(4,2,2,2,TRANDN);
    //b.reshape().print();

    return 0;
}
