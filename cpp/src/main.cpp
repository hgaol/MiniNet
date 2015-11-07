#include <iostream>
#include <armadillo>
#include "../include/blob.hpp"
#include "../include/layer.hpp"
#include "../include/test.hpp"

using namespace arma;
using namespace mini_net;

void testArma() {
    /*! test conv */
    /*! test cude/vec/mat to blob */
    vec va = linspace(1, 12, 12);
    va.print();
    vectorise(va).print();
    mat mma = reshape(va, 2,6);
    mma.print();
    // A * B, run!
    mat A = randu<mat>(5, 10);
    mat B = randu<mat>(10, 5);
    mat C = A * B;
    C.print();
    // mat2blob
    mat mc(2,8,fill::randu);
    mc.print();
    //mc = mc.t();
    //mc.print();
    //cube cb = cube(mc.colptr(0), 2, 2, 3);
    //cb.print();
    Blob *bc = NULL;
    mat2Blob(mc, &bc, 2,2,2);
    // vectorisze
    cube ca(2,2,2,fill::ones);
    ca *= 2;
    ca.print();
    mat mb = reshape(vectorise(ca), 4, 2);
    mb.print();

    mat ma(4,2,fill::ones);
    ma *= 0.5;
    //cout << vectorise(ma) << endl;
    cout << accu(vectorise(ma) % vectorise(ca)) << endl;
   
    //
    cube a(2,2,2);
    a.print();
    cout << a(0,0,0) << endl;
    a(0,0,1) = 1;
    cout << a(0,0,1);
}

void testBlob() {
    /*! test pad() */
    Blob oa(2,2,2,2,TONES);
    oa.pad(1, 0).print();

    /*! test sum() abs()*/
    oa.print();
    oa.max(0.5).print();
    Blob oc(2,2,2,2,TRANDN);
    oc.print();
    oc.abs().print();
    cout << oc.maxVal() << endl;

    /*! test operation */
    Blob ob(2,2,2,2,TONES);
    (oa - ob)[0].print();
    (oa + ob)[0].print();
    (oa * ob)[0].print();
    (oa / ob)[0].print();

    // check sum()
    Blob aa(2,2,2,2,TONES);
    cout << aa.sum();

    // reshape
    Blob pc(5,2,2,2,TONES);
    mat mc = pc.reshape();
    cout << "row: " << mc.n_rows << "\t" << "col: " << mc.n_cols << endl;
    mc.print();

    // ptr
    Blob *pa = new Blob(5,2,2,1,TONES);
    (*pa)[0].print();
    Blob *pb = new Blob(5,2,2,2,TDEFAULT);
    (*pb)[0].print();

    // += -= *= /+
    Blob a(2,3,2,2,TONES);
    a[0].print("a[0] = \n");
    a += 0.5;
    a[0].print("a[0] = \n");
    a -= 0.5;
    a[0].print("a[0] = \n");
    a *= 2;
    a[0].print("a[0] = \n");
    a /= 2;
    a[0].print("a[0] = \n");

    return;
}

void testAffineLayer() {
    Blob a(5,3,2,2,TONES);
    Blob b(10,3,2,2,TONES);
    Blob c(10,1,1,1,TONES);
    b *= 2;
    a *= 3;
    vector<Blob*> vblob{&a, &b, &c};
    Blob *out = NULL;
    // test Affine Layer forward
    AffineLayer::forward(vblob, &out);
    vector<Blob*> grads;
    AffineLayer::backward(out, vblob, grads);
}

void testConvLayer() {
    Blob x(4,3,5,5,TRANDN);
    Blob w(2,3,3,3,TRANDN);
    Blob b(2,1,1,1,TRANDN);
    Blob dout(4,2,5,5,TRANDN);
    Param param;
    param.setConvParam(1,1);
    vector<Blob*> in{&x, &w, &b};
    //Blob *out = NULL;
    //ConvLayer::forward(in, &out, param);
    //(*out).print();
    vector<Blob*> grads;
    ConvLayer::backward(&dout, in, grads, param);

    /*! test num_grads */
    Blob num_dx = Test::calcNumGradientBlobParam(in, &dout, param, ConvLayer::forward, TDX);
    Blob num_dw = Test::calcNumGradientBlobParam(in, &dout, param, ConvLayer::forward, TDW);
    Blob num_db = Test::calcNumGradientBlobParam(in, &dout, param, ConvLayer::forward, TDB);
    //vector<Blob*> grads;
    //ConvLayer::backward(&dout, in, grads, param);
    num_db.print("num_dw:\n");
    (*grads[2]).print("dw:\n");

    cout << Test::relError(num_dx, *grads[0]) << endl;
    cout << Test::relError(num_dw, *grads[1]) << endl;
    cout << Test::relError(num_db, *grads[2]) << endl;

    return;
}

void testPoolLayer() {
    Blob x(1,1,8,8,TRANDN);
    Blob dout(1,1,4,4,TRANDN);
    vector<Blob*> in{&x};
    Blob *out = NULL;
    Param param;
    param.setPoolParam(2,2,2);
    PoolLayer::forward(in, &out, param);
    x.print("x:\n");
    (*out).print("out:\n");
    vector<Blob*> grads;
    PoolLayer::backward(&dout, in, grads, param);
    Blob num_dx = Test::calcNumGradientBlobParam(in, &dout, param, PoolLayer::forward, TDX);
    num_dx.print("num_dx:\n");
    (*grads[0]).print("dx:\n");
    cout << Test::relError(num_dx, *grads[0]) << endl;
}

void testRelu() {
    Blob x(1,1,5,5,TRANDN);
    Blob dout(1,1,5,5,TRANDN);
    vector<Blob*> in{&x};
    Blob *out = NULL;
    //ReluLayer::forward(in, &out);
    //x.print();
    //(*out).print();
    vector<Blob*> grads;
    ReluLayer::backward(&dout, in, grads);
    Blob num_dx = Test::calcNumGradientBlob(in, &dout, ReluLayer::forward, TDX);
    num_dx.print();
    (*grads[0]).print();
    cout << Test::relError(num_dx, *grads[0]) << endl;
}

void testTest() {
    /*
    // check gradient
    mat x = linspace<mat>(0, 9, 10);
    mat a(1, 10, fill::ones);
    mat num_dx = Test::calcNumGradientX(x, Test::test_fcalar, a);
    x.print();
    num_dx.print();
    mat num_da = Test::calcNumGradientA(a, Test::test_fcalar, x);
    a.print();
    num_da.print();
    */

    /*! check gradient mat */
    /*
    // affine layer
    Blob x(1,2,2,2,TRANDN);
    Blob w(2,2,2,2,TRANDN);
    Blob b(2,1,1,1,TRANDN);
    Blob dout(1,2,1,1,TRANDN);
    vector<Blob*> in{&x, &w, &b};
    //Blob *oo = NULL;
    //w.print();
    //AffineLayer::forward(in, &oo);
    Blob num_dx = Test::calcNumGradientBlob(in, &dout, AffineLayer::forward, TDX);
    Blob num_dw = Test::calcNumGradientBlob(in, &dout, AffineLayer::forward, TDW);
    Blob num_db = Test::calcNumGradientBlob(in, &dout, AffineLayer::forward, TDB);
    vector<Blob*> grads;
    AffineLayer::backward(&dout, in, grads);
    num_db.print("num_dw:\n");
    (*grads[2]).print("dw:\n");

    cout << Test::relError(num_dx, *grads[0]) << endl;
    cout << Test::relError(num_dw, *grads[1]) << endl;
    cout << Test::relError(num_db, *grads[2]) << endl;
    */

    return;
}

int main()
{
    //testArma();
    //testBlob();
    //testAffineLayer();
    //testConvLayer();
    //testPoolLayer();
    testRelu();
    //testTest();

    return 0;
}
