#include <iostream>
#include <armadillo>
#include "../include/blob.hpp"
#include "../include/layer.hpp"
#include "../include/test.hpp"
//#include "test.cpp"

using namespace arma;
using namespace mini_net;

void testArma() {
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
    /*! test sum() abs()*/
    Blob oa(2,2,2,2,TRANDU);
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

    /*! chech gradient mat */
    // affine layer
    Blob x(3,2,2,2,TRANDN);
    Blob w(2,2,2,2,TRANDN);
    Blob b(2,1,1,1,TRANDN);
    Blob dout(3,2,1,1,TRANDN);
    vector<Blob*> in{&x, &w, &b};
    Blob num_dx = Test::calcNumGradientBlob(in, &dout, AffineLayer::forward, TDX);
    Blob num_dw = Test::calcNumGradientBlob(in, &dout, AffineLayer::forward, TDW);
    Blob num_db = Test::calcNumGradientBlob(in, &dout, AffineLayer::forward, TDB);
    vector<Blob*> grads;
    AffineLayer::backward(&dout, in, grads);
    cout << Test::relError(num_dx, *grads[0]) << endl;
    cout << Test::relError(num_dw, *grads[1]) << endl;
    cout << Test::relError(num_db, *grads[2]) << endl;

    return;
}

int main()
{
    //testArma();
    //testBlob();
    //testAffineLayer();
    testTest();

    return 0;
}
