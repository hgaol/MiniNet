#include <iostream>
#include <armadillo>
#include "../include/blob.hpp"
#include "../include/layer.hpp"

using namespace arma;
using namespace mini_net;

void testArmo() {
    cube a(2,2,2);
    a.print();
    cout << a(0,0,0) << endl;
    a(0,0,1) = 1;
    cout << a(0,0,1);
}
void testBlob() {
    // ptr
    Blob *pa = new Blob(5,2,2,2,TONES);
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
    AffineLayer::forward(vblob, out);
}

int main()
{
    //testArmo();
    //testBlob();
    testAffineLayer();

	return 0;
}
