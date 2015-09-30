// #include "layer.hpp"

// namespace mini_net {

// /**
//  in:		[N, C, Hx, Wx]
//  weight:	[F, C, Hw, Ww]
//  bias:		[F, 1, 1, 1]
//  out:		[N, F, 1, 1]
//  */
// void AffineLayer::forward(const vector<Blob*>& in, Blob* out, const Param* param = NULL) {
// 	if (out) {
// 		delete out;
// 	}
// 	int N = in[0]->get_N();
// 	int F = in[1]->get_N();
	
// 	out = new Blob(N, F, 1, 1);
// 	for (int i = 0; i < N; ++i) {
// 		double tmp = 0;
// 		for (int f = 0; f < F; ++f) {
// 			tmp += sum_blob_ith(in[0], i, in[1], f);
// 		}
		
// 	}
// }
// void AffineLayer::backward(const Blob* dout, const vector<Blob*>& cache, vector<Blob*> grads, const Param* param = NULL) {

// }

// //TODO

// } //namespace mini_net