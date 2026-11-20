#include "../core/tensor.h"
#include "../core/loss.h"
#include "../layers/conv2d.h"
#include "../layers/pool.h"
#include "../layers/dense.h"
#include "../layers/activations.h"
#include "../optimizer/adamw.h"
#ifndef SHADE_CNN_NO_MAIN
#include "../utils/mnist.h"
#endif
#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>

// CNN: Conv(1->8,3x3) + ReLU + Pool(2x2) -> Conv(8->16,3x3) + ReLU + Pool(2x2) -> Dense(784->128) -> Dense(128->10)
// Shapes:
//   28x28x1 -> Conv(3x3,p=1) -> 28x28x8 -> Pool(2x2) -> 14x14x8
//   -> Conv(3x3,p=1) -> 14x14x16 -> Pool(2x2) -> 7x7x16 = 784
//   -> Dense(784->128) -> Dense(128->10)

// Conv(3x3, p=1, s=1) preserves the spatial size; Pool(2x2, s=2) halves it.
static i32 cnn_pooled(i32 s) { return (s - 2) / 2 + 1; }
static i32 cnn_flat_dim(i32 h, i32 w, i32 c) {
    i32 ph = cnn_pooled(cnn_pooled(h));
    i32 pw = cnn_pooled(cnn_pooled(w));
    return ph * pw * c;
}

struct CNN {
    Conv2D c1, c2;
    Act a1, a2;
    MaxPool2D p1, p2;
    Dense f1, f2;
    i32 H, W, C1, C2, poolH, poolW, flat_dim;

    CNN() : CNN(28, 28, 1, 8, 16, 128, 10) {}

    // Dimension-configurable for gradient checking with a tiny model; the
    // default MNIST shape above is unchanged.
    CNN(i32 h, i32 w, i32 cin, i32 c1o, i32 c2o, i32 hid, i32 out)
        : c1(cin, c1o, 3, 3, 1, 1), a1(Activation::RELU), p1(2, 2, 2),
          c2(c1o, c2o, 3, 3, 1, 1), a2(Activation::RELU), p2(2, 2, 2),
          f1(cnn_flat_dim(h, w, c2o), hid, true), f2(hid, out, false),
          H(h), W(w), C1(c1o), C2(c2o),
          poolH(cnn_pooled(h)), poolW(cnn_pooled(w)),
          flat_dim(cnn_flat_dim(h, w, c2o)) {}

    Tensor forward(const Tensor& x) {
        // x: (N, C*H*W) in channel-major layout
        c1.set_spatial(H, W);
        Tensor h = a1.forward(c1.forward(x));    // (N, C1*H*W)
        p1.set_spatial(C1, H, W);
        h = p1.forward(h);                        // (N, C1*poolH*poolW)

        c2.set_spatial(poolH, poolW);
        h = a2.forward(c2.forward(h));            // (N, C2*poolH*poolW)
        p2.set_spatial(C2, poolH, poolW);
        h = p2.forward(h);                        // (N, flat_dim)

        Tensor flat(h.rows, flat_dim);
        for (int i = 0; i < h.rows; i++)
            for (int j = 0; j < flat_dim; j++)
                flat.at(i, j) = h.at(i, j);

        Tensor hh = f1.forward(flat);
        return f2.forward(hh);
    }

    // Returns d(input); callers that ignore it (the CLI demo) still compile.
    Tensor backward(const Tensor& dlogits) {
        Tensor d = f2.backward(dlogits);
        Tensor dflat = f1.backward(d);            // (N, flat_dim)

        Tensor dh(dflat.rows, flat_dim);
        for (int i = 0; i < dh.rows; i++)
            for (int j = 0; j < flat_dim; j++)
                dh.at(i, j) = dflat.at(i, j);

        Tensor d2 = p2.backward(dh);              // dL/da2.out
        d2 = c2.backward(a2.backward(d2));        // ReLU first, then conv
        Tensor d1 = p1.backward(d2);              // dL/da1.out
        return c1.backward(a1.backward(d1));      // ReLU first, then conv
    }

    std::vector<Tensor*> params() {
        auto p = c1.params();
        for (auto* pp : c2.params()) p.push_back(pp);
        p.push_back(&f1.w); p.push_back(&f1.b);
        p.push_back(&f2.w); p.push_back(&f2.b);
        return p;
    }
    std::vector<Tensor*> grads() {
        auto g = c1.grads();
        for (auto* gp : c2.grads()) g.push_back(gp);
        g.push_back(&f1.gw); g.push_back(&f1.gb);
        g.push_back(&f2.gw); g.push_back(&f2.gb);
        return g;
    }
};

#ifndef SHADE_CNN_NO_MAIN
void demo_mnist() {
    setbuf(stdout, NULL);
    printf("=== MNIST CNN ===\n");
    srand(42);

    MNIST mnist;
    if (!mnist.load("data", 20000, 5000)) return;

    CNN model;
    AdamW opt(0.001f, 0.0f);
    for (auto* p : model.params()) opt.add(*p);

    int N = mnist.train_x.rows;
    int Nt = mnist.test_x.rows;
    int B = 200, epochs = 30;

    printf("  params: (%d + %d + %d + %d) x 2 = %d\n",
           model.c1.w.size(), model.c2.w.size(), model.f1.w.size(), model.f2.w.size(),
           model.c1.w.size() + model.c2.w.size() + model.f1.w.size() + model.f2.w.size()
           + model.f1.b.size() + model.f2.b.size() + model.c1.b.size() + model.c2.b.size());

    for (int ep = 0; ep < epochs; ep++) {
        float tl = 0;
        int batches = 0;

        for (int b = 0; b < N; b += B) {
            auto pp = model.params();
            auto gp = model.grads();
            opt.zero_grad(gp);
            int bs = (b + B <= N) ? B : N - b;
            Tensor bx = mnist.train_x.slice(b, bs);
            Tensor by = mnist.train_y.slice(b, bs);

            Tensor logits = model.forward(bx);
            Tensor d;
            tl += CrossEntropyLoss::forward(logits, by, d);
            model.backward(d);
            opt.step(pp, gp);
            batches++;
        }

        if (ep == epochs - 1) {
            int correct = 0;
            int eval_batch = 200;
            for (int i = 0; i < Nt; i += eval_batch) {
                int bs = (i + eval_batch <= Nt) ? eval_batch : Nt - i;
                Tensor bx = mnist.test_x.slice(i, bs);
                Tensor logits = model.forward(bx);
                for (int s = 0; s < bs; s++) {
                    int pred = 0;
                    for (int j = 1; j < 10; j++) if (logits.at(s, j) > logits.at(s, pred)) pred = j;
                    int truth = 0;
                    for (int j = 1; j < 10; j++) if (mnist.test_y.at(i + s, j) > mnist.test_y.at(i + s, truth)) truth = j;
                    if (pred == truth) correct++;
                }
            }
            printf("  ep %2d loss=%.4f test_acc=%d/%d (%.1f%%)\n",
                   ep, tl / batches, correct, Nt, 100.0f * correct / Nt);
        }
    }
}

int main() {
    demo_mnist();
    return 0;
}
#endif // SHADE_CNN_NO_MAIN
