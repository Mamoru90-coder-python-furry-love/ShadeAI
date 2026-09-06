#include "../core/tensor.h"
#include "../core/loss.h"
#include "../layers/conv2d.h"
#include "../layers/pool.h"
#include "../layers/dense.h"
#include "../layers/activations.h"
#include "../optimizer/adamw.h"
#include "../utils/mnist.h"
#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>

// CNN: Conv(1->8,3x3) + ReLU + Pool(2x2) -> Conv(8->16,3x3) + ReLU + Pool(2x2) -> Dense(400->128) -> Dense(128->10)
// Shapes:
//   28x28x1 -> Conv(3x3,p=1) -> 28x28x8 -> Pool(2x2) -> 14x14x8
//   -> Conv(3x3,p=1) -> 14x14x16 -> Pool(2x2) -> 7x7x16 = 784
//   -> Dense(784->128) -> Dense(128->10)

struct CNN {
    Conv2D c1, c2;
    Act a1, a2;
    MaxPool2D p1, p2;
    Dense f1, f2;

    CNN() : c1(1, 8, 3, 3, 1, 1), a1(Activation::RELU), p1(2, 2, 2),
            c2(8, 16, 3, 3, 1, 1), a2(Activation::RELU), p2(2, 2, 2),
            f1(784, 128, true), f2(128, 10, false) {}

    Tensor forward(const Tensor& x) {
        // x: (N, 784) — reshape to (N, 1*28*28) conceptually same layout
        c1.set_spatial(28, 28);
        Tensor h = a1.forward(c1.forward(x));   // (N, 8*28*28)
        p1.set_spatial(8, 28, 28);
        h = p1.forward(h);                       // (N, 8*14*14)

        c2.set_spatial(14, 14);
        h = a2.forward(c2.forward(h));           // (N, 16*14*14)
        p2.set_spatial(16, 14, 14);
        h = p2.forward(h);                       // (N, 16*7*7) = (N, 784)

        Tensor flat(h.rows, 784);
        for (int i = 0; i < h.rows; i++)
            for (int j = 0; j < 784; j++)
                flat.at(i, j) = h.at(i, j);

        Tensor h1 = f1.forward(flat);
        return f2.forward(h1);
    }

    void backward(const Tensor& dlogits) {
        Tensor d = f2.backward(dlogits);
        Tensor dflat = f1.backward(d);            // (N, 784)

        // reshape dflat to (N, 16*7*7)
        Tensor dh = Tensor(dflat.rows, 16 * 7 * 7);
        for (int i = 0; i < dh.rows; i++)
            for (int j = 0; j < 16 * 7 * 7; j++)
                dh.at(i, j) = dflat.at(i, j);

        Tensor d2 = p2.backward(dh);              // (N, 16*14*14)
        d2 = a2.backward(c2.backward(d2));         // (N, 8*14*14)
        Tensor d1 = p1.backward(d2);              // (N, 8*28*28)
        a1.backward(c1.backward(d1));             // (N, 1*28*28) — discarded, no prev layer
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
