#include "../core/tensor.h"
#include "../core/loss.h"
#include "../layers/dense.h"
#include "../layers/activations.h"
#include "../optimizer/adamw.h"
#include "../utils/mnist.h"
#include <cstdio>
#include <cmath>

struct MLP {
    Dense f1, f2, f3;
    Act a1, a2;

    MLP() : f1(784, 256, true), f2(256, 128, true), f3(128, 10, false),
            a1(Activation::RELU), a2(Activation::RELU) {}

    Tensor forward(const Tensor& x) {
        Tensor h = a1.forward(f1.forward(x));
        h = a2.forward(f2.forward(h));
        return f3.forward(h);
    }

    void backward(const Tensor& d) {
        Tensor h = f3.backward(d);
        h = f2.backward(a2.backward(h));
        f1.backward(a1.backward(h));
    }

    std::vector<Tensor*> params() { return {&f1.w, &f1.b, &f2.w, &f2.b, &f3.w, &f3.b}; }
    std::vector<Tensor*> grads() { return {&f1.gw, &f1.gb, &f2.gw, &f2.gb, &f3.gw, &f3.gb}; }
};

int main() {
    setbuf(stdout, NULL);
    printf("=== MNIST MLP ===\n");
    srand(42);

    MNIST mnist;
    if (!mnist.load("data", 60000, 10000)) return 1;

    MLP model;
    AdamW opt(0.005f, 0.0f);
    for (auto* p : model.params()) opt.add(*p);

    int N = mnist.train_x.rows, Nt = mnist.test_x.rows;
    int B = 500, epochs = 15;

    for (int ep = 0; ep < epochs; ep++) {
        float tl = 0;
        int bat = 0;
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
            bat++;
        }

        if (ep == epochs - 1) {
            int correct = 0;
            int eval_batch = 100;
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
            printf("  ep %3d loss=%.4f test_acc=%d/%d (%.1f%%)\n",
                   ep, tl / bat, correct, Nt, 100.0f * correct / Nt);
        }
    }

    return 0;
}
