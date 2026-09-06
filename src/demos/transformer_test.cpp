#include "../core/tensor.h"
#include "../core/loss.h"
#include "../layers/transformer.h"
#include "../optimizer/adamw.h"
#include <cstdio>
#include <cmath>

// Tiny transformer: copy task (predict same sequence)
void demo_transformer() {
    printf("=== Transformer Block (copy sequence) ===\n");
    srand(42);

    int T = 3, D = 4, N = 80;
    Tensor X(N, T * D), Y(N, T * D);
    for (int i = 0; i < N; i++) {
        for (int t = 0; t < T; t++)
            for (int j = 0; j < D; j++) {
                float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
                X.at(i, t * D + j) = v;
                Y.at(i, t * D + j) = v;
            }
    }

    TransformerBlock tb(D);
    Dense out(D, D, false);
    AdamW opt(0.005f, 0.0001f);
    for (auto* p : tb.params()) opt.add(*p);
    opt.add(out.w); opt.add(out.b);

    auto mse = [](const Tensor& p, const Tensor& t, Tensor& d) -> float {
        float loss = 0;
        d = Tensor(p.rows, p.cols);
        for (int i = 0; i < p.size(); i++) {
            float diff = p.data[i] - t.data[i];
            loss += diff * diff;
            d.data[i] = 2.0f * diff / p.size();
        }
        return loss / p.size();
    };

    for (int ep = 0; ep < 150; ep++) {
        auto pp = tb.params(); pp.push_back(&out.w); pp.push_back(&out.b);
        auto gp = tb.grads(); gp.push_back(&out.gw); gp.push_back(&out.gb);
        opt.zero_grad(gp);
        float tl = 0;

        for (int s = 0; s < N; s++) {
            Tensor x(T, D);
            for (int t = 0; t < T; t++) for (int j = 0; j < D; j++) x.at(t, j) = X.at(s, t * D + j);
            Tensor target(T, D);
            for (int t = 0; t < T; t++) for (int j = 0; j < D; j++) target.at(t, j) = Y.at(s, t * D + j);

            Tensor tb_out = tb.forward(x);
            Tensor pred = out.forward(tb_out);

            Tensor d;
            tl += mse(pred, target, d);

            // backward through out
            Tensor d_tb = out.backward(d);
            // backward through transformer
            tb.backward(d_tb);
        }

        opt.step(pp, gp);
        if (ep % 50 == 0) printf("  ep %d loss=%.6f\n", ep, tl / N);
    }

    // test
    float err = 0;
    for (int t = 0; t < 5; t++) {
        Tensor x(T, D);
        for (int i = 0; i < T; i++) for (int j = 0; j < D; j++) x.at(i, j) = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        Tensor target = x.clone();
        Tensor pred = out.forward(tb.forward(x));
        for (int i = 0; i < pred.size(); i++) err += fabs(pred.data[i] - target.data[i]);
    }
    printf("  test avg err: %.4f\n", err / (5 * T * D));
}

int main() {
    demo_transformer();
    printf("\n===== DONE =====\n");
    return 0;
}
