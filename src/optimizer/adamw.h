#pragma once
#include "../core/tensor.h"
#include <vector>

struct AdamW {
    f32 lr, beta1, beta2, eps, wd;
    i32 t;
    std::vector<Tensor> m, v;

    AdamW(f32 lr = 1e-3f, f32 wd = 0.01f)
        : lr(lr), beta1(0.9f), beta2(0.999f), eps(1e-8f), wd(wd), t(0) {}

    void add(Tensor& p) {
        m.push_back(Tensor(p.rows, p.cols, 0.0f));
        v.push_back(Tensor(p.rows, p.cols, 0.0f));
    }

    void step(const std::vector<Tensor*>& params, const std::vector<Tensor*>& grads) {
        t++;
        // bias-correction factors + fused step coefficients — computed ONCE per step
        f32 p1t = std::pow(beta1, t);
        f32 p2t = std::pow(beta2, t);
        f32 sb2 = std::sqrt(1.0f - p2t);
        f32 ib1 = 1.0f / (1.0f - p1t);
        f32 ib2 = 1.0f / (1.0f - p2t);
        f32 km = lr * sb2 * ib1 * ib1;   // update multiplier
        f32 kw = lr * sb2 * ib1 * wd;    // weight-decay multiplier
        for (size_t i = 0; i < params.size(); i++) {
            const f32* g = grads[i]->data.data();
            f32* p = params[i]->data.data();
            f32* pm = m[i].data.data();
            f32* pv = v[i].data.data();
            const i32 n = params[i]->size();
            for (i32 j = 0; j < n; j++) {
                f32 gi = g[j];
                f32 mi = beta1 * pm[j] + (1 - beta1) * gi;
                f32 vi = beta2 * pv[j] + (1 - beta2) * gi * gi;
                pm[j] = mi; pv[j] = vi;
                p[j] -= km * mi / (std::sqrt(vi * ib2) + eps);
                p[j] -= kw * p[j];
            }
        }
    }

    void zero_grad(const std::vector<Tensor*>& grads) {
        for (auto* g : grads) g->fill(0.0f);
    }
};
