// Finite-difference gradient checking, independent of analytic backward.
// cl /nologo /std:c++17 /O2 /arch:AVX /openmp /EHsc /I src tests\gradcheck.cpp
#include "../src/core/tensor.h"
#include "../src/layers/dense.h"
#include "../src/layers/layernorm.h"
#include "../src/layers/gru.h"
#include "../src/layers/attention.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(2026);
static std::uniform_real_distribution<f32> dist(-0.6f, 0.6f);

static Tensor random_tensor(i32 r, i32 c) {
    Tensor t(r, c);
    for (i32 i = 0; i < t.size(); i++) t.data[i] = dist(rng);
    return t;
}

static void randomize(Tensor& t) {
    for (i32 i = 0; i < t.size(); i++) t.data[i] = dist(rng);
}

// d(loss)/d(out) = (out - target) / N; loss = 0.5 * sum((out-target)^2) / N
static f32 mse(const Tensor& pred, const Tensor& target, Tensor* dout) {
    f32 loss = 0;
    if (dout) *dout = Tensor(pred.rows, pred.cols);
    for (i32 i = 0; i < pred.size(); i++) {
        f32 d = pred.data[i] - target.data[i];
        loss += 0.5f * d * d;
        if (dout) dout->data[i] = d / pred.size();
    }
    return loss / pred.size();
}

template <typename Forward, typename Backward>
static bool gradcheck(const std::string& name,
                      Forward forward, Backward backward,
                      const std::vector<Tensor*>& params,
                      const std::vector<Tensor*>& grads,
                      const std::vector<Tensor*>& inputs) {
    const f32 eps = 1e-3f;
#ifndef GRAD_TOL
#define GRAD_TOL 2e-3f
#endif
    const f32 tol = GRAD_TOL;
    for (auto* g : grads) g->fill(0.0f);
    Tensor out = forward();
    Tensor target = random_tensor(out.rows, out.cols);
    Tensor dout;
    mse(out, target, &dout);
    Tensor dx = backward(dout);

    std::vector<std::vector<f32>> analytic;
    for (auto* g : grads) analytic.push_back(g->data);

    f32 worst = 0;
    std::string worst_at;
    auto compare = [&](f32 num, f32 ana, const std::string& at) {
        f32 err = std::fabs(num - ana) / std::fmax(1.0f, std::fmax(std::fabs(num), std::fabs(ana)));
        if (err > worst) { worst = err; worst_at = at; }
    };

    auto numeric = [&](Tensor& p, i32 idx, f32 base) {
        f32 orig = p.data[idx];
        p.data[idx] = orig + eps;
        f32 lp = mse(forward(), target, nullptr);
        p.data[idx] = orig - eps;
        f32 lm = mse(forward(), target, nullptr);
        p.data[idx] = orig;
        return (lp - lm) / (2.0f * eps);
    };

    auto sweep = [&](const std::vector<Tensor*>& list, const std::string& kind,
                     const std::vector<std::vector<f32>>& ana) {
        for (size_t k = 0; k < list.size(); k++) {
            f32 group_worst = 0;
            std::string group_at;
            for (i32 i = 0; i < list[k]->size(); i++) {
                f32 num = numeric(*list[k], i, 0.0f);
                f32 a = ana[k][i];
                f32 err = std::fabs(num - a) / std::fmax(1.0f, std::fmax(std::fabs(num), std::fabs(a)));
                if (err > group_worst) { group_worst = err; group_at = std::to_string(i); }
                if (err > worst) { worst = err; worst_at = kind + "[" + std::to_string(k) + "].data[" + std::to_string(i) + "]"; }
            }
            if (std::getenv("GRAD_VERBOSE"))
                std::printf("  %s[%zu] worst %.3e at %s\n", kind.c_str(), k, group_worst, group_at.c_str());
        }
    };

    sweep(params, "param", analytic);
    sweep(inputs, "input", { dx.data });

    bool ok = worst <= tol;
    std::printf("%-12s %s  worst rel err %.3e at %s\n",
                name.c_str(), ok ? "OK" : "FAIL", worst, worst_at.c_str());
    return ok;
}

int main() {
    srand(2026);
    bool ok = true;

    {
        Dense fc(4, 3, true);
        randomize(fc.w); randomize(fc.b);
        Tensor x = random_tensor(5, 4);
        ok &= gradcheck("Dense",
            [&] { return fc.forward(x); },
            [&](const Tensor& dout) { return fc.backward(dout); },
            {&fc.w, &fc.b}, {&fc.gw, &fc.gb}, {&x});
    }

    {
        LayerNorm ln(5);
        randomize(ln.gamma); randomize(ln.beta);
        Tensor x = random_tensor(4, 5);
        ok &= gradcheck("LayerNorm",
            [&] { return ln.forward(x); },
            [&](const Tensor& dout) { return ln.backward(dout); },
            {&ln.gamma, &ln.beta}, {&ln.ggamma, &ln.gbeta}, {&x});
    }

    {
        GRU gru(3, 3);
        for (auto* p : gru.params()) randomize(*p);
        Tensor x = random_tensor(4, 3);
        Tensor h0 = random_tensor(1, 3);
        ok &= gradcheck("GRU",
            [&] { return gru.forward(x, h0); },
            [&](const Tensor& dout) { return gru.backward(dout); },
            gru.params(), gru.grads(), {&x});
    }

    {
        SelfAttention att(6, 2, true);
        for (auto* p : att.params()) randomize(*p);
        Tensor x = random_tensor(4, 6);
        ok &= gradcheck("Attention",
            [&] { return att.forward(x); },
            [&](const Tensor& dout) { return att.backward(dout); },
            att.params(), att.grads(), {&x});
    }

    if (!ok) {
        std::fprintf(stderr, "gradcheck: FAILED\n");
        return 1;
    }
    std::puts("gradcheck: OK");
    return 0;
}
