// Finite-difference gradient checking, independent of analytic backward.
// cl /nologo /std:c++17 /O2 /arch:AVX /openmp /EHsc /I src tests\gradcheck.cpp
#include "../src/core/tensor.h"
#include "../src/layers/dense.h"
#include "../src/layers/layernorm.h"
#include "../src/layers/gru.h"
#include "../src/layers/attention.h"
#include "../src/utils/embedding.h"
#include "../src/layers/transformer.h"
// Exercise the real GPT composition (embedding + positional encoding +
// transformer blocks + output head), not a reimplementation. Rename its CLI
// main so it does not clash with this test's main.
#define main gpt_cli_main
#include "../src/gpt.cpp"
#undef main
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
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
// expect_fail is used by the NaN sentinel self-test below: it inverts the
// result and suppresses output when a deliberately broken backward is fed in.
static bool gradcheck(const std::string& name,
                      Forward forward, Backward backward,
                      const std::vector<Tensor*>& params,
                      const std::vector<Tensor*>& grads,
                      const std::vector<Tensor*>& inputs,
                      bool expect_fail = false) {
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
    std::vector<Tensor> input_grads;
    backward(dout, input_grads);
    if (input_grads.size() != inputs.size()) {
        std::fprintf(stderr, "%s: backward returned %zu input grads, expected %zu\n",
                     name.c_str(), input_grads.size(), inputs.size());
        return false;
    }

    std::vector<std::vector<f32>> analytic;
    for (auto* g : grads) analytic.push_back(g->data);
    std::vector<std::vector<f32>> analytic_input;
    for (auto& g : input_grads) analytic_input.push_back(g.data);

    f32 worst = 0;
    std::string worst_at;
    // NaN > worst is false, so without this flag a non-finite gradient could
    // silently leave worst at 0 and report a false OK.
    bool finite_ok = true;

    auto numeric = [&](Tensor& p, i32 idx) {
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
                f32 num = numeric(*list[k], i);
                f32 a = ana[k][i];
                if (!std::isfinite(num) || !std::isfinite(a)) {
                    if (!expect_fail)
                        std::fprintf(stderr, "%s: %s[%zu].data[%d] non-finite gradient (numeric=%g, analytic=%g)\n",
                                     name.c_str(), kind.c_str(), k, i, (double)num, (double)a);
                    finite_ok = false;
                    continue;
                }
                f32 err = std::fabs(num - a) / std::fmax(1.0f, std::fmax(std::fabs(num), std::fabs(a)));
                if (!std::isfinite(err)) {
                    if (!expect_fail)
                        std::fprintf(stderr, "%s: %s[%zu].data[%d] non-finite error\n",
                                     name.c_str(), kind.c_str(), k, i);
                    finite_ok = false;
                    continue;
                }
                if (err > group_worst) { group_worst = err; group_at = std::to_string(i); }
                if (err > worst) { worst = err; worst_at = kind + "[" + std::to_string(k) + "].data[" + std::to_string(i) + "]"; }
            }
            if (std::getenv("GRAD_VERBOSE"))
                std::printf("  %s[%zu] worst %.3e at %s\n", kind.c_str(), k, group_worst, group_at.c_str());
        }
    };

    sweep(params, "param", analytic);
    sweep(inputs, "input", analytic_input);

    bool ok = finite_ok && worst <= tol;
    if (expect_fail) return !ok; // sentinel: rejection is the expected result
    std::printf("%-12s %s  worst rel err %.3e at %s\n",
                name.c_str(), ok ? "OK" : "FAIL", worst, worst_at.c_str());
    return ok;
}

// The training loop uses GPT::compute_loss (cross-entropy), not the MSE the
// generic harness uses to feed dout. Check that its dlogits really are the
// finite-difference gradient of the scalar mean loss.
static bool gpt_loss_gradcheck() {
    GPT gpt(8, 4, 1, 8);
    Tensor logits = random_tensor(5, 8);
    Tensor targets(1, 5);
    for (i32 i = 0; i < targets.cols; i++) targets.data[i] = (f32)(i % 8);
    Tensor scratch(5, 8), dlogits(5, 8);
    gpt.compute_loss(logits, targets, dlogits);

    const f32 eps = 1e-3f;
    f32 worst = 0;
    bool finite_ok = true;
    for (i32 i = 0; i < logits.size(); i++) {
        f32 orig = logits.data[i];
        logits.data[i] = orig + eps;
        f32 lp = gpt.compute_loss(logits, targets, scratch);
        logits.data[i] = orig - eps;
        f32 lm = gpt.compute_loss(logits, targets, scratch);
        logits.data[i] = orig;
        f32 num = (lp - lm) / (2.0f * eps);
        f32 a = dlogits.data[i];
        if (!std::isfinite(num) || !std::isfinite(a)) {
            std::fprintf(stderr, "GPTLoss: non-finite gradient at logit %d (numeric=%g, analytic=%g)\n",
                         i, (double)num, (double)a);
            finite_ok = false;
            continue;
        }
        f32 err = std::fabs(num - a) / std::fmax(1.0f, std::fmax(std::fabs(num), std::fabs(a)));
        if (!std::isfinite(err)) {
            std::fprintf(stderr, "GPTLoss: non-finite error at logit %d\n", i);
            finite_ok = false;
            continue;
        }
        if (err > worst) worst = err;
    }
    bool ok = finite_ok && worst <= 2e-3f;
    std::printf("%-12s %s  worst rel err %.3e\n", "GPTLoss", ok ? "OK" : "FAIL", worst);
    return ok;
}

// Regression for the NaN trap: `NaN > worst` is false, so a non-finite
// gradient used to be able to leave worst at 0 and report a false OK. Probe
// both paths: a NaN analytic gradient and a NaN numeric (forward) gradient.
// expect_fail inverts the harness result and silences the expected diagnostics.
static bool nan_guard_selftests() {
    const f32 nan = std::numeric_limits<f32>::quiet_NaN();
    bool ok = true;
    {
        Dense fc(2, 2);
        randomize(fc.w); randomize(fc.b);
        Tensor x = random_tensor(3, 2);
        bool rejected = gradcheck("NaNSentinelA",
            [&] { return fc.forward(x); },
            [&](const Tensor& dout, std::vector<Tensor>& ig) {
                ig.push_back(fc.backward(dout));
                for (i32 i = 0; i < fc.gw.size(); i++) fc.gw.data[i] = nan; // poison analytic
            },
            {&fc.w}, {&fc.gw}, {&x}, true);
        if (!rejected) {
            std::fprintf(stderr, "gradcheck: NaN analytic gradient was not rejected\n");
            ok = false;
        }
    }
    {
        Dense fc(2, 2);
        randomize(fc.w); randomize(fc.b);
        Tensor x = random_tensor(3, 2);
        bool rejected = gradcheck("NaNSentinelN",
            [&] { Tensor y = fc.forward(x); y.data[0] = nan; return y; }, // poison numeric
            [&](const Tensor& dout, std::vector<Tensor>& ig) {
                Tensor dx = fc.backward(dout);
                for (i32 i = 0; i < fc.gw.size(); i++) fc.gw.data[i] = 0.0f; // keep analytic finite
                for (i32 i = 0; i < dx.size(); i++) dx.data[i] = 0.0f;
                ig.push_back(dx);
            },
            {&fc.w}, {&fc.gw}, {&x}, true);
        if (!rejected) {
            std::fprintf(stderr, "gradcheck: NaN numeric gradient was not rejected\n");
            ok = false;
        }
    }
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
            [&](const Tensor& dout, std::vector<Tensor>& ig) { ig.push_back(fc.backward(dout)); },
            {&fc.w, &fc.b}, {&fc.gw, &fc.gb}, {&x});
    }

    {
        LayerNorm ln(5);
        randomize(ln.gamma); randomize(ln.beta);
        Tensor x = random_tensor(4, 5);
        ok &= gradcheck("LayerNorm",
            [&] { return ln.forward(x); },
            [&](const Tensor& dout, std::vector<Tensor>& ig) { ig.push_back(ln.backward(dout)); },
            {&ln.gamma, &ln.beta}, {&ln.ggamma, &ln.gbeta}, {&x});
    }

    {
        GRU gru(3, 3);
        for (auto* p : gru.params()) randomize(*p);
        Tensor x = random_tensor(4, 3);
        Tensor h0 = random_tensor(1, 3);
        ok &= gradcheck("GRU",
            [&] { return gru.forward(x, h0); },
            [&](const Tensor& dout, std::vector<Tensor>& ig) {
                Tensor dh0;
                ig.push_back(gru.backward(dout, &dh0));
                ig.push_back(std::move(dh0));
            },
            gru.params(), gru.grads(), {&x, &h0});
    }

    {
        SelfAttention att(6, 2, true);
        for (auto* p : att.params()) randomize(*p);
        Tensor x = random_tensor(4, 6);
        ok &= gradcheck("Attention",
            [&] { return att.forward(x); },
            [&](const Tensor& dout, std::vector<Tensor>& ig) { ig.push_back(att.backward(dout)); },
            att.params(), att.grads(), {&x});
    }

    {
        // Discrete ids have no numeric gradient, so only the embedding table
        // is checked. backward() accumulates into gw; the harness zeros it
        // before the single analytic call and numeric forward() never writes
        // grads. Every vocab row is present in the sequence so all weights
        // receive a non-zero finite-difference signal.
        Embedding emb(8, 4);
        randomize(emb.w);
        // Repeated IDs exercise gradient accumulation into gw for the same
        // token; all vocab rows still appear so no weight is left untested.
        const f32 seq[] = { 0, 1, 0, 2, 1, 0, 3, 1, 4, 5, 6, 7 };
        Tensor ids(1, (i32)(sizeof(seq) / sizeof(seq[0])));
        for (i32 i = 0; i < ids.cols; i++) ids.data[i] = seq[i];
        ok &= gradcheck("Embedding",
            [&] { return emb.forward(ids); },
            [&](const Tensor& dout, std::vector<Tensor>&) { emb.backward(dout, ids); },
            {&emb.w}, {&emb.gw}, {});
    }

    {
        TransformerBlock tb(8);   // embed_dim divisible by 4 -> FFN dim 32
        for (auto* p : tb.params()) randomize(*p);
        Tensor x = random_tensor(5, 8);
        ok &= gradcheck("TransformerBlock",
            [&] { return tb.forward(x); },
            [&](const Tensor& dout, std::vector<Tensor>& ig) { ig.push_back(tb.backward(dout)); },
            tb.params(), tb.grads(), {&x});
    }

    {
        // Small real GPT: 8 vocab, dim 4 (4 heads, head_dim 1), TWO blocks so
        // the reversed block loop in GPT::backward is actually exercised.
        // Input ids are discrete, so only parameter gradients are compared.
        GPT gpt(8, 4, 2, 8);
        for (auto* p : gpt.params()) randomize(*p);
        Tensor ids(1, 5);
        for (i32 i = 0; i < ids.cols; i++) ids.data[i] = (f32)(i % 8);
        ok &= gradcheck("GPT",
            [&] { return gpt.forward(ids); },
            [&](const Tensor& dout, std::vector<Tensor>&) { gpt.backward(dout, ids); },
            gpt.params(), gpt.grads(), {});
    }

    ok &= gpt_loss_gradcheck();
    ok &= nan_guard_selftests();

    if (!ok) {
        std::fprintf(stderr, "gradcheck: FAILED\n");
        return 1;
    }
    std::puts("gradcheck: OK");
    return 0;
}
