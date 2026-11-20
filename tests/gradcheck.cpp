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
// The char-RNN demo has no main, so it can be included directly to check the
// real embedding -> GRU (BPTT) -> dense composition.
#include "../src/demos/char_rnn.cpp"
// The transformer-LM demo has a CLI main; rename it to check the real
// Embedding -> PositionalEncoding -> TransformerBlocks -> Dense composition.
#define main transformer_lm_cli_main
#include "../src/demos/transformer_lm.cpp"
#undef main
// Include only the MLP struct from the MNIST demo (its CLI main pulls in the
// MNIST loader / stb_image.h). Checks the real
// Dense -> ReLU -> Dense -> ReLU -> Dense composition on a tiny model.
#define SHADE_MLP_NO_MAIN
#include "../src/demos/mnist_mlp.cpp"
#undef SHADE_MLP_NO_MAIN
// Same pattern for the CNN demo: include only the model struct, not the CLI
// (which pulls in the missing stb_image.h via the MNIST loader).
#define SHADE_CNN_NO_MAIN
#include "../src/demos/mnist_cnn.cpp"
#undef SHADE_CNN_NO_MAIN
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
// Shared by every finite-difference harness and by the ReLU margin guard, so
// the two cannot drift apart.
static const f32 GRAD_EPS = 1e-3f;

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
// expect_fail is used by the NaN sentinel self-test below: it requires that
// the harness failed *specifically* because it saw a non-finite value, so a
// failure for any other reason cannot make the sentinel look successful.
static bool gradcheck(const std::string& name,
                      Forward forward, Backward backward,
                      const std::vector<Tensor*>& params,
                      const std::vector<Tensor*>& grads,
                      const std::vector<Tensor*>& inputs,
                      bool expect_fail = false) {
    const f32 eps = GRAD_EPS;
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
    // silently leave worst at 0 and report a false OK. nonfinite_seen lets the
    // sentinel self-test prove the *reason* for a rejection.
    bool finite_ok = true;
    bool nonfinite_seen = false;

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
                    nonfinite_seen = true;
                    continue;
                }
                f32 err = std::fabs(num - a) / std::fmax(1.0f, std::fmax(std::fabs(num), std::fabs(a)));
                if (!std::isfinite(err)) {
                    if (!expect_fail)
                        std::fprintf(stderr, "%s: %s[%zu].data[%d] non-finite error\n",
                                     name.c_str(), kind.c_str(), k, i);
                    finite_ok = false;
                    nonfinite_seen = true;
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
    // Sentinel: reject only if a non-finite value was the actual cause.
    if (expect_fail) return !ok && nonfinite_seen;
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

    const f32 eps = GRAD_EPS;
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

// CharRNN's training loop used to emit unscaled (sum) dlogits while reporting
// a mean loss. Check the extracted compute_loss the same way as GPTLoss.
static bool charrnn_loss_gradcheck() {
    CharRNN model(8, 4, 6);
    Tensor logits = random_tensor(5, 8);
    Tensor targets(1, 5);
    for (i32 i = 0; i < targets.cols; i++) targets.data[i] = (f32)(i % 8);
    Tensor scratch(5, 8), dlogits(5, 8);
    model.compute_loss(logits, targets, dlogits);

    const f32 eps = GRAD_EPS;
    f32 worst = 0;
    bool finite_ok = true;
    for (i32 i = 0; i < logits.size(); i++) {
        f32 orig = logits.data[i];
        logits.data[i] = orig + eps;
        f32 lp = model.compute_loss(logits, targets, scratch);
        logits.data[i] = orig - eps;
        f32 lm = model.compute_loss(logits, targets, scratch);
        logits.data[i] = orig;
        f32 num = (lp - lm) / (2.0f * eps);
        f32 a = dlogits.data[i];
        if (!std::isfinite(num) || !std::isfinite(a)) {
            std::fprintf(stderr, "CharRNNLoss: non-finite gradient at logit %d (numeric=%g, analytic=%g)\n",
                         i, (double)num, (double)a);
            finite_ok = false;
            continue;
        }
        f32 err = std::fabs(num - a) / std::fmax(1.0f, std::fmax(std::fabs(num), std::fabs(a)));
        if (!std::isfinite(err)) {
            std::fprintf(stderr, "CharRNNLoss: non-finite error at logit %d\n", i);
            finite_ok = false;
            continue;
        }
        if (err > worst) worst = err;
    }
    bool ok = finite_ok && worst <= 2e-3f;
    std::printf("%-12s %s  worst rel err %.3e\n", "CharRNNLoss", ok ? "OK" : "FAIL", worst);
    return ok;
}

// Same check for transformer_lm's extracted compute_loss. The demo previously
// fed unscaled (sum) dlogits to backward while printing a mean loss.
static bool transformer_lm_loss_gradcheck() {
    TransformerLM model(8, 4, 1);
    Tensor logits = random_tensor(5, 8);
    Tensor targets(1, 5);
    for (i32 i = 0; i < targets.cols; i++) targets.data[i] = (f32)(i % 8);
    Tensor scratch(5, 8), dlogits(5, 8);
    model.compute_loss(logits, targets, dlogits);

    const f32 eps = GRAD_EPS;
    f32 worst = 0;
    bool finite_ok = true;
    for (i32 i = 0; i < logits.size(); i++) {
        f32 orig = logits.data[i];
        logits.data[i] = orig + eps;
        f32 lp = model.compute_loss(logits, targets, scratch);
        logits.data[i] = orig - eps;
        f32 lm = model.compute_loss(logits, targets, scratch);
        logits.data[i] = orig;
        f32 num = (lp - lm) / (2.0f * eps);
        f32 a = dlogits.data[i];
        if (!std::isfinite(num) || !std::isfinite(a)) {
            std::fprintf(stderr, "TransformerLMLoss: non-finite gradient at logit %d (numeric=%g, analytic=%g)\n",
                         i, (double)num, (double)a);
            finite_ok = false;
            continue;
        }
        f32 err = std::fabs(num - a) / std::fmax(1.0f, std::fmax(std::fabs(num), std::fabs(a)));
        if (!std::isfinite(err)) {
            std::fprintf(stderr, "TransformerLMLoss: non-finite error at logit %d\n", i);
            finite_ok = false;
            continue;
        }
        if (err > worst) worst = err;
    }
    bool ok = finite_ok && worst <= 2e-3f;
    std::printf("%-16s %s  worst rel err %.3e\n", "TransformerLMLoss", ok ? "OK" : "FAIL", worst);
    return ok;
}

// compute_loss is reachable outside the demo now; verify the input guards
// reject malformed data instead of indexing past the row or dividing by zero.
static bool charrnn_loss_validation_check() {
    CharRNN model(8, 4, 6);
    Tensor dlogits;
    bool ok = true;
    auto rejects = [&](const char* what, const Tensor& logits, const Tensor& targets) {
        bool threw = false;
        try { model.compute_loss(logits, targets, dlogits); }
        catch (const std::invalid_argument&) { threw = true; }
        if (!threw) {
            std::fprintf(stderr, "CharRNNVal: %s was not rejected\n", what);
            ok = false;
        }
    };
    Tensor good_logits = random_tensor(5, 8);
    Tensor good_targets(1, 5);
    for (i32 i = 0; i < good_targets.cols; i++) good_targets.data[i] = (f32)(i % 8);

    rejects("wrong logits width", random_tensor(5, 7), good_targets);
    rejects("empty sequence", Tensor(), Tensor());
    rejects("length mismatch", good_logits, Tensor(1, 4));

    Tensor t = good_targets.clone(); t.data[0] = -1.0f;
    rejects("negative target", good_logits, t);
    t = good_targets.clone(); t.data[0] = 8.0f;
    rejects("target >= vocab", good_logits, t);
    t = good_targets.clone(); t.data[0] = 1.5f;
    rejects("non-integral target", good_logits, t);
    // Non-finite / huge values must be rejected BEFORE any float->int cast:
    // the cast is undefined for NaN, infinities, and out-of-int-range floats.
    t = good_targets.clone(); t.data[0] = std::numeric_limits<f32>::quiet_NaN();
    rejects("NaN target", good_logits, t);
    t = good_targets.clone(); t.data[0] = std::numeric_limits<f32>::infinity();
    rejects("+Inf target", good_logits, t);
    t = good_targets.clone(); t.data[0] = -std::numeric_limits<f32>::infinity();
    rejects("-Inf target", good_logits, t);
    t = good_targets.clone(); t.data[0] = 1e30f;
    rejects("huge finite target", good_logits, t);

    // NOTE: the vocab_size > 0 guard in compute_loss is not directly testable
    // through a real CharRNN: its Dense head already rejects a zero output
    // dimension at construction, so no valid model can reach that branch.

    std::printf("%-12s %s\n", "CharRNNVal", ok ? "OK" : "FAIL");
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

// ReLU has a kink at 0: if a pre-activation sits within GRAD_EPS of it, the
// central difference sees a different branch than the analytic derivative and
// the check fails for a reason unrelated to the backward math. Rather than
// trusting the RNG seed, regenerate the tiny model until every pre-activation
// is at least RELU_MARGIN_FACTOR * GRAD_EPS away from zero. Returns the smallest
// margin found, or a negative value if no acceptable draw was found.
//
// Note: f1/f2 are constructed with use_relu=true, so Dense::forward already
// clamps in place and does not cache the raw pre-activation. Act::last_in is
// therefore post-ReLU (mostly zeros) and cannot be used here; recompute the
// raw values directly. f3 has use_relu=false and needs no margin.
static const f32 RELU_MARGIN_FACTOR = 20.0f;
static f32 prepare_mlp_with_relu_margin(MLP& model, Tensor& x) {
    const f32 required = RELU_MARGIN_FACTOR * GRAD_EPS;
    for (int attempt = 0; attempt < 64; attempt++) {
        for (auto* p : model.params()) randomize(*p);
        x = random_tensor(3, 4);
        Tensor pre1 = x.matmul(model.f1.w).add(model.f1.b);
        Tensor h1 = pre1.clone();
        for (i32 i = 0; i < h1.size(); i++)
            if (h1.data[i] < 0.0f) h1.data[i] = 0.0f;
        Tensor pre2 = h1.matmul(model.f2.w).add(model.f2.b);

        f32 min_abs = std::numeric_limits<f32>::infinity();
        for (const Tensor* t : { &pre1, &pre2 })
            for (i32 i = 0; i < t->size(); i++)
                min_abs = std::fmin(min_abs, std::fabs(t->data[i]));
        if (min_abs >= required) return min_abs;
    }
    return -1.0f;
}

// CNN has three ReLU sites: after c1 (a1), after c2 (a2), and inside f1
// (Dense(use_relu=true), whose raw pre-activation is not cached). Conv2D does
// not clamp, so a1/a2.last_in are the raw conv outputs and can be used
// directly; f1.last_input is the flattened tensor, so its pre-activation is
// recomputed from the cached input. As with the MLP, a kink anywhere makes the
// finite difference disagree with backward for reasons unrelated to the math.
static f32 prepare_cnn_with_relu_margin(CNN& model, Tensor& x) {
    const f32 required = RELU_MARGIN_FACTOR * GRAD_EPS;
    for (int attempt = 0; attempt < 64; attempt++) {
        for (auto* p : model.params()) randomize(*p);
        x = random_tensor(1, model.H * model.W);
        model.forward(x); // fills a1/a2.last_in and f1.last_input
        Tensor pre_f1 = model.f1.last_input.matmul(model.f1.w).add(model.f1.b);
        f32 min_abs = std::numeric_limits<f32>::infinity();
        for (const Tensor* t : { &model.a1.last_in, &model.a2.last_in, &pre_f1 })
            for (i32 i = 0; i < t->size(); i++)
                min_abs = std::fmin(min_abs, std::fabs(t->data[i]));
        if (min_abs >= required) return min_abs;
    }
    return -1.0f;
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

    {
        // Real char-RNN composition from the demo: embedding -> GRU over the
        // sequence -> dense head. Token ids are discrete, so only parameter
        // gradients are compared; GRU's BPTT path is covered through the model.
        CharRNN model(8, 4, 6);
        for (auto* p : model.params()) randomize(*p);
        Tensor ids(1, 5);
        const f32 seq[] = { 0, 1, 0, 2, 3 };
        for (i32 i = 0; i < ids.cols; i++) ids.data[i] = seq[i];
        ok &= gradcheck("CharRNN",
            [&] { return model.forward(ids); },
            [&](const Tensor& dout, std::vector<Tensor>&) { model.backward(dout, ids); },
            model.params(), model.grads(), {});
    }

    ok &= gpt_loss_gradcheck();
    ok &= charrnn_loss_gradcheck();
    ok &= charrnn_loss_validation_check();
    ok &= nan_guard_selftests();

    // Kept last: randomize()/random_tensor() advance the shared RNG, so this
    // test must not shift the numbers reported by the checks above.
    {
        // Real MNIST MLP composition from the demo on a tiny model:
        // Dense -> ReLU -> Dense -> ReLU -> Dense. Input is continuous, so
        // both parameter and input gradients are checked.
        MLP model(4, 6, 5, 3);
        Tensor x;
        f32 margin = prepare_mlp_with_relu_margin(model, x);
        if (margin < 0.0f) {
            std::fprintf(stderr, "MLP: no draw with ReLU pre-activation margin >= %.3e\n",
                         RELU_MARGIN_FACTOR * GRAD_EPS);
            ok = false;
        } else {
            std::printf("MLP          ReLU margin %.3e (>= %.3e)\n",
                        margin, RELU_MARGIN_FACTOR * GRAD_EPS);
            ok &= gradcheck("MLP",
                [&] { return model.forward(x); },
                [&](const Tensor& dout, std::vector<Tensor>& ig) {
                    ig.push_back(model.backward(dout)); // MLP now returns d(input)
                },
                model.params(), model.grads(), {&x});
        }
    }

    // Also last (same RNG reason). Tiny spatial config keeps the full param
    // sweep cheap while exercising the real Conv -> ReLU -> MaxPool x2 ->
    // Dense -> Dense composition, including maxpool routing and conv col2im.
    {
        CNN model(6, 6, 1, 2, 3, 4, 3); // 6->3->1 spatial, flat_dim = 3, 108 params
        Tensor x;
        f32 margin = prepare_cnn_with_relu_margin(model, x);
        if (margin < 0.0f) {
            std::fprintf(stderr, "CNN: no draw with ReLU pre-activation margin >= %.3e\n",
                         RELU_MARGIN_FACTOR * GRAD_EPS);
            ok = false;
        } else {
            std::printf("CNN          ReLU margin %.3e (>= %.3e)\n",
                        margin, RELU_MARGIN_FACTOR * GRAD_EPS);
            ok &= gradcheck("CNN",
                [&] { return model.forward(x); },
                [&](const Tensor& dout, std::vector<Tensor>& ig) {
                    ig.push_back(model.backward(dout)); // CNN now returns d(input)
                },
                model.params(), model.grads(), {&x});
        }
    }

    // Last, for the same RNG reason. Checks the demo's own copy of the real
    // Embedding -> PositionalEncoding -> TransformerBlocks -> Dense stack
    // (two blocks, so the reversed block loop is exercised). Ids are discrete.
    {
        TransformerLM lm(8, 4, 2);
        for (auto* p : lm.params()) randomize(*p);
        Tensor ids(1, 5);
        for (i32 i = 0; i < ids.cols; i++) ids.data[i] = (f32)(i % 8);
        ok &= gradcheck("TransformerLM",
            [&] { return lm.forward(ids); },
            [&](const Tensor& dout, std::vector<Tensor>&) { lm.backward(dout, ids); },
            lm.params(), lm.grads(), {});
    }
    ok &= transformer_lm_loss_gradcheck();

    if (!ok) {
        std::fprintf(stderr, "gradcheck: FAILED\n");
        return 1;
    }
    std::puts("gradcheck: OK");
    return 0;
}
