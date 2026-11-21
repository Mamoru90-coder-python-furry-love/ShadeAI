// AdamW edge-case audit on base da9b8c7.
// Independent double-precision reference for textbook decoupled AdamW:
//   m_t = b1*m_{t-1} + (1-b1)*g
//   v_t = b2*v_{t-1} + (1-b2)*g^2
//   p_{t+1} = p_t*(1-lr*wd) - lr*(m_t/(1-b1^t)) / (sqrt(v_t/(1-b2^t)) + eps)
// Production must match this formula elementwise, including where the formula
// itself becomes non-finite (beta=1, eps=0 with zero gradient). No production
// code is modified by this harness.
#include "../src/core/tensor.h"
#include "../src/optimizer/adamw.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

struct EdgeRef {
    double lr, b1, b2, eps, wd;
    int t = 0;
    std::vector<double> m, v;

    EdgeRef(double lr_, double b1_, double b2_, double eps_, double wd_)
        : lr(lr_), b1(b1_), b2(b2_), eps(eps_), wd(wd_) {}

    void step(std::vector<double>& p, const std::vector<double>& g) {
        ++t;
        const double ib1 = 1.0 / (1.0 - std::pow(b1, t));
        const double ib2 = 1.0 / (1.0 - std::pow(b2, t));
        for (size_t i = 0; i < p.size(); ++i) {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            p[i] = p[i] * (1.0 - lr * wd)
                 - lr * (m[i] * ib1) / (std::sqrt(v[i] * ib2) + eps);
        }
    }
};

static bool near(double a, double b, double rel) {
    return std::fabs(a - b) <= rel * (1.0 + std::fabs(b));
}

enum GradMode { ZERO, TINY, LARGE, MIXED };

static double grad_for(int i, GradMode mode) {
    switch (mode) {
        case ZERO:  return 0.0;
        case TINY:  return ((i % 2) ? -1.0 : 1.0) * 1e-20;
        case LARGE: return ((i % 2) ? -1.0 : 1.0) * 1e6;
        default:    return 0.01 * (((i * 7) % 13) - 6);
    }
}

struct RunResult {
    double worst = 0.0;
    bool prod_nan = false, ref_nan = false;
    int first_divergent_index = -1;
};

// Compare production AdamW against the independent reference for `steps`
// updates. `skip` marks an element intentionally poisoned by a test.
static RunResult compare_run(double lr, double wd, double b1, double b2, double eps,
                             int n, int steps, GradMode mode,
                             const std::vector<int>& skip = {}) {
    std::vector<double> pref(n), gref(n);
    for (int i = 0; i < n; ++i) {
        pref[i] = 0.5 + 0.01 * i;
        gref[i] = grad_for(i, mode);
    }

    Tensor p(1, n), g(1, n);
    for (int i = 0; i < n; ++i) { p.data[i] = (f32)pref[i]; g.data[i] = (f32)gref[i]; }
    AdamW opt((f32)lr, (f32)wd);
    opt.beta1 = (f32)b1; opt.beta2 = (f32)b2; opt.eps = (f32)eps;
    opt.add(p);
    EdgeRef ref(lr, b1, b2, eps, wd);
    ref.m.assign(n, 0.0); ref.v.assign(n, 0.0);
    std::vector<Tensor*> ps{ &p }, gs{ &g };
    for (int s = 0; s < steps; ++s) { opt.step(ps, gs); ref.step(pref, gref); }

    RunResult r;
    for (int i = 0; i < n; ++i) {
        if (std::find(skip.begin(), skip.end(), i) != skip.end()) continue;
        const double got = p.data[i], want = pref[i];
        const bool wf = std::isfinite(want), gf = std::isfinite(got);
        if (!wf) r.ref_nan = true;
        if (!gf) r.prod_nan = true;
        if (!wf || !gf) continue;
        const double d = std::fabs(got - want);
        if (d > r.worst) { r.worst = d; r.first_divergent_index = i; }
    }
    return r;
}

int main() {
    bool ok = true;
    auto report = [&](const char* name, const RunResult& r, double tol) {
        const bool finite_agreement = r.prod_nan == r.ref_nan;
        const bool good = finite_agreement && near(r.worst, 0.0, tol);
        std::printf("  %-28s %s worst=%.6g tol=%.3g prodNaN=%d refNaN=%d i=%d\n",
                    name, good ? "OK" : "FAIL", r.worst, tol,
                    r.prod_nan, r.ref_nan, r.first_divergent_index);
        if (!good) ok = false;
    };
    auto math_case = [&](const char* name, double lr, double wd, double b1,
                         double b2, double eps, int steps, GradMode mode, double tol) {
        report(name, compare_run(lr, wd, b1, b2, eps, 40, steps, mode), tol);
    };

    std::puts("[numerical reference: ordinary ranges]");
    math_case("default_mixed",       1e-3, 0.01, 0.9, 0.999, 1e-8, 20, MIXED, 1e-4);
    math_case("lr_tiny_1e-8",        1e-8, 0.0,  0.9, 0.999, 1e-8, 10, MIXED, 1e-4);
    math_case("lr_large_1",          1.0,  0.0,  0.9, 0.999, 1e-8, 3,  MIXED, 1e-3);
    math_case("wd_large_1",          0.01, 1.0,  0.9, 0.999, 1e-8, 5,  MIXED, 1e-4);
    math_case("wd_negative",         0.01,-0.5,  0.9, 0.999, 1e-8, 5,  MIXED, 1e-4);
    math_case("lr_negative",        -0.01, 0.1,  0.9, 0.999, 1e-8, 5,  MIXED, 1e-4);
    math_case("beta1_zero",          0.01, 0.1,  0.0, 0.999, 1e-8, 5,  MIXED, 1e-4);
    math_case("beta2_zero",          0.01, 0.1,  0.9, 0.0,   1e-8, 5,  MIXED, 1e-4);
    math_case("beta_low",            0.02, 0.1,  0.5, 0.9,   1e-8, 12, MIXED, 1e-4);
    math_case("eps_large_1e-3",      0.02, 0.1,  0.9, 0.999, 1e-3, 10, MIXED, 1e-4);
    math_case("lr_wd_product_gt_1",  0.5,  3.0,  0.9, 0.999, 1e-8, 3,  MIXED, 1e-3);

    std::puts("[gradient magnitudes]");
    math_case("zero_gradient",       0.01, 0.1,  0.9, 0.999, 1e-8, 6,  ZERO,  1e-5);
    math_case("tiny_gradient_1e-20", 0.01, 0.1,  0.9, 0.999, 1e-8, 6,  TINY,  1e-4);
    math_case("large_gradient_1e6",  0.01, 0.1,  0.9, 0.999, 1e-8, 6,  LARGE, 1e-4);

    std::puts("[long run / AVX + scalar tail]");
    math_case("long_run_2000",       1e-3, 0.01, 0.9, 0.999, 1e-8, 2000, MIXED, 5e-3);
    // n=37 => 32 elements through AVX, 5 through the scalar tail in one step.
    report("avx_plus_scalar_tail",
           compare_run(0.02, 0.15, 0.85, 0.99, 1e-8, 37, 20, MIXED), 1e-4);

    std::puts("[formula breakdown: both must agree it is non-finite]");
    math_case("beta1_eq_1",          0.01, 0.0,  1.0, 0.999, 1e-8, 1, MIXED, 1e-4);
    math_case("beta2_eq_1",          0.01, 0.0,  0.9, 1.0,   1e-8, 1, MIXED, 1e-4);
    math_case("eps_zero_zero_grad",  0.01, 0.0,  0.9, 0.999, 0.0,  1, ZERO,  1e-4);

    std::puts("[NaN / Inf locality]");
    {
        const int n = 24;
        Tensor p(1, n), g(1, n);
        std::vector<double> pref(n), gref(n);
        for (int i = 0; i < n; ++i) {
            pref[i] = 1.0 + 0.01 * i;
            p.data[i] = (f32)pref[i];
            gref[i] = 0.01 * (i + 1);
            g.data[i] = (i == 5) ? std::numeric_limits<f32>::quiet_NaN() : (f32)gref[i];
        }
        gref[5] = g.data[5]; // poison the reference at exactly one element
        AdamW opt(0.01f, 0.1f);
        opt.add(p);
        EdgeRef ref(0.01, 0.9, 0.999, 1e-8, 0.1);
        ref.m.assign(n, 0.0); ref.v.assign(n, 0.0);
        std::vector<Tensor*> ps{ &p }, gs{ &g };
        opt.step(ps, gs); ref.step(pref, gref);
        double worst = 0.0;
        for (int i = 0; i < n; ++i) {
            if (i == 5) continue;
            worst = std::max(worst, std::fabs((double)p.data[i] - pref[i]));
        }
        report("nan_gradient_is_local", { worst, false, false, -1 }, 1e-4);
        if (!std::isnan(p.data[5])) { std::fprintf(stderr, "FAIL: nan gradient did not reach its own weight\n"); ok = false; }
    }
    {
        const int n = 24;
        Tensor p(1, n), g(1, n);
        std::vector<double> pref(n), gref(n);
        for (int i = 0; i < n; ++i) {
            pref[i] = 1.0 + 0.01 * i;
            p.data[i] = (f32)pref[i];
            gref[i] = 0.01 * (i + 1);
            g.data[i] = (i == 11) ? std::numeric_limits<f32>::infinity() : (f32)gref[i];
        }
        gref[11] = g.data[11];
        AdamW opt(0.01f, 0.1f);
        opt.add(p);
        EdgeRef ref(0.01, 0.9, 0.999, 1e-8, 0.1);
        ref.m.assign(n, 0.0); ref.v.assign(n, 0.0);
        std::vector<Tensor*> ps{ &p }, gs{ &g };
        opt.step(ps, gs); ref.step(pref, gref);
        double worst = 0.0;
        for (int i = 0; i < n; ++i) {
            if (i == 11 || !std::isfinite(pref[i])) continue;
            worst = std::max(worst, std::fabs((double)p.data[i] - pref[i]));
        }
        report("inf_gradient_is_local", { worst, false, false, -1 }, 1e-4);
    }
    {
        // Pre-poisoned moment at one element must not spread to neighbours.
        const int n = 24;
        Tensor p(1, n), g(1, n);
        std::vector<double> pref(n), gref(n);
        for (int i = 0; i < n; ++i) {
            pref[i] = 1.0 + 0.01 * i;
            p.data[i] = (f32)pref[i];
            gref[i] = g.data[i] = 0.02f * (i + 1);
        }
        AdamW opt(0.01f, 0.1f);
        opt.add(p);
        opt.m[0].data[7] = std::numeric_limits<f32>::quiet_NaN();
        EdgeRef ref(0.01, 0.9, 0.999, 1e-8, 0.1);
        ref.m.assign(n, 0.0); ref.v.assign(n, 0.0);
        ref.m[7] = std::numeric_limits<double>::quiet_NaN();
        std::vector<Tensor*> ps{ &p }, gs{ &g };
        opt.step(ps, gs); ref.step(pref, gref);
        double worst = 0.0;
        for (int i = 0; i < n; ++i) {
            if (i == 7 || !std::isfinite(pref[i])) continue;
            worst = std::max(worst, std::fabs((double)p.data[i] - pref[i]));
        }
        report("nan_moment_is_local", { worst, false, false, -1 }, 1e-4);
    }

    {
        // Pre-poisoned parameter must not spread to its neighbours.
        const int n = 24;
        Tensor p(1, n), g(1, n);
        std::vector<double> pref(n), gref(n);
        for (int i = 0; i < n; ++i) {
            pref[i] = 1.0 + 0.01 * i;
            p.data[i] = (f32)pref[i];
            gref[i] = g.data[i] = 0.02f * (i + 1);
        }
        p.data[3] = std::numeric_limits<f32>::quiet_NaN();
        pref[3] = std::numeric_limits<double>::quiet_NaN();
        AdamW opt(0.01f, 0.1f);
        opt.add(p);
        EdgeRef ref(0.01, 0.9, 0.999, 1e-8, 0.1);
        ref.m.assign(n, 0.0); ref.v.assign(n, 0.0);
        std::vector<Tensor*> ps{ &p }, gs{ &g };
        opt.step(ps, gs); ref.step(pref, gref);
        double worst = 0.0;
        for (int i = 0; i < n; ++i) {
            if (i == 3 || !std::isfinite(pref[i])) continue;
            worst = std::max(worst, std::fabs((double)p.data[i] - pref[i]));
        }
        report("nan_param_is_local", { worst, false, false, -1 }, 1e-4);
        if (!std::isnan(p.data[3])) { std::fprintf(stderr, "FAIL: poisoned parameter stayed finite\n"); ok = false; }
    }

    std::puts("[clip -> decay interaction]");
    {
        const int n = 37;
        std::vector<double> graw(n);
        for (int i = 0; i < n; ++i) graw[i] = 0.5 * (((i * 3) % 11) - 5);
        double norm = 0.0;
        for (double v : graw) norm += v * v;
        norm = std::sqrt(norm);
        const double scale = (norm > 1.0) ? 1.0 / norm : 1.0;
        Tensor p(1, n), g(1, n);
        std::vector<double> pref(n), gclipped(n);
        for (int i = 0; i < n; ++i) {
            pref[i] = 0.7 + 0.01 * i;
            gclipped[i] = graw[i] * scale;
            p.data[i] = (f32)pref[i];
            g.data[i] = (f32)gclipped[i];
        }
        AdamW opt(0.05f, 0.2f);
        opt.add(p);
        EdgeRef ref(0.05, 0.9, 0.999, 1e-8, 0.2);
        ref.m.assign(n, 0.0); ref.v.assign(n, 0.0);
        std::vector<Tensor*> ps{ &p }, gs{ &g };
        for (int s = 0; s < 5; ++s) { opt.step(ps, gs); ref.step(pref, gclipped); }
        double worst = 0.0;
        for (int i = 0; i < n; ++i) worst = std::max(worst, std::fabs((double)p.data[i] - pref[i]));
        report("clip_then_decay", { worst, false, false, -1 }, 1e-4);
    }

    if (!ok) { std::fprintf(stderr, "adamw_edge_checks: FAILED\n"); return 1; }
    std::puts("adamw_edge_checks: OK");
    return 0;
}
