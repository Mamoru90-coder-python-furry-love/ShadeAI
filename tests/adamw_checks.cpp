// AdamW audit: independent double-precision reference vs the production kernel.
// The reference implements textbook decoupled weight decay:
//   p_{t+1} = p_t * (1 - lr*wd) - lr * m_hat / (sqrt(v_hat) + eps)
// Weight decay must apply to the OLD parameter, not to the post-gradient value.
#include "../src/core/tensor.h"
#include "../src/optimizer/adamw.h"
#include <cmath>
#include <cstdio>
#include <vector>

struct AdamWRef {
    double lr, b1, b2, eps, wd;
    int t = 0;
    std::vector<double> m, v;

    AdamWRef(double lr_, double b1_, double b2_, double eps_, double wd_)
        : lr(lr_), b1(b1_), b2(b2_), eps(eps_), wd(wd_) {}

    void step(std::vector<double>& p, const std::vector<double>& g) {
        ++t;
        const double ib1 = 1.0 / (1.0 - std::pow(b1, t));
        const double ib2 = 1.0 / (1.0 - std::pow(b2, t));
        for (size_t i = 0; i < p.size(); ++i) {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            const double mhat = m[i] * ib1;
            const double vhat = v[i] * ib2;
            // DECOUPLED: decay uses p BEFORE the adaptive step.
            p[i] = p[i] * (1.0 - lr * wd) - lr * mhat / (std::sqrt(vhat) + eps);
        }
    }
};

static bool close(double a, double b, double rel) {
    return std::fabs(a - b) <= rel * (1.0 + std::fabs(b));
}

static int check(const char* name, double lr, double wd, int n, int steps,
                 double rel, bool vary_g) {
    std::vector<double> pref(n), gref(n);
    for (int i = 0; i < n; ++i) {
        pref[i] = 1.0 + 0.01 * i;
        gref[i] = vary_g ? ((i % 3 == 0) ? -0.7 : 0.3 + 0.05 * (i % 5))
                         : 0.0;
    }

    Tensor p(1, n), g(1, n);
    for (int i = 0; i < n; ++i) { p.data[i] = (f32)pref[i]; g.data[i] = (f32)gref[i]; }

    AdamW opt(lr, wd);
    opt.add(p);
    AdamWRef ref(lr, 0.9, 0.999, 1e-8, wd);
    ref.m.assign(n, 0.0); ref.v.assign(n, 0.0);
    std::vector<Tensor*> ps{ &p }, gs{ &g };
    for (int s = 0; s < steps; ++s) {
        opt.step(ps, gs);
        ref.step(pref, gref);
    }

    double worst = 0.0;
    int worst_i = -1;
    for (int i = 0; i < n; ++i) {
        const double d = std::fabs((double)p.data[i] - pref[i]);
        if (d > worst) { worst = d; worst_i = i; }
    }
    const bool ok = close(worst, 0.0, rel);
    std::printf("%s: %s (worst |got-ref|=%.6g at i=%d, rel tol=%.3g)\n",
                name, ok ? "OK" : "FAIL", worst, worst_i, rel);
    return ok ? 0 : 1;
}

int main() {
    int rc = 0;
    // Analytic one-step case: at t=1, m_hat=g and v_hat=g^2, so the adaptive
    // step is ±lr. With lr=0.5/wd=0.2 the ordering bug shifts p by 0.05.
    rc |= check("adamw_analytic_1step", 0.5, 0.2, 40, 1, 1e-5, true);
    // Multi-step: accumulates the ordering error over 12 updates.
    rc |= check("adamw_multistep", 0.05, 0.3, 40, 12, 1e-4, true);
    // Regression controls: these MUST pass on both old and new code.
    rc |= check("adamw_zero_grad_decay", 0.1, 0.2, 40, 3, 1e-5, false);
    rc |= check("adamw_no_decay", 0.01, 0.0, 40, 6, 1e-5, true);
    if (rc) {
        std::fprintf(stderr, "adamw_checks: FAILED\n");
        return 1;
    }
    std::puts("adamw_checks: OK");
    return 0;
}
