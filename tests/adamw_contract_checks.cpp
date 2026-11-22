// AdamW::step caller-contract audit (base cabaf64).
// Every invalid case is shaped so the CURRENT implementation can only mutate
// state in-bounds — never read/write past a vector. That lets us prove the
// negative controls on the unfixed code without invoking UB.
#include "../src/core/tensor.h"
#include "../src/optimizer/adamw.h"
#include <cstdio>
#include <vector>

struct Snap {
    std::vector<std::vector<f32>> p, m, v;
    i32 t;
};

static Snap snapshot(const AdamW& o, const std::vector<Tensor>& params) {
    Snap s;
    for (const auto& t : params) s.p.push_back(t.data);
    for (const auto& t : o.m) s.m.push_back(t.data);
    for (const auto& t : o.v) s.v.push_back(t.data);
    s.t = o.t;
    return s;
}

static bool same(const Snap& a, const Snap& b) {
    if (a.t != b.t) return false;
    if (a.p.size() != b.p.size() || a.m.size() != b.m.size() || a.v.size() != b.v.size()) return false;
    for (size_t i = 0; i < a.p.size(); ++i) if (a.p[i] != b.p[i]) return false;
    for (size_t i = 0; i < a.m.size(); ++i) if (a.m[i] != b.m[i]) return false;
    for (size_t i = 0; i < a.v.size(); ++i) if (a.v[i] != b.v[i]) return false;
    return true;
}

static bool reject_case(const char* name, std::vector<Tensor> p,
                        std::vector<Tensor> g, std::vector<Tensor> m,
                        std::vector<Tensor> v, i32 t0 = 0) {
    AdamW opt(0.1f, 0.1f);
    opt.m = m; opt.v = v; opt.t = t0;
    const Snap before = snapshot(opt, p);
    std::vector<Tensor*> pp, gg;
    for (auto& t : p) pp.push_back(&t);
    for (auto& t : g) gg.push_back(&t);
    opt.step(pp, gg);
    const bool ok = same(before, snapshot(opt, p));
    std::printf("  %-38s %s (t %d->%d)\n", name, ok ? "OK" : "FAIL", before.t, opt.t);
    return ok;
}

static Tensor filled(i32 r, i32 c, f32 v) { return Tensor(r, c, v); }

int main() {
    bool ok = true;
    auto run = [&](const char* name, std::vector<Tensor> p, std::vector<Tensor> g,
                   std::vector<Tensor> m, std::vector<Tensor> v, i32 t0 = 0) {
        if (!reject_case(name, p, g, m, v, t0)) ok = false;
    };

    std::puts("[vector count mismatches — extras are ignored by old code, never OOB]");
    // grads has an extra tensor: old loop stops at params.size(), so safe.
    run("extra grads tensor", { filled(1, 8, 1.0f) },
        { filled(1, 8, 1.0f), filled(1, 8, 1.0f) },
        { filled(1, 8, 0.0f) }, { filled(1, 8, 0.0f) });
    // Extra moment tensors: old code only touches m[0]/v[0].
    run("extra m tensor", { filled(1, 8, 1.0f) }, { filled(1, 8, 1.0f) },
        { filled(1, 8, 0.0f), filled(1, 8, 0.0f) }, { filled(1, 8, 0.0f) });
    run("extra v tensor", { filled(1, 8, 1.0f) }, { filled(1, 8, 1.0f) },
        { filled(1, 8, 0.0f) }, { filled(1, 8, 0.0f), filled(1, 8, 0.0f) });

    // The short-side cases below are only executable after the fix: on the
    // unfixed code they would index past a vector (UB), so they are regression
    // tests for the hardened implementation rather than negative controls.
    std::puts("[missing tensors — short side]");
    run("missing grads tensor", { filled(1, 8, 1.0f), filled(1, 8, 1.0f) },
        { filled(1, 8, 1.0f) },
        { filled(1, 8, 0.0f), filled(1, 8, 0.0f) },
        { filled(1, 8, 0.0f), filled(1, 8, 0.0f) });
    run("missing m tensor", { filled(1, 8, 1.0f), filled(1, 8, 1.0f) },
        { filled(1, 8, 1.0f), filled(1, 8, 1.0f) },
        { filled(1, 8, 0.0f) },
        { filled(1, 8, 0.0f), filled(1, 8, 0.0f) });
    run("missing v tensor", { filled(1, 8, 1.0f), filled(1, 8, 1.0f) },
        { filled(1, 8, 1.0f), filled(1, 8, 1.0f) },
        { filled(1, 8, 0.0f), filled(1, 8, 0.0f) },
        { filled(1, 8, 0.0f) });

    std::puts("[per-tensor shape mismatches — larger right side, safe on old code]");
    run("grads[0] larger than params[0]", { filled(1, 8, 1.0f) }, { filled(1, 12, 1.0f) },
        { filled(1, 8, 0.0f) }, { filled(1, 8, 0.0f) });
    run("m[0] larger than params[0]", { filled(1, 8, 1.0f) }, { filled(1, 8, 1.0f) },
        { filled(1, 12, 0.0f) }, { filled(1, 8, 0.0f) });
    run("v[0] larger than params[0]", { filled(1, 8, 1.0f) }, { filled(1, 8, 1.0f) },
        { filled(1, 8, 0.0f) }, { filled(1, 12, 0.0f) });
    // Same element count, different shape: memory-safe even on old code.
    run("grads[0] shape 4x2 vs params 2x4", { filled(2, 4, 1.0f) }, { filled(4, 2, 1.0f) },
        { filled(2, 4, 0.0f) }, { filled(2, 4, 0.0f) });

    // Short per-tensor sides: UB before the fix, must be rejected now.
    run("grads[0] smaller than params[0]", { filled(1, 8, 1.0f) }, { filled(1, 4, 1.0f) },
        { filled(1, 8, 0.0f) }, { filled(1, 8, 0.0f) });
    run("m[0] smaller than params[0]", { filled(1, 8, 1.0f) }, { filled(1, 8, 1.0f) },
        { filled(1, 4, 0.0f) }, { filled(1, 8, 0.0f) });
    run("v[0] smaller than params[0]", { filled(1, 8, 1.0f) }, { filled(1, 8, 1.0f) },
        { filled(1, 8, 0.0f) }, { filled(1, 4, 0.0f) });

    std::puts("[invalid element in the middle — no earlier tensor may be touched]");
    run("grads[1] invalid of 3", { filled(1, 8, 1.0f), filled(1, 8, 1.0f), filled(1, 8, 1.0f) },
        { filled(1, 8, 1.0f), filled(1, 12, 1.0f), filled(1, 8, 1.0f) },
        { filled(1, 8, 0.0f), filled(1, 8, 0.0f), filled(1, 8, 0.0f) },
        { filled(1, 8, 0.0f), filled(1, 8, 0.0f), filled(1, 8, 0.0f) });

    // data.size() may be changed independently of rows/cols because all
    // Tensor fields are public. Directions below keep the unfixed kernel
    // in-bounds: params side is SHORT (loop count follows params->size()),
    // the other three sides are LONG (only the first 8 elements are used).
    std::puts("[data.size() desynchronised from rows/cols]");
    {
        auto p = std::vector<Tensor>{ filled(1, 8, 1.0f) };
        p[0].data.resize(4);
        run("params data.size() 4 vs 8", p, { filled(1, 8, 1.0f) },
            { filled(1, 8, 0.0f) }, { filled(1, 8, 0.0f) });
    }
    {
        auto g = std::vector<Tensor>{ filled(1, 8, 1.0f) };
        g[0].data.resize(12);
        run("grads data.size() 12 vs 8", { filled(1, 8, 1.0f) }, g,
            { filled(1, 8, 0.0f) }, { filled(1, 8, 0.0f) });
    }
    {
        auto m = std::vector<Tensor>{ filled(1, 8, 0.0f) };
        m[0].data.resize(12);
        run("m data.size() 12 vs 8", { filled(1, 8, 1.0f) }, { filled(1, 8, 1.0f) },
            m, { filled(1, 8, 0.0f) });
    }
    {
        auto v = std::vector<Tensor>{ filled(1, 8, 0.0f) };
        v[0].data.resize(12);
        run("v data.size() 12 vs 8", { filled(1, 8, 1.0f) }, { filled(1, 8, 1.0f) },
            { filled(1, 8, 0.0f) }, v);
    }

    std::puts("[step counter bounds — storable vs steppable]");
    {
        const i32 imax = std::numeric_limits<i32>::max();
        const bool bounds = adamw_t_valid(0) && adamw_t_valid(imax - 1) &&
                            !adamw_t_valid(-1) && !adamw_t_valid(imax) &&
                            adamw_t_can_step(0) && adamw_t_can_step(imax - 2) &&
                            !adamw_t_can_step(imax - 1) && !adamw_t_can_step(imax);
        std::printf("  %-38s %s\n", "store [0,MAX), step [0,MAX-1)", bounds ? "OK" : "FAIL");
        if (!bounds) ok = false;
        run("t=-1 is a no-op", { filled(1, 8, 1.0f) }, { filled(1, 8, 0.1f) },
            { filled(1, 8, 0.0f) }, { filled(1, 8, 0.0f) }, -1);
        run("t=INT32_MAX-1 is a no-op", { filled(1, 8, 1.0f) }, { filled(1, 8, 0.1f) },
            { filled(1, 8, 0.0f) }, { filled(1, 8, 0.0f) }, imax - 1);
        run("t=INT32_MAX is a no-op", { filled(1, 8, 1.0f) }, { filled(1, 8, 0.1f) },
            { filled(1, 8, 0.0f) }, { filled(1, 8, 0.0f) }, imax);

        // The last permitted step must still leave a state that can be saved.
        std::vector<Tensor> p{ filled(1, 8, 1.0f) }, g{ filled(1, 8, 0.1f) };
        AdamW opt(0.1f, 0.1f);
        opt.add(p[0]);
        opt.t = imax - 2;
        const f32 before_p = p[0].data[0];
        opt.step({ &p[0] }, { &g[0] });
        const bool last_step_ok = p[0].data[0] != before_p && opt.t == imax - 1 &&
                                  adamw_t_valid(opt.t) && !adamw_t_can_step(opt.t);
        std::printf("  %-38s %s (t=%d, storable=%d)\n", "last step leaves storable state",
                    last_step_ok ? "OK" : "FAIL", opt.t, adamw_t_valid(opt.t));
        if (!last_step_ok) ok = false;
    }

    std::puts("[size_t -> i32 guard before step]");
    {
        const size_t max_i32 = static_cast<size_t>(std::numeric_limits<i32>::max());
        const bool guarded = fits_i32(max_i32) && !fits_i32(max_i32 + 1) &&
                             !fits_i32(std::numeric_limits<size_t>::max());
        std::printf("  %-38s %s\n", "oversized element count rejected",
                    guarded ? "OK" : "FAIL");
        if (!guarded) ok = false;
    }

    std::puts("[control: well-formed call must still update]");
    {
        std::vector<Tensor> p{ filled(1, 8, 1.0f) }, g{ filled(1, 8, 1.0f) };
        AdamW opt(0.1f, 0.1f);
        opt.add(p[0]);
        std::vector<Tensor*> pp{ &p[0] }, gg{ &g[0] };
        const f32 before_p = p[0].data[0];
        opt.step(pp, gg);
        const bool updated = p[0].data[0] != before_p && opt.t == 1;
        std::printf("  %-38s %s (p %.6g->%.6g t=%d)\n", "valid pair updates once",
                    updated ? "OK" : "FAIL", before_p, p[0].data[0], opt.t);
        if (!updated) ok = false;
    }

    if (!ok) { std::fprintf(stderr, "adamw_contract_checks: FAILED\n"); return 1; }
    std::puts("adamw_contract_checks: OK");
    return 0;
}
