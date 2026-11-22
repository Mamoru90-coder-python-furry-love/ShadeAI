// Optimizer checkpoint audit: training continuity across save/reload.
// Compares 10 uninterrupted AdamW steps against 4 steps + checkpoint + 6 steps.
// Also covers the negative control (model-only resume) and corrupt/incompatible
// optimizer sidecars. Built only for tests; no production code is touched here.
#include "../src/utils/serialize.h"
#include "../src/optimizer/adamw.h"
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static std::vector<Tensor*> ptrs(std::vector<Tensor>& v) {
    std::vector<Tensor*> r; r.reserve(v.size());
    for (auto& t : v) r.push_back(&t);
    return r;
}

static void clean(const fs::path& base_model, const fs::path& base_opt) {
    fs::remove(base_model); fs::remove(base_opt);
    fs::remove(base_model.string() + ".manifest");
    fs::remove(base_model.string() + ".tmp");
    fs::remove(base_opt.string() + ".tmp");
    for (int g = 1; g <= 8; g++) {
        const std::string s = std::to_string(g);
        fs::remove(base_model.string() + ".g" + s);
        fs::remove(base_opt.string() + ".g" + s);
        fs::remove(base_model.string() + ".g" + s + ".tmp");
        fs::remove(base_opt.string() + ".g" + s + ".tmp");
    }
}

static void init_params(std::vector<Tensor>& p) {
    for (size_t i = 0; i < p.size(); i++)
        for (i32 j = 0; j < p[i].size(); j++)
            p[i].data[j] = 0.1f * (f32)(j + 1) - 0.5f + 0.01f * (f32)i;
}

static void set_grads(std::vector<Tensor>& g, int step) {
    for (size_t i = 0; i < g.size(); i++)
        for (i32 j = 0; j < g[i].size(); j++)
            g[i].data[j] = 0.1f * (f32)(((step * 7 + (int)i * 3 + j) % 11) - 5);
}

static void step_all(std::vector<Tensor>& p, AdamW& opt, std::vector<Tensor>& g, int step) {
    set_grads(g, step);
    auto pp = ptrs(p), gg = ptrs(g);
    opt.step(pp, gg);
}

static std::vector<Tensor> make_params() {
    std::vector<Tensor> p;
    p.emplace_back(2, 9); // 18 elems: exercises AVX blocks + 2-elem tail
    p.emplace_back(1, 4);
    p.emplace_back(4, 1);
    init_params(p);
    return p;
}

static bool same_params(const std::vector<Tensor>& a, const std::vector<Tensor>& b, double* worst) {
    if (a.size() != b.size()) { *worst = 1e30; return false; }
    double w = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].size() != b[i].size()) { *worst = 1e30; return false; }
        for (i32 j = 0; j < a[i].size(); j++)
            w = std::max(w, (double)std::fabs(a[i].data[j] - b[i].data[j]));
    }
    *worst = w;
    return w <= 1e-6;
}

static std::vector<Tensor> snap4_params, snap4_m, snap4_v;
static i32 snap4_t = 0;

static void poke_byte(const fs::path& p, size_t off, unsigned char val) {
    std::FILE* f = fopen(p.string().c_str(), "r+b");
    if (!f) return;
    fseek(f, (long)off, SEEK_SET);
    fputc(val, f);
    fclose(f);
}

static void poke_i32(const fs::path& p, size_t off, i32 val) {
    std::FILE* f = fopen(p.string().c_str(), "r+b");
    if (!f) return;
    fseek(f, (long)off, SEEK_SET);
    fwrite(&val, sizeof(val), 1, f);
    fclose(f);
}

int main() {
    const fs::path dir = fs::temp_directory_path();
    const fs::path m_cont = dir / "shade_resume_continuous.bin";
    const fs::path m_r = dir / "shade_resume_r.bin";
    const fs::path o_r = dir / "shade_resume_o_r.bin";
    const fs::path m_only = dir / "shade_resume_only.bin";
    const fs::path o_only = dir / "shade_resume_o_only.bin";
    const fs::path m_legacy = dir / "shade_resume_legacy.bin";
    const fs::path o_legacy = dir / "shade_resume_o_legacy.bin";
    const fs::path m_missing = dir / "shade_resume_missing.bin";
    const fs::path o_missing = dir / "shade_resume_o_missing.bin";
    const fs::path m_badshape = dir / "shade_resume_badshape.bin";
    const fs::path o_badshape = dir / "shade_resume_o_badshape.bin";

    for (auto pr : { std::make_pair(m_cont, o_r), std::make_pair(m_r, o_r),
                     std::make_pair(m_only, o_only), std::make_pair(m_legacy, o_legacy),
                     std::make_pair(m_missing, o_missing), std::make_pair(m_badshape, o_badshape) })
        clean(pr.first, pr.second);
    fs::remove(m_cont);

    bool ok = true;
    auto expect = [&](bool cond, const char* what) {
        if (!cond) { std::fprintf(stderr, "FAIL: %s\n", what); ok = false; }
    };

    // Ground truth: 10 uninterrupted steps.
    auto pref = make_params();
    std::vector<Tensor> gref;
    for (auto& p : pref) gref.emplace_back(p.rows, p.cols);
    AdamW oref(0.05f, 0.1f);
    for (auto* p : ptrs(pref)) oref.add(*p);
    for (int s = 1; s <= 10; s++) {
        step_all(pref, oref, gref, s);
        if (s == 4) {
            // Snapshot the exact state a checkpoint at step 4 must preserve.
            snap4_params = pref; snap4_m = oref.m; snap4_v = oref.v; snap4_t = oref.t;
        }
    }

    // 1. Checkpoint/resume at step 4 must reproduce the uninterrupted run.
    {
        auto p = make_params();
        std::vector<Tensor> g;
        for (auto& t : p) g.emplace_back(t.rows, t.cols);
        AdamW opt(0.05f, 0.1f);
        for (auto* q : ptrs(p)) opt.add(*q);
        for (int s = 1; s <= 4; s++) step_all(p, opt, g, s);
        expect(Serialize::CheckpointPair::publish(m_r.string(), o_r.string(), ptrs(p),
                                                  4, 2, 16, ptrs(opt.m), ptrs(opt.v), opt.t),
               "publish model+optimizer pair");

        auto p2 = make_params();
        std::vector<Tensor> g2;
        for (auto& t : p2) g2.emplace_back(t.rows, t.cols);
        AdamW opt2(0.05f, 0.1f);
        for (auto* q : ptrs(p2)) opt2.add(*q);
        auto res = Serialize::CheckpointPair::load(m_r.string(), o_r.string(),
                                                   ptrs(p2), ptrs(opt2.m), ptrs(opt2.v));
        expect(res.valid && res.has_opt && res.t == 4, "resume reads t/m/v");
        opt2.t = res.t; // Result::t must be propagated into AdamW explicitly
        double wp = 0.0, wm = 0.0, wv = 0.0;
        same_params(p2, snap4_params, &wp);
        same_params(opt2.m, snap4_m, &wm);
        same_params(opt2.v, snap4_v, &wv);
        std::printf("  after load: |dp|=%.6g |dm|=%.6g |dv|=%.6g t=%d (want %d)\n",
                    wp, wm, wv, res.t, snap4_t);
        for (int s = 5; s <= 10; s++) step_all(p2, opt2, g2, s);
        double worst = 0.0;
        expect(same_params(p2, pref, &worst), "resumed training matches continuous run");
        std::printf("  continuity worst |diff|=%.6g\n", worst);
    }

    // 2. NEGATIVE CONTROL: resume from the model only (no moments, t reset).
    //    This MUST diverge, otherwise the continuity test would be meaningless.
    {
        auto p = make_params();
        std::vector<Tensor> g;
        for (auto& t : p) g.emplace_back(t.rows, t.cols);
        AdamW opt(0.05f, 0.1f);
        for (auto* q : ptrs(p)) opt.add(*q);
        for (int s = 1; s <= 4; s++) step_all(p, opt, g, s);
        const std::vector<Tensor*> none;
        expect(Serialize::CheckpointPair::publish(m_only.string(), o_only.string(), ptrs(p),
                                                  4, 2, 16, none, none, 0),
               "publish model-only pair");

        auto p2 = make_params();
        std::vector<Tensor> g2;
        for (auto& t : p2) g2.emplace_back(t.rows, t.cols);
        AdamW opt2(0.05f, 0.1f);
        for (auto* q : ptrs(p2)) opt2.add(*q);
        auto res = Serialize::CheckpointPair::load(m_only.string(), o_only.string(),
                                                   ptrs(p2), ptrs(opt2.m), ptrs(opt2.v));
        expect(res.valid && !res.has_opt, "model-only pair loads without optimizer");
        opt2.t = res.t;
        for (int s = 5; s <= 10; s++) step_all(p2, opt2, g2, s);
        double worst = 0.0;
        expect(!same_params(p2, pref, &worst), "dropping moments diverges (negative control)");
        std::printf("  no-moments worst |diff|=%.6g\n", worst);
    }

    // 3. Legacy model + optimizer sidecar with a CORRUPT sidecar must be a hard
    //    failure, not a silent downgrade to "model-only" (which resumes with
    //    zero moments and the wrong step counter).
    {
        auto p = make_params();
        std::vector<Tensor> g;
        for (auto& t : p) g.emplace_back(t.rows, t.cols);
        AdamW opt(0.05f, 0.1f);
        for (auto* q : ptrs(p)) opt.add(*q);
        for (int s = 1; s <= 4; s++) step_all(p, opt, g, s);
        expect(Serialize::save_model(m_legacy.string(), ptrs(p), 4, 2, 16), "legacy model save");
        expect(Serialize::save_optimizer(o_legacy.string(), ptrs(opt.m), ptrs(opt.v), opt.t),
               "legacy optimizer save");

        const size_t sz = fs::file_size(o_legacy);
        // sz/2 lands in the header of the first v tensor for these shapes:
        // this proves structural corruption is rejected, not payload corruption.
        poke_byte(o_legacy, sz / 2, 0x42);

        auto p2 = make_params();
        std::vector<Tensor> g2;
        for (auto& t : p2) g2.emplace_back(t.rows, t.cols);
        AdamW opt2(0.05f, 0.1f);
        for (auto* q : ptrs(p2)) opt2.add(*q);
        const f32 before = p2[0].data[0];
        auto res = Serialize::CheckpointPair::load(m_legacy.string(), o_legacy.string(),
                                                   ptrs(p2), ptrs(opt2.m), ptrs(opt2.v));
        expect(!res.valid, "corrupt legacy optimizer fails the whole load");
        expect(p2[0].data[0] == before && opt2.t == 0, "corrupt legacy optimizer commits nothing");
    }

    // 4. Regression: a legacy model with NO sidecar is a legitimate model-only
    //    checkpoint and must still load.
    {
        auto p = make_params();
        expect(Serialize::save_model(m_missing.string(), ptrs(p), 4, 2, 16), "legacy model-only save");
        auto p2 = make_params();
        std::vector<Tensor> g2;
        AdamW opt2(0.05f, 0.1f);
        for (auto* q : ptrs(p2)) opt2.add(*q);
        auto res = Serialize::CheckpointPair::load(m_missing.string(), o_missing.string(),
                                                   ptrs(p2), ptrs(opt2.m), ptrs(opt2.v));
        expect(res.valid && !res.has_opt && res.t == 0, "missing legacy sidecar = model-only");
        double worst = 0.0;
        expect(same_params(p2, p, &worst), "missing sidecar restores model");
    }

    // 5. Regression: an incompatible sidecar (different shapes) is rejected and
    //    commits neither weights nor moments.
    {
        auto p = make_params();
        std::vector<Tensor> wrong_m, wrong_v;
        wrong_m.emplace_back(1, 2); wrong_v.emplace_back(1, 2);
        for (int i = 1; i < (int)p.size(); i++) {
            wrong_m.emplace_back(p[i].rows, p[i].cols);
            wrong_v.emplace_back(p[i].rows, p[i].cols);
        }
        expect(Serialize::save_model(m_badshape.string(), ptrs(p), 4, 2, 16), "badshape model save");
        expect(Serialize::save_optimizer(o_badshape.string(), ptrs(wrong_m), ptrs(wrong_v), 9),
               "badshape optimizer save");
        auto p2 = make_params();
        std::vector<Tensor> g2;
        AdamW opt2(0.05f, 0.1f);
        for (auto* q : ptrs(p2)) opt2.add(*q);
        const f32 before = p2[0].data[0];
        auto res = Serialize::CheckpointPair::load(m_badshape.string(), o_badshape.string(),
                                                   ptrs(p2), ptrs(opt2.m), ptrs(opt2.v));
        expect(!res.valid, "incompatible optimizer shape rejected");
        expect(p2[0].data[0] == before && opt2.t == 0, "incompatible sidecar commits nothing");
    }

    // 6. P3-A: v without m is a caller bug on LOAD too (publish already rejects
    //    it). The size check must sit at the common level of load(), before
    //    the legacy/generation split — a model-only checkpoint must not hide it.
    {
        std::vector<Tensor> lone_v;
        lone_v.emplace_back(1, 2);
        const std::vector<Tensor*> no_m;

        // 6a. legacy model-only checkpoint.
        {
            auto p = make_params();
            const fs::path mp = dir / "shade_resume_vonly_model.bin";
            const fs::path op = dir / "shade_resume_vonly_opt.bin";
            clean(mp, op);
            expect(Serialize::save_model(mp.string(), ptrs(p), 4, 2, 16), "v-only legacy model save");
            auto p2 = make_params();
            auto res = Serialize::CheckpointPair::load(mp.string(), op.string(),
                                                       ptrs(p2), no_m, ptrs(lone_v));
            expect(!res.valid, "v without m rejected on legacy load");
            clean(mp, op);
        }

        // 6b. generation model-only checkpoint (manifest exists, FLAG_HAS_OPT=0).
        //     This is the case the previous fix missed entirely.
        {
            auto p = make_params();
            const fs::path mp = dir / "shade_resume_vonly_gen_model.bin";
            const fs::path op = dir / "shade_resume_vonly_gen_opt.bin";
            clean(mp, op);
            const std::vector<Tensor*> none;
            expect(Serialize::CheckpointPair::publish(mp.string(), op.string(), ptrs(p),
                                                      4, 2, 16, none, none, 0),
                   "publish generation model-only");
            auto p2 = make_params();
            auto res = Serialize::CheckpointPair::load(mp.string(), op.string(),
                                                       ptrs(p2), no_m, ptrs(lone_v));
            expect(!res.valid, "v without m rejected on generation model-only load");
            clean(mp, op);
        }
    }

    // 7. An unreadable legacy sidecar (directory at that path) is not "absent".
    {
        auto p = make_params();
        expect(Serialize::save_model(m_missing.string(), ptrs(p), 4, 2, 16), "unreadable opt model save");
        fs::remove(o_missing);
        fs::create_directory(o_missing);
        auto p2 = make_params();
        AdamW opt2(0.05f, 0.1f);
        for (auto* q : ptrs(p2)) opt2.add(*q);
        auto res = Serialize::CheckpointPair::load(m_missing.string(), o_missing.string(),
                                                   ptrs(p2), ptrs(opt2.m), ptrs(opt2.v));
        expect(!res.valid, "unreadable legacy optimizer rejected");
        fs::remove(o_missing);
    }

    // 8. The step counter is i32 and step() increments it before use. A
    //    checkpoint at INT32_MAX must be rejected before it can wrap to a
    //    negative t and silently poison the next resume.
    {
        const i32 imax = std::numeric_limits<i32>::max();
        const fs::path mp = dir / "shade_resume_tmax_model.bin";
        const fs::path op = dir / "shade_resume_tmax_opt.bin";
        clean(mp, op);

        // 8a. publish() refuses an unresumable counter.
        {
            auto p = make_params();
            std::vector<Tensor> g;
            for (auto& t : p) g.emplace_back(t.rows, t.cols);
            AdamW opt(0.05f, 0.1f);
            for (auto* q : ptrs(p)) opt.add(*q);
            expect(!Serialize::CheckpointPair::publish(mp.string(), op.string(), ptrs(p),
                                                       4, 2, 16, ptrs(opt.m), ptrs(opt.v), imax),
                   "publish rejects t=INT32_MAX");
        }

        // 8b. A legacy sidecar whose trailing t was poked to INT32_MAX is a hard
        //     failure; no weight or moment may be committed.
        {
            auto p = make_params();
            std::vector<Tensor> g;
            for (auto& t : p) g.emplace_back(t.rows, t.cols);
            AdamW opt(0.05f, 0.1f);
            for (auto* q : ptrs(p)) opt.add(*q);
            for (int s = 1; s <= 4; s++) step_all(p, opt, g, s);
            expect(Serialize::save_model(mp.string(), ptrs(p), 4, 2, 16), "tmax model save");
            expect(Serialize::save_optimizer(op.string(), ptrs(opt.m), ptrs(opt.v), opt.t),
                   "tmax optimizer save");
            poke_i32(op, fs::file_size(op) - sizeof(imax), imax);

            auto p2 = make_params();
            std::vector<Tensor> g2;
            AdamW opt2(0.05f, 0.1f);
            for (auto* q : ptrs(p2)) opt2.add(*q);
            const f32 before = p2[0].data[0];
            auto res = Serialize::CheckpointPair::load(mp.string(), op.string(),
                                                       ptrs(p2), ptrs(opt2.m), ptrs(opt2.v));
            expect(!res.valid, "legacy sidecar t=INT32_MAX rejected");
            expect(p2[0].data[0] == before && opt2.t == 0,
                   "t=INT32_MAX sidecar commits nothing");
        }

        // 8c. Negative control for the documented caller contract: correct m/v
        //     but forgetting `opt.t = res.t` must still diverge.
        {
            auto p = make_params();
            std::vector<Tensor> g;
            for (auto& t : p) g.emplace_back(t.rows, t.cols);
            AdamW opt(0.05f, 0.1f);
            for (auto* q : ptrs(p)) opt.add(*q);
            auto res = Serialize::CheckpointPair::load(m_r.string(), o_r.string(),
                                                       ptrs(p), ptrs(opt.m), ptrs(opt.v));
            expect(res.valid && res.has_opt && res.t == 4, "control pair still readable");
            // DELIBERATELY omit opt.t = res.t.
            for (int s = 5; s <= 10; s++) step_all(p, opt, g, s);
            double worst = 0.0;
            expect(!same_params(p, pref, &worst), "forgetting res.t diverges (negative control)");
            std::printf("  forgotten t worst |diff|=%.6g\n", worst);
        }

        // 8d. A terminal counter is storable but not resumable. publish() and
        //     the raw sidecar must still round-trip it; the training-oriented
        //     CheckpointPair::load must refuse it before committing anything.
        {
            clean(mp, op);
            auto p = make_params();
            std::vector<Tensor> g;
            for (auto& t : p) g.emplace_back(t.rows, t.cols);
            AdamW opt(0.05f, 0.1f);
            for (auto* q : ptrs(p)) opt.add(*q);
            expect(Serialize::CheckpointPair::publish(mp.string(), op.string(), ptrs(p),
                                                       4, 2, 16, ptrs(opt.m), ptrs(opt.v), imax - 1),
                   "publish stores terminal t=INT32_MAX-1");

            auto p2 = make_params();
            AdamW opt2(0.05f, 0.1f);
            for (auto* q : ptrs(p2)) opt2.add(*q);
            const f32 before = p2[0].data[0];
            auto res = Serialize::CheckpointPair::load(mp.string(), op.string(),
                                                       ptrs(p2), ptrs(opt2.m), ptrs(opt2.v));
            expect(!res.valid, "CheckpointPair::load rejects terminal t");
            expect(p2[0].data[0] == before && opt2.t == 0,
                   "terminal checkpoint commits nothing");

            expect(Serialize::save_optimizer(op.string(), ptrs(opt.m), ptrs(opt.v), imax - 1),
                   "raw sidecar stores terminal t");
            std::vector<Tensor> rm, rv;
            for (auto& q : opt.m) rm.emplace_back(q.rows, q.cols);
            for (auto& q : opt.v) rv.emplace_back(q.rows, q.cols);
            i32 rt = -1;
            expect(Serialize::load_optimizer(op.string(), ptrs(rm), ptrs(rv), rt) &&
                   rt == imax - 1, "raw sidecar reads terminal t");
        }
        clean(mp, op);
    }

    for (auto pr : { std::make_pair(m_cont, o_r), std::make_pair(m_r, o_r),
                     std::make_pair(m_only, o_only), std::make_pair(m_legacy, o_legacy),
                     std::make_pair(m_missing, o_missing), std::make_pair(m_badshape, o_badshape) })
        clean(pr.first, pr.second);

    if (!ok) { std::fprintf(stderr, "optimizer_resume_checks: FAILED\n"); return 1; }
    std::puts("optimizer_resume_checks: OK");
    return 0;
}
