// Checkpoint integrity: truncated / poisoned / trailing-garbage files must fail
// without modifying any live tensor or the AdamW step counter.
#include "../src/utils/serialize.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool same(const Tensor& a, const Tensor& b) {
    return a.rows == b.rows && a.cols == b.cols && a.data == b.data;
}
static void fill(Tensor& t, f32 v) { std::fill(t.data.begin(), t.data.end(), v); }

static std::vector<char> read_all(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return { std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>() };
}
static bool write_all(const fs::path& p, const std::vector<char>& bytes) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(bytes.data(), (std::streamsize)bytes.size());
    return f.good();
}
static bool poke_nan(const fs::path& p, size_t offset) {
    auto bytes = read_all(p);
    if (bytes.size() < offset + 4) return false;
    const uint32_t bits = 0x7fc00000u; // quiet NaN
    std::memcpy(bytes.data() + offset, &bits, 4);
    return write_all(p, bytes);
}
static bool truncate_by(const fs::path& src, const fs::path& dst, size_t bytes) {
    auto data = read_all(src);
    if (data.size() <= bytes) return false;
    data.resize(data.size() - bytes);
    return write_all(dst, data);
}
static bool append_garbage(const fs::path& src, const fs::path& dst) {
    auto data = read_all(src);
    data.push_back(0x42);
    return write_all(dst, data);
}

static void remove_pair_files(const fs::path& model, const fs::path& opt) {
    fs::remove(model.string() + ".manifest");
    fs::remove(model.string() + ".manifest.tmp");
    for (i32 g = 1; g <= 8; g++) {
        const std::string n = std::to_string(g);
        fs::remove(model.string() + ".g" + n);
        fs::remove(opt.string() + ".g" + n);
        fs::remove(model.string() + ".g" + n + ".tmp");
        fs::remove(opt.string() + ".g" + n + ".tmp");
    }
}

struct PairLoad {
    bool valid = false;
    Serialize::ModelInfo info{ 0, 0, 0, false };
    bool has_opt = false;
    i32 t = 0;
    f32 model0 = -7.0f, m0 = -7.0f, v0 = -7.0f;
};

static PairLoad read_pair(const fs::path& pm, const fs::path& po) {
    PairLoad r;
    Tensor a(2, 3), m(2, 3), v(2, 3);
    fill(a, -7.0f); fill(m, -7.0f); fill(v, -7.0f);
    auto x = Serialize::CheckpointPair::load(pm.string(), po.string(),
                                             { &a }, { &m }, { &v });
    r.valid = x.valid; r.info = x.info; r.has_opt = x.has_opt; r.t = x.t;
    r.model0 = a.data[0]; r.m0 = m.data[0]; r.v0 = v.data[0];
    return r;
}

// Atomic publishing: a failed/interrupted write must never touch the previous
// good checkpoint, and AsyncSaver must survive back-to-back spawns.
static int atomic_and_async_checks() {
    const fs::path dir = fs::temp_directory_path();
    const fs::path model = dir / "shade_atomic_model.bin";
    const fs::path opt = dir / "shade_atomic_opt.bin";
    const fs::path amodel = dir / "shade_async_model.bin";
    const fs::path aopt = dir / "shade_async_opt.bin";
    const fs::path model_tmp(model.string() + ".tmp"), opt_tmp(opt.string() + ".tmp");
    const fs::path amodel_tmp(amodel.string() + ".tmp"), aopt_tmp(aopt.string() + ".tmp");
    for (const auto& p : { model, opt, amodel, aopt, model_tmp, opt_tmp, amodel_tmp, aopt_tmp })
        fs::remove(p);

    bool ok = true;
    auto expect = [&](bool cond, const char* what) {
        if (!cond) { std::fprintf(stderr, "FAIL: %s\n", what); ok = false; }
    };

    Tensor a(2, 3);
    for (i32 i = 0; i < a.size(); i++) a.data[i] = (f32)(i + 1);
    std::vector<Tensor*> params{ &a };

    // Successful save publishes through a temp file and leaves none behind.
    expect(Serialize::save_model(model.string(), params, 4, 2, 16), "atomic save_model");
    expect(!fs::exists(model_tmp), "no temp after successful save");
    const auto good_model = read_all(model);

    // A stale temp from an interrupted write cannot corrupt the good file.
    expect(write_all(model_tmp, { 'x', 'x', 'x' }), "leave stale temp");
    Tensor check(2, 3); fill(check, -7.0f);
    auto info = Serialize::load_model(model.string(), { &check });
    expect(info.valid && same(a, check), "stale temp does not affect checkpoint");
    fs::remove(model_tmp);

    // Blocked temp path: the save fails and the old bytes/old model survive.
    fs::create_directory(model_tmp);
    a.data[0] = 99.0f;
    expect(!Serialize::save_model(model.string(), params, 4, 2, 16), "save_model fails when temp blocked");
    expect(read_all(model) == good_model, "old bytes survive failed save");
    fill(check, -7.0f);
    info = Serialize::load_model(model.string(), { &check });
    expect(info.valid && check.data[0] == 1.0f, "old model loadable after failed save");
    fs::remove(model_tmp);
    a.data[0] = 1.0f;

    // Poisoned tensors are refused before any file is touched.
    a.data[0] = std::numeric_limits<f32>::quiet_NaN();
    expect(!Serialize::save_model(model.string(), params, 4, 2, 16), "poisoned model refused");
    expect(read_all(model) == good_model, "poisoned save leaves old bytes intact");
    a.data[0] = 1.0f;

    // Optimizer sidecar: same publish contract.
    Tensor m(2, 3), v(2, 3);
    fill(m, 0.25f); fill(v, 0.5f);
    std::vector<Tensor*> ms{ &m }, vs{ &v };
    expect(Serialize::save_optimizer(opt.string(), ms, vs, 7), "atomic save_optimizer");
    expect(!fs::exists(opt_tmp), "no optimizer temp after successful save");
    const auto good_opt = read_all(opt);
    fs::create_directory(opt_tmp);
    fill(m, 9.0f);
    expect(!Serialize::save_optimizer(opt.string(), ms, vs, 8), "save_optimizer fails when temp blocked");
    expect(read_all(opt) == good_opt, "old optimizer bytes survive failed save");
    fs::remove(opt_tmp);
    fill(m, 0.25f);

    // AsyncSaver: two back-to-back spawns must join the previous writer instead
    // of assigning to a joinable std::thread (which calls std::terminate).
    {
        Serialize::AsyncSaver saver(amodel.string(), aopt.string());
        Tensor w(2, 3); fill(w, 3.0f);
        std::vector<Tensor*> wp{ &w };

        saver.queue_model(wp, 4, 2, 16);
        saver.spawn();
        fill(w, 5.0f);
        saver.queue_model(wp, 4, 2, 16);
        saver.spawn();
        saver.flush(); // process surviving is the negative control for terminate

        auto r = read_pair(amodel, aopt);
        expect(r.valid && r.info.dim == 4 && r.info.blocks == 2 && r.info.ctx == 16,
               "async model info");
        expect(r.model0 == 5.0f, "async second snapshot published");
        expect(!r.has_opt, "model-only generation has no optimizer");

        // NaN snapshot: keep the last good checkpoint.
        fill(w, std::numeric_limits<f32>::quiet_NaN());
        saver.queue_model(wp, 4, 2, 16);
        saver.spawn(); saver.flush();
        r = read_pair(amodel, aopt);
        expect(r.valid && r.model0 == 5.0f, "NaN snapshot keeps last good model");

        // Happy path with optimizer: generation-stamped model + sidecar.
        fill(w, 8.0f);
        saver.queue_model(wp, 4, 2, 16);
        saver.queue_opt(ms, vs, 33);
        saver.spawn(); saver.flush();
        r = read_pair(amodel, aopt);
        expect(r.valid && r.model0 == 8.0f && r.has_opt && r.t == 33, "async pair round-trip");
        expect(r.m0 == 0.25f && r.v0 == 0.5f, "async optimizer values");
    }

    for (const auto& p : { model, opt, amodel, aopt, model_tmp, opt_tmp, amodel_tmp, aopt_tmp })
        fs::remove(p);
    remove_pair_files(amodel, aopt);

    if (!ok) { std::fprintf(stderr, "atomic_checks: FAILED\n"); return 1; }
    std::puts("atomic_checks: OK");
    return 0;
}

// P2 from the 36a4b3b audit: two individually atomic files can still form a
// mismatched pair. The manifest must make the pair a single observable unit.
static int pair_checks() {
    const fs::path dir = fs::temp_directory_path();
    const fs::path pm = dir / "shade_pair_model.bin";
    const fs::path po = dir / "shade_pair_opt.bin";
    remove_pair_files(pm, po);
    fs::remove(pm); fs::remove(po);

    bool ok = true;
    auto expect = [&](bool cond, const char* what) {
        if (!cond) { std::fprintf(stderr, "FAIL: %s\n", what); ok = false; }
    };

    Tensor a(2, 3), m(2, 3), v(2, 3);
    fill(a, 1.0f); fill(m, 0.25f); fill(v, 0.5f);
    std::vector<Tensor*> ap{ &a }, ms{ &m }, vs{ &v };
    const fs::path manifest(pm.string() + ".manifest");

    // 1. First generation is published as a pair.
    expect(Serialize::CheckpointPair::publish(pm.string(), po.string(), ap,
                                              4, 2, 16, ms, vs, 7), "publish g1");
    auto r = read_pair(pm, po);
    expect(r.valid && r.model0 == 1.0f && r.has_opt && r.t == 7 && r.m0 == 0.25f,
           "g1 round-trip");
    expect(fs::exists(pm.string() + ".g1") && fs::exists(po.string() + ".g1"), "g1 files exist");

    // 2. P2 negative: model written, optimizer write blocked. The manifest must
    //    stay on g1 even though the stray g2 model is already on disk.
    const fs::path blocked(po.string() + ".g2.tmp");
    fs::create_directory(blocked);
    fill(a, 2.0f);
    expect(!Serialize::CheckpointPair::publish(pm.string(), po.string(), ap,
                                               4, 2, 16, ms, vs, 8),
           "publish fails when optimizer temp blocked");
    expect(fs::exists(pm.string() + ".g2"), "g2 model written before failure");
    expect(!fs::exists(po.string() + ".g2"), "g2 optimizer not written");
    r = read_pair(pm, po);
    expect(r.valid && r.model0 == 1.0f && r.has_opt && r.t == 7, "reader still sees old pair");

    // Prove why per-file atomicity is insufficient: both stray files load fine.
    Tensor stray(2, 3), sm(2, 3), sv(2, 3);
    fill(stray, -7.0f); fill(sm, -7.0f); fill(sv, -7.0f);
    i32 st = -1;
    expect(Serialize::load_model(pm.string() + ".g2", { &stray }).valid && stray.data[0] == 2.0f,
           "stray new model loads on its own");
    expect(Serialize::load_optimizer(po.string() + ".g1", { &sm }, { &sv }, st) && st == 7,
           "old optimizer loads on its own");
    fs::remove(blocked);

    // 3. Recovery: the same generation is published correctly.
    expect(Serialize::CheckpointPair::publish(pm.string(), po.string(), ap,
                                              4, 2, 16, ms, vs, 9), "publish g2 retry");
    r = read_pair(pm, po);
    expect(r.valid && r.model0 == 2.0f && r.t == 9, "g2 published after recovery");

    // 4. Corrupt manifest is rejected without touching the tensors.
    {
        auto bytes = read_all(manifest);
        expect(bytes.size() == 12, "manifest size");
        if (bytes.size() == 12) {
            const char orig = bytes[0];
            bytes[0] = 'X';
            expect(write_all(manifest, bytes), "corrupt manifest");
            r = read_pair(pm, po);
            expect(!r.valid && r.model0 == -7.0f && r.m0 == -7.0f, "corrupt manifest rejected");
            bytes[0] = orig;
            expect(write_all(manifest, bytes), "restore manifest");

            auto huge = bytes;
            const i32 big_gen = 2000000;
            std::memcpy(huge.data() + 4, &big_gen, sizeof(big_gen));
            expect(write_all(manifest, huge), "write huge-gen manifest");
            r = read_pair(pm, po);
            expect(!r.valid && r.model0 == -7.0f, "huge generation rejected");
            expect(write_all(manifest, bytes), "restore manifest again");
        }
    }

    // 5. Valid manifest but poisoned optimizer generation: staged load must not
    //    commit the model half of the pair.
    fill(a, 3.0f);
    expect(Serialize::CheckpointPair::publish(pm.string(), po.string(), ap,
                                              4, 2, 16, ms, vs, 11), "publish g3");
    expect(poke_nan(fs::path(po.string() + ".g3"), 12), "poison g3 optimizer");
    r = read_pair(pm, po);
    expect(!r.valid && r.model0 == -7.0f && r.m0 == -7.0f, "poisoned pair leaves tensors untouched");

    // 6. Next publish advances the manifest and collects old generations.
    fill(a, 4.0f);
    expect(Serialize::CheckpointPair::publish(pm.string(), po.string(), ap,
                                              4, 2, 16, ms, vs, 13), "publish g4");
    expect(!fs::exists(pm.string() + ".g2") && !fs::exists(po.string() + ".g2"),
           "old generation collected");
    expect(fs::exists(pm.string() + ".g4"), "latest generation kept");
    r = read_pair(pm, po);
    expect(r.valid && r.model0 == 4.0f && r.t == 13, "g4 round-trip");

    // 7. Model-only generation: no optimizer expected, moments untouched.
    const std::vector<Tensor*> none;
    expect(Serialize::CheckpointPair::publish(pm.string(), po.string(), ap,
                                              4, 2, 16, none, none, 0), "publish g5 model-only");
    r = read_pair(pm, po);
    expect(r.valid && r.model0 == 4.0f && !r.has_opt && r.m0 == -7.0f, "model-only pair");

    // 8. Legacy plain files remain readable when there is no manifest.
    remove_pair_files(pm, po);
    fs::remove(pm); fs::remove(po);
    fill(a, 6.0f);
    expect(Serialize::save_model(pm.string(), ap, 4, 2, 16), "legacy save model");
    expect(Serialize::save_optimizer(po.string(), ms, vs, 21), "legacy save opt");
    r = read_pair(pm, po);
    expect(r.valid && r.model0 == 6.0f && r.has_opt && r.t == 21, "legacy pair fallback");

    remove_pair_files(pm, po);
    fs::remove(pm); fs::remove(po);
    if (!ok) { std::fprintf(stderr, "pair_checks: FAILED\n"); return 1; }
    std::puts("pair_checks: OK");
    return 0;
}

int main() {
    const fs::path dir = fs::temp_directory_path();
    const fs::path model = dir / "shade_check_model.bin";
    const fs::path bad = dir / "shade_check_bad.bin";
    const fs::path plain = dir / "shade_check_plain.bin";
    const fs::path opt = dir / "shade_check_opt.bin";
    for (const auto& p : { model, bad, plain, opt }) fs::remove(p);

    bool ok = true;
    auto expect = [&](bool cond, const char* what) {
        if (!cond) { std::fprintf(stderr, "FAIL: %s\n", what); ok = false; }
    };

    Tensor a(2, 3), b(1, 4);
    for (i32 i = 0; i < a.size(); i++) a.data[i] = (f32)(i + 1);
    for (i32 i = 0; i < b.size(); i++) b.data[i] = (f32)(i + 10);
    std::vector<Tensor*> params{ &a, &b };

    expect(Serialize::save_model(model.string(), params, 8, 3, 32), "save_model");
    expect(Serialize::read_model_info(model.string()).valid, "read_model_info valid");

    // Happy path.
    Tensor x(2, 3), y(1, 4);
    std::vector<Tensor*> dst{ &x, &y };
    auto info = Serialize::load_model(model.string(), dst);
    expect(info.valid && info.dim == 8 && info.blocks == 3 && info.ctx == 32, "load_model info");
    expect(same(a, x) && same(b, y), "load_model values");

    // Poisoned weight (first float of the first tensor): 4*4 header + 4 count + 4 rows + 4 cols.
    fill(x, -7.0f); fill(y, -7.0f);
    expect(poke_nan(model, 28), "poke NaN");
    info = Serialize::load_model(model.string(), dst);
    expect(!info.valid, "poisoned model rejected");
    expect(x.data[0] == -7.0f && y.data[0] == -7.0f, "poisoned model leaves tensors unchanged");

    // Truncated and trailing-garbage files.
    expect(truncate_by(model, bad, 4), "truncate model");
    info = Serialize::load_model(bad.string(), dst);
    expect(!info.valid, "truncated model rejected");
    expect(x.data[0] == -7.0f && y.data[0] == -7.0f, "truncated model leaves tensors unchanged");

    expect(append_garbage(model, bad), "append garbage");
    info = Serialize::load_model(bad.string(), dst);
    expect(!info.valid, "trailing garbage rejected");

    // Wrong parameter count and wrong shape.
    Tensor one(2, 3); fill(one, -7.0f);
    info = Serialize::load_model(model.string(), { &one });
    expect(!info.valid, "wrong param count rejected");
    expect(one.data[0] == -7.0f, "wrong count leaves tensor unchanged");

    Tensor wrong(3, 2), y2(1, 4); fill(wrong, -7.0f); fill(y2, -7.0f);
    info = Serialize::load_model(model.string(), { &wrong, &y2 });
    expect(!info.valid, "shape mismatch rejected");
    expect(wrong.data[0] == -7.0f && y2.data[0] == -7.0f, "shape mismatch leaves tensors unchanged");

    // Plain save/load (no SHDE header) uses offset 12 for the first float.
    Tensor plain_x(2, 3), plain_y(1, 4);
    std::vector<Tensor*> plain_dst{ &plain_x, &plain_y };
    expect(Serialize::save(plain.string(), params), "save plain");
    expect(Serialize::load(plain.string(), plain_dst), "load plain");
    expect(same(a, plain_x) && same(b, plain_y), "plain values");
    fill(plain_x, -7.0f); fill(plain_y, -7.0f);
    expect(poke_nan(plain, 12), "poke plain NaN");
    expect(!Serialize::load(plain.string(), plain_dst), "poisoned plain rejected");
    expect(plain_x.data[0] == -7.0f && plain_y.data[0] == -7.0f, "plain failure leaves tensors unchanged");

    // Optimizer state: truncation must not change m, v or t.
    Tensor m0(2, 3), v0(2, 3), m1(1, 4), v1(1, 4);
    for (i32 i = 0; i < m0.size(); i++) { m0.data[i] = 0.1f * i; v0.data[i] = 0.01f * i; }
    for (i32 i = 0; i < m1.size(); i++) { m1.data[i] = 0.2f * i; v1.data[i] = 0.02f * i; }
    std::vector<Tensor*> ms{ &m0, &m1 }, vs{ &v0, &v1 };
    expect(Serialize::save_optimizer(opt.string(), ms, vs, 17), "save_optimizer");

    Tensor sm0(2, 3), sv0(2, 3), sm1(1, 4), sv1(1, 4);
    std::vector<Tensor*> dms{ &sm0, &sm1 }, dvs{ &sv0, &sv1 };
    i32 t = 0;
    expect(Serialize::load_optimizer(opt.string(), dms, dvs, t), "load_optimizer");
    expect(t == 17 && same(m0, sm0) && same(v0, sv0) && same(m1, sm1) && same(v1, sv1),
           "optimizer values");

    fill(sm0, -7.0f); fill(sv0, -7.0f); fill(sm1, -7.0f); fill(sv1, -7.0f);
    t = 0;
    expect(truncate_by(opt, bad, 4), "truncate optimizer");
    expect(!Serialize::load_optimizer(bad.string(), dms, dvs, t), "truncated optimizer rejected");
    expect(t == 0 && sm0.data[0] == -7.0f && sv0.data[0] == -7.0f
           && sm1.data[0] == -7.0f && sv1.data[0] == -7.0f,
           "optimizer failure leaves state unchanged");

    for (const auto& p : { model, bad, plain, opt }) fs::remove(p);
    if (atomic_and_async_checks() != 0) ok = false;
    if (pair_checks() != 0) ok = false;

    if (!ok) { std::fprintf(stderr, "serialize_checks: FAILED\n"); return 1; }
    std::puts("serialize_checks: OK");
    return 0;
}
