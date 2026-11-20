// Checkpoint integrity: truncated / poisoned / trailing-garbage files must fail
// without modifying any live tensor or the AdamW step counter.
#include "../src/utils/serialize.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
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

    if (!ok) { std::fprintf(stderr, "serialize_checks: FAILED\n"); return 1; }
    std::puts("serialize_checks: OK");
    return 0;
}
