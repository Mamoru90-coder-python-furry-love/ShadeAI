// End-to-end GPT resume: byte-exact continuity, transactional rejection of a
// corrupt resume section, config/tokenizer restore, and moment sensitivity.
// gpt.cpp's CLI main is renamed so the production functions can be driven.
#define main gpt_cli_main
#include "../src/gpt.cpp"
#undef main

#include <filesystem>
#include <fstream>
#include <iterator>

namespace fs = std::filesystem;

static std::vector<char> read_all(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return { std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>() };
}
static bool write_all(const fs::path& p, const std::vector<char>& b) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(b.data(), (std::streamsize)b.size());
    return f.good();
}
static bool same_bytes(const fs::path& a, const fs::path& b) {
    return read_all(a) == read_all(b);
}
// Recompute the v2/v3 checksum after a deliberate byte poke. This lets the
// controls test STRUCTURAL validation, not merely CRC detection.
static bool reseal(const fs::path& p) {
    FILE* f = fopen(p.string().c_str(), "r+b");
    if (!f) return false;
    uint32_t crc = 0;
    bool good = gpt_compute_file_crc(f, crc);
    if (good) good = fseek(f, (long)GPT_CRC_OFFSET, SEEK_SET) == 0;
    if (good) good = gpt_write_u32(f, crc);
    fclose(f);
    return good;
}
static bool poke_i32(const fs::path& p, size_t off, i32 v, bool seal) {
    auto b = read_all(p);
    if (off + sizeof(v) > b.size()) return false;
    std::memcpy(b.data() + off, &v, sizeof(v));
    if (!write_all(p, b)) return false;
    return seal ? reseal(p) : true;
}
// Offsets of the v3 resume section, derived from the file tail so the tests
// do not depend on tokenizer length.
struct ResumeOffsets { size_t epoch, has_opt, n, first_rows, t; };
static bool resume_offsets(const fs::path& p, const std::vector<Tensor>& m,
                           ResumeOffsets& out) {
    size_t blocks = 0;
    for (const auto& t : m) blocks += 8 + (size_t)t.rows * t.cols * 4;
    const size_t bytes = 16 + 4 + 2 * blocks + 4;
    const size_t size = read_all(p).size();
    if (size < bytes) return false;
    const size_t off = size - bytes;
    out.epoch = off;
    out.has_opt = off + 12;
    out.n = off + 16;
    out.first_rows = off + 20;
    out.t = off + 20 + 2 * blocks;
    return true;
}

int main() {
    const fs::path dir = fs::temp_directory_path();
    const fs::path data = dir / "shade_gpt_resume_data.txt";
    const fs::path full = dir / "shade_gpt_resume_full.bin";
    const fs::path part = dir / "shade_gpt_resume_part.bin";
    const fs::path bad = dir / "shade_gpt_resume_bad.bin";
    const fs::path v2 = dir / "shade_gpt_resume_v2.bin";
    const fs::path zero = dir / "shade_gpt_resume_zero.bin";
    for (const auto& p : { data, full, part, bad, v2, zero }) fs::remove(p);

    bool ok = true;
    auto expect = [&](bool cond, const char* what) {
        if (!cond) { std::fprintf(stderr, "FAIL: %s\n", what); ok = false; }
    };

    const std::string text =
        "the quick brown fox jumps over the lazy dog and the dog runs again "
        "into the forest where a slow robot waits for another quiet morning "
        "the story continues with a second line of ordinary training words";
    expect(write_all(data, std::vector<char>(text.begin(), text.end())), "write data");

    GPTTrainConfig small;
    small.dim = 8; small.blocks = 1; small.max_len = 4;
    small.epochs = 3; small.batch_size = 2; small.max_chars = 4096;
    small.verbose = false;
    GPTTrainConfig two = small;
    two.epochs = 2;

    // ---- 1. continuity: 3 epochs vs 2 epochs + resume, byte for byte ----
    train_gpt(data.string(), full.string(), false, small);
    train_gpt(data.string(), part.string(), false, two);
    train_gpt(data.string(), part.string(), true, small);
    expect(fs::exists(full) && fs::exists(part), "checkpoints written");
    {
        GPTConfig ca{}; Tokenizer ta; std::vector<Tensor> wa; GPTResume ra;
        GPTConfig cb{}; Tokenizer tb; std::vector<Tensor> wb; GPTResume rb;
        bool oka = read_gpt_checkpoint(full.string(), ca, ta, wa, Tokenizer::MAX_BYTES, &ra);
        bool okb = read_gpt_checkpoint(part.string(), cb, tb, wb, Tokenizer::MAX_BYTES, &rb);
        float worst = 0;
        if (oka && okb && wa.size() == wb.size())
            for (size_t i = 0; i < wa.size(); ++i)
                for (size_t j = 0; j < wa[i].data.size() && j < wb[i].data.size(); ++j)
                    worst = std::fmax(worst, std::fabs(wa[i].data[j] - wb[i].data[j]));
        std::printf("  full: ep=%d t=%d calls=%d | 2+resume: ep=%d t=%d calls=%d | worst|dw|=%.9g\n",
                    ra.epoch, ra.t, ra.rand_calls,
                    rb.epoch, rb.t, rb.rand_calls, worst);
    }
    expect(same_bytes(full, part), "3 epochs == 2 + resume (byte-identical)");

    GPTConfig cfg{}; Tokenizer tok; std::vector<Tensor> weights; GPTResume res;
    bool read_ok = read_gpt_checkpoint(full.string(), cfg, tok, weights,
                                       Tokenizer::MAX_BYTES, &res);
    expect(read_ok, "v3 checkpoint reads with resume state");
    if (read_ok) {
        expect(cfg.dim == 8 && cfg.blocks == 1 && cfg.max_len == 4,
               "architecture restored from checkpoint");
        expect(res.epoch == 3 && res.has_opt && res.t > 0 && res.rand_calls > 0,
               "progress, moments flag and RNG position restored");
        expect(res.m.size() == weights.size() && res.v.size() == weights.size(),
               "m/v tensor counts match weights");
        expect(tok.id2word.size() > 0, "tokenizer restored from checkpoint");

        // ---- 2. offset probe: proves the poke offsets below are correct ----
        expect(write_all(bad, read_all(full)), "stage bad checkpoint");
        ResumeOffsets o{};
        if (resume_offsets(full, res.m, o)) {
            GPTConfig c2{}; Tokenizer t2; std::vector<Tensor> w2; GPTResume r2;
            const bool poked = poke_i32(bad, o.epoch, 99, true);
            const bool reread = poked && read_gpt_checkpoint(bad.string(), c2, t2, w2,
                                                              Tokenizer::MAX_BYTES, &r2);
            // Confirms the structural pokes below hit the resume section.
            expect(poked && reread && r2.epoch == 99, "resume section offset probe");

            // ---- 3. transactional rejection; caller state must not move ----
            auto rejected = [&](const char* label, auto write_bad) {
                GPTConfig c3{}; Tokenizer t3;
                std::vector<Tensor> w3{ Tensor(1, 1, 123.0f) };
                GPTResume r3;
                r3.epoch = 777;
                const bool r = write_bad() && read_gpt_checkpoint(bad.string(), c3, t3, w3,
                                                                 Tokenizer::MAX_BYTES, &r3);
                expect(!r, label);
                expect(w3.size() == 1 && w3[0].data[0] == 123.0f,
                       "rejected load leaves weights untouched");
            };
            auto bytes = read_all(full);
            // CRC failure inside the resume section (no reseal).
            rejected("CRC-damaged resume state rejected", [&] {
                return poke_i32(bad, o.epoch, 98, false); // no reseal: CRC must fail
            });
            // Structural failures with a VALID CRC — these exercise the new checks.
            rejected("has_opt=2 rejected", [&] { return poke_i32(bad, o.has_opt, 2, true); });
            rejected("negative epoch rejected", [&] { return poke_i32(bad, o.epoch, -1, true); });
            rejected("too-large rand_calls rejected", [&] {
                return poke_i32(bad, o.epoch + 8, 100000001, true);
            });
            rejected("wrong optimizer tensor count rejected", [&] {
                return poke_i32(bad, o.n, (i32)res.m.size() + 1, true);
            });
            rejected("wrong moment shape rejected", [&] {
                return poke_i32(bad, o.first_rows, res.m[0].rows + 1, true);
            });
            rejected("negative t rejected", [&] { return poke_i32(bad, o.t, -1, true); });
            // Truncation of the resume tail.
            rejected("truncated resume section rejected", [&] {
                auto b = bytes; b.resize(b.size() - 4); return write_all(bad, b);
            });

            // ---- 4. sensitivity: zeroing m/v must change the continuation ----
            GPTResume zm = res;
            for (auto& t : zm.m) std::fill(t.data.begin(), t.data.end(), 0.0f);
            for (auto& t : zm.v) std::fill(t.data.begin(), t.data.end(), 0.0f);
            GPT rebuild(cfg.vocab, cfg.dim, cfg.blocks, cfg.max_len);
            auto rpp = rebuild.params();
            for (size_t i = 0; i < rpp.size(); ++i) rpp[i]->data = weights[i].data;
            expect(save_gpt(zero.string(), rebuild, tok, &zm),
                   "save zeroed-moment checkpoint");
            train_gpt(data.string(), zero.string(), true, small);
            expect(!same_bytes(full, zero), "resume without moments diverges (sensitivity)");
        }
    }

    // ---- 5. v1/v2 compatibility: inference still reads old files; explicit
    //         resume must refuse them instead of silently restarting. ----
    {
        GPTConfig c{}; Tokenizer t; std::vector<Tensor> w;
        expect(read_gpt_checkpoint(full.string(), c, t, w), "v3 readable without resume ptr");
        GPT g(c.vocab, c.dim, c.blocks, c.max_len);
        auto pp = g.params();
        for (size_t i = 0; i < pp.size(); ++i) pp[i]->data = std::move(w[i].data);
        expect(save_gpt(v2.string(), g, t), "write legacy v2 checkpoint");
        std::vector<Tensor> w2; GPTResume unused;
        expect(read_gpt_checkpoint(v2.string(), c, t, w2), "v2 still readable for inference");
        expect(!read_gpt_checkpoint(v2.string(), c, t, w2,
                                    Tokenizer::MAX_BYTES, &unused),
               "explicit resume rejects v2 (no resume state)");
    }

    for (const auto& p : { data, full, part, bad, v2, zero }) fs::remove(p);
    if (!ok) { std::fprintf(stderr, "gpt_resume_checks: FAILED\n"); return 1; }
    std::puts("gpt_resume_checks: OK");
    return 0;
}
