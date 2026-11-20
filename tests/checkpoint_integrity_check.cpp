// Checkpoint integrity: SHDG v2 checksum + backward-compatible v1 reading.
// Negative cases verify BOTH rejection and that no partial state leaks to the
// caller. Cases marked "CRC fixed" prove checksum != validation.
#define main gpt_cli_main
#include "../src/gpt.cpp"
#undef main

#include <algorithm>
#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

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
static void put_i32(std::vector<char>& b, size_t off, i32 v) {
    std::memcpy(b.data() + off, &v, sizeof(v));
}
static void put_u32(std::vector<char>& b, size_t off, uint32_t v) {
    std::memcpy(b.data() + off, &v, sizeof(v));
}
static i32 get_i32(const std::vector<char>& b, size_t off) {
    i32 v; std::memcpy(&v, b.data() + off, sizeof(v)); return v;
}

// Same CRC as gpt.cpp, applied directly to the byte buffer (covered ranges
// [0,8) and [12,end); the checksum field itself is excluded).
static uint32_t crc_of(const std::vector<char>& b) {
    uint32_t c = 0xFFFFFFFFu;
    if (b.size() >= 8) c = gpt_crc32_update(c, b.data(), 8);
    if (b.size() > 12) c = gpt_crc32_update(c, b.data() + 12, b.size() - 12);
    return c ^ 0xFFFFFFFFu;
}
static void fix_crc(std::vector<char>& b) {
    if (b.size() >= 12) put_u32(b, 8, crc_of(b));
}

struct Snap { GPTConfig cfg; Tokenizer tok; std::vector<Tensor> w; };
static bool same_state(const Snap& s, const GPTConfig& c, const Tokenizer& t,
                       const std::vector<Tensor>& w) {
    if (s.cfg.vocab != c.vocab || s.cfg.dim != c.dim ||
        s.cfg.blocks != c.blocks || s.cfg.max_len != c.max_len) return false;
    if (s.tok.vocab_size != t.vocab_size || s.tok.id2word != t.id2word) return false;
    if (s.w.size() != w.size()) return false;
    for (size_t i = 0; i < w.size(); i++)
        if (s.w[i].rows != w[i].rows || s.w[i].cols != w[i].cols ||
            s.w[i].data != w[i].data) return false;
    return true;
}

int main() {
    const fs::path dir = fs::temp_directory_path();
    const fs::path good = dir / "shade_integrity_v2.bin";
    const fs::path good2 = dir / "shade_integrity_v2b.bin";
    const fs::path bad = dir / "shade_integrity_bad.bin";
    const fs::path v2fromv1 = dir / "shade_integrity_from_v1.bin";
    for (const auto& p : { good, good2, bad, v2fromv1 }) fs::remove(p);

    bool ok = true;
    auto expect = [&](bool cond, const char* what) {
        if (!cond) { std::fprintf(stderr, "FAIL: %s\n", what); ok = false; }
    };

    Tokenizer tok;
    tok.build("the quick brown fox jumps over the lazy dog and the dog runs again", 32);
    GPT gpt(tok.vocab_size, 16, 2, 8);
    for (auto* p : gpt.params())
        for (i32 i = 0; i < p->size(); i++)
            p->data[i] = 0.01f * (f32)((i % 17) - 8);

    // ---- valid v2: write, verify checksum field, round-trip ----
    expect(save_gpt(good.string(), gpt, tok), "save v2");
    auto bytes = read_all(good);
    expect(bytes.size() > 128, "v2 file is non-trivial");
    expect(get_i32(bytes, 0) == GPT_MAGIC, "v2 magic");
    expect(get_i32(bytes, 4) == GPT_VERSION_V2, "v2 version tag");
    {
        uint32_t stored = 0; std::memcpy(&stored, bytes.data() + 8, 4);
        expect(stored == crc_of(bytes), "stored CRC covers all other bytes");
    }
    GPTConfig cfg{}; Tokenizer rt_tok; std::vector<Tensor> rt_w;
    expect(read_gpt_checkpoint(good.string(), cfg, rt_tok, rt_w), "read v2");
    expect(cfg.vocab == tok.vocab_size && cfg.dim == 16 && cfg.blocks == 2 && cfg.max_len == 8,
           "v2 config round-trip");
    expect(rt_tok.id2word == tok.id2word, "v2 tokenizer round-trip");
    {
        auto live = gpt.params();
        expect(live.size() == rt_w.size(), "v2 tensor count round-trip");
        for (size_t i = 0; i < live.size() && i < rt_w.size(); i++)
            expect(live[i]->data == rt_w[i].data, "v2 weight round-trip");
    }
    expect(save_gpt(good2.string(), gpt, tok) && read_all(good2) == bytes,
           "v2 serialization is deterministic");

    // Snapshot used to prove failures never mutate caller state.
    Snap snap{ cfg, rt_tok, rt_w };
    auto reject = [&](const std::vector<char>& b, const char* what) {
        if (!write_all(bad, b)) { expect(false, "write corrupt file"); return; }
        GPTConfig c = snap.cfg; Tokenizer t = snap.tok; std::vector<Tensor> w = snap.w;
        expect(!read_gpt_checkpoint(bad.string(), c, t, w), what);
        expect(same_state(snap, c, t, w), "no partial state after rejection");
    };

    // ---- structural corruption ----
    reject({}, "zero-length file rejected");
    { auto b = bytes; b.resize(7); reject(b, "shorter than header rejected"); }
    { auto b = bytes; put_i32(b, 0, 0); reject(b, "bad magic rejected"); }
    { auto b = bytes; put_i32(b, 4, 3); reject(b, "unknown version rejected"); }
    { auto b = bytes; b.resize(b.size() - 1); reject(b, "truncated by one byte rejected"); }
    { auto b = bytes; b.resize(b.size() - 100); reject(b, "heavily truncated rejected"); }

    // ---- checksum detection ----
    { auto b = bytes; b[8] ^= 0x01; reject(b, "wrong checksum rejected"); }
    // One flipped byte anywhere in config / tokenizer / weights is caught.
    size_t tok_off = 28 + 4;                 // v2 header + tokenizer word count
    size_t tensor_off = tok_off;
    for (const auto& w : tok.id2word) tensor_off += 4 + w.size();
    tensor_off += 4;                         // tensor count
    {
        auto b = bytes; b[tensor_off + 8 + 13] ^= 0x20;
        reject(b, "flipped weight byte rejected by checksum");
    }
    {
        auto b = bytes; b.push_back(0x7f);
        reject(b, "trailing byte rejected (checksum covers it)");
    }
    {
        auto b = bytes; b.push_back(0x7f); fix_crc(b);
        reject(b, "trailing byte rejected even with valid checksum (EOF policy)");
    }

    // ---- semantic validation with a RECOMPUTED checksum: checksum is not enough ----
    {
        auto b = bytes; put_i32(b, 12, 1000000); fix_crc(b);
        reject(b, "absurd vocab rejected despite valid checksum");
    }
    {
        auto b = bytes; put_i32(b, 16, INT_MAX); fix_crc(b);
        reject(b, "overflow dim rejected despite valid checksum");
    }
    {
        auto b = bytes; put_i32(b, 16, 65536); put_i32(b, 24, 1000000); fix_crc(b);
        reject(b, "combined config budget rejected despite valid checksum");
    }
    {
        auto b = bytes; b[tok_off + 4 + 4] = 'X'; fix_crc(b); // first char of "<PAD>"
        reject(b, "corrupt special token rejected despite valid checksum");
    }
    {
        auto b = bytes; put_i32(b, tensor_off - 4, 1); fix_crc(b);
        reject(b, "wrong tensor count rejected despite valid checksum");
    }
    {
        auto b = bytes; put_i32(b, tensor_off, 100000); fix_crc(b);
        reject(b, "oversized tensor rejected before allocation");
    }
    {
        auto b = bytes; put_i32(b, tensor_off + 4, -1); fix_crc(b);
        reject(b, "negative tensor dimension rejected");
    }
    {
        auto b = bytes;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(b.data() + tensor_off + 8, &nan, 4); fix_crc(b);
        reject(b, "NaN weight rejected despite valid checksum");
    }

    // ---- stable SHDG v1 fixture: read-only compatibility, re-save as v2 ----
    fs::path fixture;
    for (const char* c : { "tests/fixtures/shdg_v1_min.bin", "fixtures/shdg_v1_min.bin" }) {
        if (fs::exists(c)) { fixture = c; break; }
    }
    expect(!fixture.empty(), "v1 fixture found");
    if (!fixture.empty()) {
        auto vb = read_all(fixture);
        expect(vb.size() == 1408 && get_i32(vb, 0) == GPT_MAGIC && get_i32(vb, 4) == 1,
               "v1 fixture is a stable 1408-byte SHDG v1 file");

        GPTConfig fc{}; Tokenizer ft; std::vector<Tensor> fw;
        expect(read_gpt_checkpoint(fixture.string(), fc, ft, fw), "v1 fixture loads");
        expect(fc.vocab == 8 && fc.dim == 4 && fc.blocks == 1 && fc.max_len == 4,
               "v1 fixture config");
        expect(ft.id2word == std::vector<std::string>(
                   { "<PAD>", "<UNK>", "<BOS>", "<EOS>", "a", "b", "c", "d" }),
               "v1 fixture tokenizer");
        expect(fw.size() == 15 && fw[0].rows == 8 && fw[0].cols == 4,
               "v1 fixture tensor layout");
        // Control values: weight i = 0.01f * (i % 5).
        expect(fw[0].data[0] == 0.0f && fw[0].data[1] == 0.01f &&
               fw[0].data[5] == 0.0f && fw[0].data[7] == 0.02f,
               "v1 fixture control weights");

        GPT fg(fc.vocab, fc.dim, fc.blocks, fc.max_len);
        auto fp = fg.params();
        expect(fp.size() == fw.size(), "v1 fixture matches GPT params");
        for (size_t i = 0; i < fp.size() && i < fw.size(); i++)
            fp[i]->data = fw[i].data;
        expect(save_gpt(v2fromv1.string(), fg, ft), "re-save v1 as v2");
        auto nb = read_all(v2fromv1);
        expect(get_i32(nb, 4) == GPT_VERSION_V2, "re-saved file carries v2 tag");
        uint32_t stored = 0; std::memcpy(&stored, nb.data() + 8, 4);
        expect(stored == crc_of(nb), "re-saved v2 checksum valid");

        GPTConfig nc{}; Tokenizer nt2; std::vector<Tensor> nw;
        expect(read_gpt_checkpoint(v2fromv1.string(), nc, nt2, nw), "re-saved v2 loads");
        expect(nc.vocab == fc.vocab && nc.dim == fc.dim && nc.blocks == fc.blocks &&
               nc.max_len == fc.max_len, "v1 -> v2 config preserved");
        expect(nw.size() == fw.size(), "v1 -> v2 tensor count preserved");
        for (size_t i = 0; i < nw.size() && i < fw.size(); i++)
            expect(nw[i].data == fw[i].data, "v1 -> v2 weights preserved");
    }

    for (const auto& p : { good, good2, bad, v2fromv1 }) fs::remove(p);
    if (!ok) { std::fprintf(stderr, "checkpoint_integrity_check: FAILED\n"); return 1; }
    std::puts("checkpoint_integrity_check: OK");
    return 0;
}
