// Self-contained GPT checkpoint: round-trip, corruption, budget, top-k, atomic save.
// gpt.cpp's CLI main is renamed so its GPT/checkpoint helpers can be tested.
#define main gpt_cli_main
#include "../src/gpt.cpp"
#undef main

#include <algorithm>
#include <cmath>
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
static int nonzero(const std::vector<float>& p) {
    return (int)std::count_if(p.begin(), p.end(), [](float v) { return v > 0.0f; });
}

int main() {
    const fs::path dir = fs::temp_directory_path();
    const fs::path ck = dir / "shade_gpt_ck.bin";
    const fs::path bad = dir / "shade_gpt_bad.bin";
    const fs::path tokf = dir / "shade_tok.bin";
    for (const auto& p : { ck, bad, tokf }) fs::remove(p);

    bool ok = true;
    auto expect = [&](bool cond, const char* what) {
        if (!cond) { std::fprintf(stderr, "FAIL: %s\n", what); ok = false; }
    };

    Tokenizer tok;
    tok.build("the quick brown fox jumps over the lazy dog and the dog runs again", 32);

    GPT gpt(tok.vocab_size, 16, 2, 8);
    for (auto* p : gpt.params())
        for (i32 i = 0; i < p->size(); i++)
            p->data[i] = 0.01f * (f32)((i % 17) - 8); // deterministic, finite

    expect(save_gpt(ck.string(), gpt, tok), "save_gpt");

    GPTConfig cfg;
    Tokenizer tok2;
    std::vector<Tensor> weights;
    expect(read_gpt_checkpoint(ck.string(), cfg, tok2, weights), "read_gpt_checkpoint");
    expect(cfg.vocab == tok.vocab_size && cfg.dim == 16 && cfg.blocks == 2 && cfg.max_len == 8,
           "config round-trip");
    expect(tok2.id2word == tok.id2word, "tokenizer round-trip");

    // The shapes derived from config must exactly match GPT::params().
    {
        auto shapes = gpt_expected_shapes(cfg);
        auto live = gpt.params();
        expect(shapes.size() == live.size(), "expected tensor count matches GPT");
        for (size_t i = 0; i < shapes.size() && i < live.size(); i++)
            expect(shapes[i].rows == live[i]->rows && shapes[i].cols == live[i]->cols,
                   "expected tensor shape matches GPT");
        int64_t params = 0;
        expect(gpt_config_budget(cfg, params), "valid config passes budget");
        int64_t actual = 0;
        for (auto* p : live) actual += p->size();
        expect(params == actual, "budget param count matches GPT");
        std::printf("  params: budget=%lld actual=%lld tensors=%zu\n",
                    (long long)params, (long long)actual, shapes.size());
    }

    // Rebuild exactly like infer_gpt() does and compare logits.
    GPT gpt2(cfg.vocab, cfg.dim, cfg.blocks, cfg.max_len);
    auto params = gpt2.params();
    expect(params.size() == weights.size(), "weight count");
    for (size_t i = 0; i < params.size() && i < weights.size(); i++) {
        expect(params[i]->rows == weights[i].rows && params[i]->cols == weights[i].cols,
               "weight shape");
        params[i]->data = std::move(weights[i].data);
    }

    Tensor ids(1, 5);
    for (i32 i = 0; i < 5; i++) ids.data[i] = (f32)(i % tok.vocab_size);
    Tensor a = gpt.forward(ids), b = gpt2.forward(ids);
    f32 worst = 0;
    for (i32 i = 0; i < a.size(); i++)
        worst = std::fmax(worst, std::fabs(a.data[i] - b.data[i]));
    expect(worst == 0.0f, "reloaded model produces identical logits");
    std::printf("  max logit diff: %g\n", worst);

    // ---- top-k selection: exactly k survive, even when probabilities tie ----
    {
        std::vector<float> p{ 0.10f, 0.30f, 0.05f, 0.40f, 0.15f };
        float s = keep_top_k(p, 3);
        expect(nonzero(p) == 3, "top-k keeps exactly 3 of 5");
        expect(p[1] > 0 && p[3] > 0 && p[4] > 0 && p[0] == 0 && p[2] == 0,
               "top-k keeps the 3 largest");
        expect(std::fabs(s - 0.85f) < 1e-6f, "top-k surviving sum");
    }
    {
        std::vector<float> p(5, 0.2f); // every probability identical
        keep_top_k(p, 3);
        expect(nonzero(p) == 3, "top-k with ties keeps exactly 3");
        expect(p[0] > 0 && p[1] > 0 && p[2] > 0 && p[3] == 0 && p[4] == 0,
               "top-k tie-break goes to lower indices");
    }
    {
        std::vector<float> p{ 0.5f, 0.5f };
        keep_top_k(p, 10);
        expect(nonzero(p) == 2, "top_k >= n keeps everything");
    }

    // ---- corrupt files must be rejected, not parsed into a broken model ----
    auto bytes = read_all(ck);
    expect(bytes.size() > 64, "checkpoint size");

    auto truncate = [&](size_t n) {
        auto b = bytes; b.resize(b.size() - n); return write_all(bad, b);
    };
    expect(truncate(4) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "truncated checkpoint rejected");

    auto append = bytes; append.push_back(0x7f);
    expect(write_all(bad, append) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "trailing garbage rejected");

    auto poke_i32 = [&](size_t off, i32 v) {
        auto b = bytes;
        std::memcpy(b.data() + off, &v, sizeof(v));
        return write_all(bad, b);
    };
    auto poke2_i32 = [&](size_t off1, i32 v1, size_t off2, i32 v2) {
        auto b = bytes;
        std::memcpy(b.data() + off1, &v1, sizeof(v1));
        std::memcpy(b.data() + off2, &v2, sizeof(v2));
        return write_all(bad, b);
    };
    auto poke_byte = [&](size_t off, char v) {
        auto b = bytes;
        b[off] = v;
        return write_all(bad, b);
    };

    expect(poke_i32(0, 0) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "bad magic rejected");
    expect(poke_i32(8, 10000000) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "absurd vocab rejected");
    expect(poke_i32(12, 0) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "invalid dim rejected");

    // Individually legal dim/ctx combine into a 6.5e10-element activation.
    expect(poke2_i32(12, 65536, 20, 1000000) &&
           !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "combined config budget rejected");

    // vocab passes its own cap but makes the *model* (params + grads) huge.
    expect(poke_i32(8, 1000000) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "model budget rejected before GPT construction");

    // Long context blows up the T*T attention buffers even with a tiny dim.
    expect(poke_i32(20, 1000000) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "attention activation budget rejected");

    // TransformerBlock/self-attention require dim divisible by 4.
    expect(poke_i32(12, 15) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "dim not divisible by 4 rejected");

    // "<PAD>" starts at 24 (header) + 4 (word count) + 4 (its length).
    expect(poke_byte(32, 'X') && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "corrupt special token in checkpoint rejected");

    // First tensor block sits after the tokenized vocabulary and tensor count.
    size_t tensor_off = 24 + 4;
    for (const auto& w : tok.id2word) tensor_off += 4 + w.size();
    tensor_off += 4;
    expect(poke2_i32(tensor_off, 100000, tensor_off + 4, 100000) &&
           !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "oversized tensor rejected before allocation");

    // Tensor count and shapes are now validated against the config while reading.
    expect(poke_i32(tensor_off - 4, 1) &&
           !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "wrong tensor count rejected");
    expect(poke_i32(tensor_off + 4, 8) &&
           !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "wrong tensor shape rejected");

    // ---- standalone tokenizer file: round-trip + special-token check ----
    expect(tok.save(tokf.string()), "tok.save");
    Tokenizer tok3;
    expect(tok3.load(tokf.string()), "tok.load");
    expect(tok3.id2word == tok.id2word && tok3.vocab_size == tok.vocab_size,
           "tokenizer file round-trip");

    auto tokbytes = read_all(tokf);
    tokbytes[8] = 'X'; // "<PAD>" -> "X<PAD"
    expect(write_all(bad, tokbytes) && !tok3.load(bad.string()),
           "corrupt special token in tokenizer file rejected");

    // ---- atomic save: no leftover temp, and a failed write keeps the old model ----
    expect(!fs::exists(ck.string() + ".tmp"), "no temp file after successful save");
    expect(save_gpt(ck.string(), gpt, tok), "re-save over existing checkpoint");
    expect(read_gpt_checkpoint(ck.string(), cfg, tok2, weights), "reload after overwrite");

    const fs::path blocked = fs::path(ck.string() + ".tmp");
    fs::create_directory(blocked);
    expect(!save_gpt(ck.string(), gpt, tok), "save fails when temp path is blocked");
    expect(read_gpt_checkpoint(ck.string(), cfg, tok2, weights),
           "previous checkpoint survives a failed save");
    fs::remove(blocked);

    for (const auto& p : { ck, bad, tokf }) fs::remove(p);

    if (!ok) { std::fprintf(stderr, "gpt_checkpoint_check: FAILED\n"); return 1; }
    std::puts("gpt_checkpoint_check: OK");
    return 0;
}
