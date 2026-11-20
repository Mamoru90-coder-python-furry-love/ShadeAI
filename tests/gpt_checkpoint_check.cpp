// Self-contained GPT checkpoint: round-trip + corruption coverage.
// gpt.cpp's CLI main is renamed so its GPT/checkpoint helpers can be tested.
#define main gpt_cli_main
#include "../src/gpt.cpp"
#undef main

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

    // Corrupt files must be rejected, not parsed into a broken model.
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

    auto poke = [&](size_t off, i32 v) {
        auto b = bytes;
        std::memcpy(b.data() + off, &v, sizeof(v));
        return write_all(bad, b);
    };
    expect(poke(0, 0) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "bad magic rejected");
    expect(poke(8, 10000000) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "absurd vocab rejected");
    expect(poke(12, 0) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "invalid dim rejected");

    // Standalone tokenizer file round-trip.
    expect(tok.save(tokf.string()), "tok.save");
    Tokenizer tok3;
    expect(tok3.load(tokf.string()), "tok.load");
    expect(tok3.id2word == tok.id2word && tok3.vocab_size == tok.vocab_size,
           "tokenizer file round-trip");

    for (const auto& p : { ck, bad, tokf }) fs::remove(p);

    if (!ok) { std::fprintf(stderr, "gpt_checkpoint_check: FAILED\n"); return 1; }
    std::puts("gpt_checkpoint_check: OK");
    return 0;
}
