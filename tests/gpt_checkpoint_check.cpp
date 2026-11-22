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
    const fs::path badcfg = dir / "shade_gpt_badcfg.bin";
    const fs::path badvocab = dir / "shade_gpt_badvocab.bin";
    for (const auto& p : { ck, bad, tokf, badcfg, badvocab }) fs::remove(p);

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

    // ---- activation ledger: global lifetime peak, not per-block transients ----
    {
        const int64_t t = cfg.max_len, d = cfg.dim;
        int64_t p = 0, f = 0, w = 0;
        expect(block_persistent_act(t, d, p), "persistent peak computes");
        expect(block_forward_transient(t, d, f), "forward peak computes");
        expect(block_backward_transient(t, d, w), "backward peak computes");

        GPTConfig one = cfg; one.blocks = 1;
        GPTConfig two = cfg; two.blocks = 2;
        GPTConfig eight = cfg; eight.blocks = 8;
        GPTConfig hundred = cfg; hundred.blocks = 100;
        int64_t a1 = -1, a2 = -1, a8 = -1, a100 = -1;
        expect(gpt_activation_estimate(one, a1) && gpt_activation_estimate(two, a2) &&
               gpt_activation_estimate(eight, a8) && gpt_activation_estimate(hundred, a100),
               "activation estimate computes");
        // Only PERSISTENT caches scale with blocks; the transient peak is global.
        expect(a2 - a1 == p && a8 - a1 == 7 * p && a100 - a1 == 99 * p,
               "only persistent activation scales with blocks");
        expect(a1 == p + (w > f ? w : f) + 4 * t * cfg.vocab + 3 * t * d,
               "one-block estimate equals P + max(F,W) + G");
        std::printf("  activations: 1 block=%lld, 8 blocks=%lld (P=%lld F=%lld W=%lld)\n",
                    (long long)a1, (long long)a8, (long long)p, (long long)f, (long long)w);
    }
    {
        // Each phase peak is auditable on its own and combines globally.
        const int64_t t = cfg.max_len, d = cfg.dim;
        int64_t p = 0, f = 0, b = 0, total = 0;
        expect(block_persistent_act(t, d, p) && p == 28 * t * d + 6 * t * t + 4 * t,
               "persistent phase peak matches ledger");
        const int64_t fwd_expected = (10 * t * d > t * t + t * d) ? 10 * t * d : t * t + t * d;
        expect(block_forward_transient(t, d, f) && f == fwd_expected,
               "forward phase peak is max(T^2+Td, 10Td), not a sum");
        expect(block_backward_transient(t, d, b) && b == 21 * t * d + 2 * t * t,
               "backward phase peak matches nested-call ledger");
        expect(gpt_activation_estimate(cfg, total) &&
               total == cfg.blocks * p + (b > f ? b : f) + 4 * t * cfg.vocab + 3 * t * d,
               "activation estimate equals B*P + max(F,W) + G");
    }
    {
        // Many blocks x long ctx: the old flat 2*T^2 estimate accepted this.
        GPTConfig big = cfg; big.blocks = 100; big.max_len = 1000;
        int64_t p = 0;
        expect(2 * (int64_t)big.max_len * big.max_len <= GPT_MAX_ACTIVATION_ELEMS,
               "regression setup: old flat estimate would accept");
        expect(!gpt_config_budget(big, p), "many-block activation budget rejected");
    }
    {
        // Large ctx, tiny dim: the quadratic T^2 attention term dominates.
        GPTConfig c = cfg; c.max_len = 1000000; c.dim = 4; c.blocks = 1;
        int64_t p = 0;
        expect(!gpt_config_budget(c, p), "large ctx / small dim rejected");
    }
    {
        // Small ctx, large dim: the linear 51*T*d term is the discriminator.
        GPTConfig small = cfg; small.max_len = 100; small.dim = 1024; small.blocks = 1;
        GPTConfig big = small; big.max_len = 2000;
        int64_t ps = 0, pb = 0;
        expect(gpt_config_budget(small, ps), "small ctx / large dim accepted");
        expect(!gpt_config_budget(big, pb), "linear dim term rejects larger ctx");
    }
    {
        // Large vocab: T*v logits blow the GPT-level activation budget.
        GPTConfig c = cfg; c.vocab = 1000000; c.dim = 4; c.blocks = 1; c.max_len = 32;
        int64_t p = 0;
        expect(!gpt_config_budget(c, p), "large vocab rejected via logits budget");
    }
    {
        // Exact budget boundary: 2 blocks fit, 3 blocks cross the limit.
        GPTConfig edge; edge.vocab = 32; edge.dim = 4; edge.blocks = 2; edge.max_len = 1000;
        int64_t p = 0, a = 0;
        expect(gpt_activation_estimate(edge, a) && a <= GPT_MAX_ACTIVATION_ELEMS,
               "boundary: just under activation limit");
        expect(gpt_config_budget(edge, p), "boundary config accepted");
        edge.blocks = 3;
        expect(gpt_activation_estimate(edge, a) && a > GPT_MAX_ACTIVATION_ELEMS,
               "boundary: just over activation limit");
        expect(!gpt_config_budget(edge, p), "boundary config rejected");
    }
    {
        // Checked int64 arithmetic cannot silently wrap.
        int64_t x = 0;
        const int64_t imax = std::numeric_limits<int64_t>::max();
        expect(!ck_mul(imax, 2, x), "checked multiply rejects overflow");
        expect(!ck_add(imax, 1, x), "checked add rejects overflow");
        expect(ck_mul(1000, 1000, x) && x == 1000000, "checked multiply exact");
        expect(ck_add(1000, 1, x) && x == 1001, "checked add exact");
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

    // ---- non-finite values are rejected at every GPT trust boundary ----
    {
        GPT ng(tok.vocab_size, 8, 1, 4);
        for (auto* p : ng.params()) p->fill(0.01f);
        const i32 V = ng.vocab_size;
        Tensor logits(2, V), targets(2, V), d;
        for (i32 i = 0; i < 2; i++) {
            for (i32 j = 0; j < V; j++) {
                logits.data[i * V + j] = (f32)j;
                targets.data[i * V + j] = (j == i ? 1.0f : 0.0f);
            }
        }
        const f32 good_loss = ng.compute_loss(logits, targets, d);
        expect(std::isfinite(good_loss), "finite loss stays finite");
        logits.data[0] = std::numeric_limits<f32>::quiet_NaN();
        expect(!std::isfinite(ng.compute_loss(logits, targets, d)), "NaN logit rejected");
        logits.data[0] = 0.0f;
        targets.data[0] = std::numeric_limits<f32>::infinity();
        expect(!std::isfinite(ng.compute_loss(logits, targets, d)), "Inf target rejected");
        targets.data[0] = (f32)V;
        expect(!std::isfinite(ng.compute_loss(logits, targets, d)), "out-of-range target rejected");
        targets.data[0] = 1.0f;

        auto gr = ng.grads();
        gr[0]->fill(0.0f);
        expect(ng.clip_grads(1.0f), "finite gradients accepted");
        gr[0]->data[0] = std::numeric_limits<f32>::quiet_NaN();
        expect(!ng.clip_grads(1.0f), "NaN gradient rejected before optimizer step");
        gr[0]->data[0] = 0.0f;

        std::vector<float> probs((size_t)V, 0.0f);
        probs[0] = 1.0f;
        expect(ng.sample(probs.data(), V, 0.8f, 20) >= 0, "finite logits sample");
        probs[1] = std::numeric_limits<f32>::quiet_NaN();
        expect(ng.sample(probs.data(), V, 0.8f, 20) == -1, "NaN sampling logit rejected");

        const std::vector<int> seed{ Tokenizer::BOS };
        std::string text;
        expect(ng.generate(seed, 3, 0.8f, 20, tok, text), "finite generation succeeds");
        auto np = ng.params();
        std::vector<f32> saved;
        saved.reserve(np.size());
        for (auto* p : np) {
            saved.push_back(p->data[0]);
            p->data[0] = std::numeric_limits<f32>::quiet_NaN();
        }
        expect(!ng.generate(seed, 3, 0.8f, 20, tok, text), "poisoned generation is reported");
        const fs::path poisoned = dir / "shade_gpt_nan_save.bin";
        fs::remove(poisoned);
        fs::remove(fs::path(poisoned.string() + ".tmp"));
        expect(!save_gpt(poisoned.string(), ng, tok), "save rejects NaN weights before publish");
        expect(!fs::exists(poisoned), "no poisoned checkpoint published");
        for (size_t i = 0; i < np.size(); i++) np[i]->data[0] = saved[i];
        fs::remove(poisoned);
        fs::remove(fs::path(poisoned.string() + ".tmp"));
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

    // v2 header: magic@0, version@4, crc@8, vocab@12, dim@16, blocks@20, ctx@24.
    expect(poke_i32(0, 0) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "bad magic rejected");
    expect(poke_i32(12, 10000000) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "absurd vocab rejected");
    expect(poke_i32(16, 0) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "invalid dim rejected");

    // Individually legal dim/ctx combine into a 6.5e10-element activation.
    expect(poke2_i32(16, 65536, 24, 1000000) &&
           !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "combined config budget rejected");

    // vocab passes its own cap but makes the *model* (params + grads) huge.
    expect(poke_i32(12, 1000000) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "model budget rejected before GPT construction");

    // Long context blows up the T*T attention buffers even with a tiny dim.
    expect(poke_i32(24, 1000000) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "attention activation budget rejected");

    // TransformerBlock/self-attention require dim divisible by 4.
    expect(poke_i32(16, 15) && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "dim not divisible by 4 rejected");

    // "<PAD>" starts at 28 (v2 header) + 4 (word count) + 4 (its length).
    expect(poke_byte(36, 'X') && !read_gpt_checkpoint(bad.string(), cfg, tok2, weights),
           "corrupt special token in checkpoint rejected");

    // First tensor block sits after the tokenized vocabulary and tensor count.
    size_t tensor_off = 28 + 4;
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

    // Writer/reader symmetry: a configuration the reader rejects must never be
    // published. Both cases construct a live GPT, so only the writer guard
    // stands between them and an unloadable checkpoint.
    {
        GPT zero_blocks(tok.vocab_size, 8, 0, 4);
        expect(!save_gpt(badcfg.string(), zero_blocks, tok), "save rejects blocks=0");
        expect(!fs::exists(badcfg) && !fs::exists(badcfg.string() + ".tmp"),
               "rejected config leaves no checkpoint or temp");

        GPT wrong_vocab(tok.vocab_size + 1, 8, 1, 4);
        expect(!save_gpt(badvocab.string(), wrong_vocab, tok),
               "save rejects config/tokenizer vocab mismatch");
        expect(!fs::exists(badvocab) && !fs::exists(badvocab.string() + ".tmp"),
               "rejected vocab leaves no checkpoint or temp");
    }

    // ---- atomic save: no leftover temp, and a failed write keeps the old model ----
    expect(!fs::exists(ck.string() + ".tmp"), "no temp file after successful save");
    expect(save_gpt(ck.string(), gpt, tok), "re-save over existing checkpoint");
    expect(read_gpt_checkpoint(ck.string(), cfg, tok2, weights), "reload after overwrite");

    const auto before_failed_save = read_all(ck);
    const fs::path blocked = fs::path(ck.string() + ".tmp");
    fs::create_directory(blocked);
    expect(!save_gpt(ck.string(), gpt, tok), "save fails when temp path is blocked");
    expect(read_all(ck) == before_failed_save,
           "previous checkpoint bytes unchanged after failed save");
    expect(read_gpt_checkpoint(ck.string(), cfg, tok2, weights),
           "previous checkpoint survives a failed save");
    fs::remove(blocked);

    for (const auto& p : { ck, bad, tokf, badcfg, badvocab }) fs::remove(p);

    if (!ok) { std::fprintf(stderr, "gpt_checkpoint_check: FAILED\n"); return 1; }
    std::puts("gpt_checkpoint_check: OK");
    return 0;
}
