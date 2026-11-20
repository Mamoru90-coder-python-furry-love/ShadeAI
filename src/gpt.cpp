#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include "core/tensor.h"
#include "core/loss.h"
#include "layers/transformer.h"
#include "layers/dense.h"
#include "utils/embedding.h"
#include "utils/tokenizer.h"
#include "utils/serialize.h"
#include "optimizer/adamw.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <limits>
#include <memory>

// Keep exactly top_k largest probabilities. Ties break by lower index, so a
// uniform distribution cannot accidentally keep more than top_k tokens.
inline float keep_top_k(std::vector<float>& probs, int top_k) {
    const int n = (int)probs.size();
    if (top_k <= 0 || top_k >= n) {
        float s = 0;
        for (float p : probs) s += p;
        return s;
    }
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (probs[a] != probs[b]) return probs[a] > probs[b];
        return a < b;
    });
    std::vector<char> keep(n, 0);
    float s = 0;
    for (int i = 0; i < top_k; i++) { keep[order[i]] = 1; s += probs[order[i]]; }
    for (int i = 0; i < n; i++) if (!keep[i]) probs[i] = 0.0f;
    return s;
}

struct GPT {
    i32 vocab_size, dim, num_blocks, max_len;
    Embedding embed;
    PositionalEncoding pos_enc;
    std::vector<TransformerBlock> blocks;
    Dense output;

    GPT(i32 vs, i32 d, i32 nb, i32 ml)
        : vocab_size(vs), dim(d), num_blocks(nb), max_len(ml),
          embed(vs, d), pos_enc(ml, d), output(d, vs, false) {
        for (i32 i = 0; i < nb; i++)
            blocks.emplace_back(d);
    }

    Tensor forward(const Tensor& ids) {
        i32 T = ids.cols;
        Tensor h = embed.forward(ids);
        Tensor pe = pos_enc.forward(T);
        for (i32 i = 0; i < T; i++)
            for (i32 j = 0; j < dim; j++)
                h.at(i, j) += pe.at(i, j);
        for (auto& block : blocks)
            h = block.forward(h);
        return output.forward(h);
    }

    void backward(const Tensor& dlogits, const Tensor& ids) {
        Tensor dh = output.backward(dlogits);
        for (i32 i = num_blocks - 1; i >= 0; i--)
            dh = blocks[i].backward(dh);
        embed.backward(dh, ids);
    }

    float compute_loss(const Tensor& logits, const Tensor& targets, Tensor& dlogits) {
        i32 T = logits.rows;
        dlogits = Tensor(T, vocab_size, 0.0f);
        float loss = 0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(+:loss)
#endif
        for (i32 i = 0; i < T; i++) {
            int tgt = (int)targets.data[i];
            const f32* lp = &logits.data[i * vocab_size];
            float maxv = lp[0];
            for (i32 j = 1; j < vocab_size; j++)
                if (lp[j] > maxv) maxv = lp[j];
            // single exp pass: e[j] = exp(logit - max)
            float sum = 0;
            f32* dp = &dlogits.data[i * vocab_size];
            for (i32 j = 0; j < vocab_size; j++) {
                float e = expf(lp[j] - maxv);
                dp[j] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            float ptgt = dp[tgt] * inv;
            loss -= logf(fmaxf(ptgt, 1e-10f));
            float gscale = inv / T; // dlogits for the mean loss over tokens
            for (i32 j = 0; j < vocab_size; j++)
                dp[j] = dp[j] * gscale - (j == tgt ? 1.0f / T : 0.0f);
        }
        return loss / T;
    }

    void clip_grads(float max_norm) {
        float norm = 0;
        for (auto* g : grads())
            for (i32 i = 0; i < g->size(); i++)
                norm += g->data[i] * g->data[i];
        norm = sqrtf(norm);
        if (norm > max_norm) {
            float scale = max_norm / norm;
            for (auto* g : grads())
                for (i32 i = 0; i < g->size(); i++)
                    g->data[i] *= scale;
        }
    }

    std::vector<Tensor*> params() {
        auto p = std::vector<Tensor*>{&embed.w, &output.w, &output.b};
        for (auto& b : blocks)
            for (auto* bp : b.params()) p.push_back(bp);
        return p;
    }
    std::vector<Tensor*> grads() {
        auto g = std::vector<Tensor*>{&embed.gw, &output.gw, &output.gb};
        for (auto& b : blocks)
            for (auto* bg : b.grads()) g.push_back(bg);
        return g;
    }

    int sample(float* logits, int n, float temp, int top_k) {
        float maxv = logits[0];
        for (int j = 1; j < n; j++)
            if (logits[j] > maxv) maxv = logits[j];
        float sum = 0;
        std::vector<float> probs((size_t)n); // vocab-dependent: no fixed-size overflow
        for (int j = 0; j < n; j++) {
            probs[j] = expf((logits[j] - maxv) / fmaxf(temp, 0.01f));
            sum += probs[j];
        }
        float filtered = keep_top_k(probs, top_k);
        if (filtered > 0) sum = filtered;
        float r = (float)rand() / RAND_MAX;
        float cum = 0;
        for (int j = 0; j < n; j++) {
            cum += probs[j] / sum;
            if (r < cum) return j;
        }
        return n - 1;
    }

    std::string generate(const std::vector<int>& seed, int max_new_tokens,
                         float temp, int top_k, const Tokenizer& tok) {
        std::vector<int> ids = seed;
        for (int step = 0; step < max_new_tokens; step++) {
            int ctx = std::min(max_len, (int)ids.size());
            int start = (int)ids.size() - ctx;
            Tensor input(1, ctx);
            for (int i = 0; i < ctx; i++) input.data[i] = (float)ids[start + i];
            Tensor logits = forward(input);
            int last = ctx - 1;
            int chosen = sample(&logits.data[last * vocab_size], vocab_size, temp, top_k);
            ids.push_back(chosen);
            if (chosen == Tokenizer::EOS) break;
        }
        return tok.decode(ids);
    }
};

// ---------------------------------------------------------------------------
// Self-contained GPT checkpoint: magic/version + config + tokenizer + weights.
// Reuses the validated block I/O from Serialize::detail so a corrupt file is
// rejected without ever touching a live model.
// ---------------------------------------------------------------------------
static constexpr i32 GPT_MAGIC = 0x53484447; // "SHDG"
// Fixed-width, little-endian (x86/x64) integer fields; floats are IEEE-754
// binary32. v1 header: magic, version, vocab, dim, blocks, ctx (24 bytes).
// v2 inserts a u32 CRC-32 at offset 8; config therefore starts at offset 12.
static constexpr i32 GPT_VERSION_V1 = 1;
static constexpr i32 GPT_VERSION_V2 = 2;
static constexpr i32 GPT_VERSION = GPT_VERSION_V2; // new saves
static constexpr size_t GPT_CRC_OFFSET = 8;
// 50M floats = 200 MB. A hostile config must be rejected before any GPT object
// is built. Weights, gradients and Dense transpose caches all count.
static constexpr int64_t GPT_MAX_MODEL_ELEMS = 50000000;
// Checkpoint load keeps the staged `weights` vector alive while infer_gpt()
// constructs the model (params extra floats), then moves buffers in.
// All GPT_MAX_* limits are counts of float ELEMENTS (4 bytes each on x64),
// not bytes: 50M elements = 200 MB of model, 75M = 300 MB including staging.
static constexpr int64_t GPT_MAX_LOAD_ELEMS = 75000000;
// Peak live activation tensors across forward+backward of every block.
static constexpr int64_t GPT_MAX_ACTIVATION_ELEMS = 20000000;

struct GPTConfig { i32 vocab, dim, blocks, max_len; };
struct ExpectedTensor { i32 rows, cols; };

// Derive the exact parameter layout of GPT::params() from the config alone.
// Order: embed.w, output.w, output.b, then 12 tensors per TransformerBlock.
static std::vector<ExpectedTensor> gpt_expected_shapes(const GPTConfig& c) {
    std::vector<ExpectedTensor> s;
    s.reserve((size_t)(3 + 12 * (int64_t)c.blocks));
    s.push_back({ c.vocab, c.dim });   // embed.w
    s.push_back({ c.dim, c.vocab });   // output.w
    s.push_back({ 1, c.vocab });       // output.b
    const i32 ffn = c.dim * 4;
    for (i32 b = 0; b < c.blocks; b++) {
        for (int k = 0; k < 4; k++) s.push_back({ c.dim, c.dim }); // Wq, Wk, Wv, Wo
        s.push_back({ 1, c.dim });     s.push_back({ 1, c.dim });     // ln1.gamma/beta
        s.push_back({ c.dim, ffn });   s.push_back({ 1, ffn });       // ff1.w/b
        s.push_back({ ffn, c.dim });   s.push_back({ 1, c.dim });     // ff2.w/b
        s.push_back({ 1, c.dim });     s.push_back({ 1, c.dim });     // ln2.gamma/beta
    }
    return s;
}

// ---------------------------------------------------------------------------
// Tensor lifetime ledger. Coefficients below are DERIVED from the layer code,
// not guessed. T = ctx, d = dim (4 heads => head_dim = d/4), v = vocab.
//
// PERSISTENT activation caches owned by ONE TransformerBlock after forward
// (all alive simultaneously through forward AND backward):
//   SelfAttention: last_x/-q/-k/-v/-out (5 Td) + head_out (4 x T*d/4 = Td)
//                    -> 6 Td
//                  last_scores (T^2) + last_softmax (T^2)
//                    + head_softmax (4 x T^2)          -> 6 T^2
//   LayerNorm x2  : last_x (2 Td) + last_mean/last_var (4 T) -> 2 Td + 4 T
//   Dense ff1/ff2 : last_input/last_output (Td+4Td+4Td+Td)   -> 10 Td
//   Act (GELU)    : last_in (T*4d)                            -> 4 Td
//   Block fields  : last_x/-attn_out/-x1/-y/-ffn_out/-x2      -> 6 Td
//   ------------------------------------------------------------------------
//   persistent subtotal P = 28 Td + 6 T^2 + 4 T
//
// FORWARD transients (moment -> extra live buffers, non-overlapping maxima):
//   attention head : sm (T^2) + Kc/Vt (0.5Td) + head_out temp (0.25Td)
//                    => T^2 + Td
//   ff1 matmul/add : r1 (4Td) + r2 (4Td); copy into last_output reuses its
//                    persistent buffer => 8 Td
//   FFN locals     : ff1_out (4Td) + ff1_act (4Td) + ff2 r1/r2 (2Td) => 10 Td
//   F = max(T^2 + Td, 10 Td)
//
// BACKWARD transients:
//   18 Td of named locals live until TransformerBlock::backward returns, but
//   at the nested attn.backward peak only 16 Td of them exist yet, plus
//   dO/dQ/dK/dV/dx (5 Td) and per-head dA/dScores (2 T^2):
//   W = 21 Td + 2 T^2
//
// GLOBAL peak -- the crucial correction: transients of different blocks NEVER
// overlap (each block call returns before the next begins), while persistent
// caches of ALL blocks stay live. Therefore:
//
//   total = B*P + max(F, W) + G
//
// NOT B*(P+max(F,W)) + G, which needlessly multiplied the transient by B.
//
// GPT level G (outside blocks): output Dense keeps last_input (Td) +
//   last_output (Tv); forward adds h (Td) + pe (Td) + logits (Tv) and
//   output.forward matmul/add (3Tv); training keeps logits+dlogits (2Tv)
//   + output.backward d (Tv) + dh (Td). Conservative: 4 Tv + 3 Td.
// ---------------------------------------------------------------------------

// All budget operands are non-negative; reject negatives and overflow before
// any comparison or allocation.
static bool ck_mul(int64_t a, int64_t b, int64_t& out) {
    if (a < 0 || b < 0) return false;
    if (a != 0 && b > INT64_MAX / a) return false;
    out = a * b;
    return true;
}
static bool ck_add(int64_t a, int64_t b, int64_t& out) {
    if (a < 0 || b < 0) return false;
    if (a > INT64_MAX - b) return false;
    out = a + b;
    return true;
}

// Separate phase peaks for ONE block, so the ledger can be audited term by
// term. Each helper returns false on int64 overflow.
static bool block_persistent_act(int64_t t, int64_t d, int64_t& out) {
    // 28 Td + 6 T^2 + 4 T
    int64_t t2, td, a, b, c;
    if (!ck_mul(t, t, t2) || !ck_mul(t, d, td)) return false;
    if (!ck_mul(28, td, a) || !ck_mul(6, t2, b) || !ck_mul(4, t, c)) return false;
    if (!ck_add(a, b, a) || !ck_add(a, c, out)) return false;
    return true;
}
static bool block_forward_transient(int64_t t, int64_t d, int64_t& out) {
    // max(attention head loop T^2+Td, FFN locals/matmul 10 Td).
    // These windows do not overlap, so take the larger, never the sum.
    int64_t t2, td, attn, ffn;
    if (!ck_mul(t, t, t2) || !ck_mul(t, d, td)) return false;
    if (!ck_add(t2, td, attn) || !ck_mul(10, td, ffn)) return false;
    out = ffn > attn ? ffn : attn;
    return true;
}
static bool block_backward_transient(int64_t t, int64_t d, int64_t& out) {
    // Nested attn.backward peak: 16 Td outer locals + 5 Td attention internals
    // + 2 T^2 (dA, dScores). Later locals (d_from_attn, dx_total) do not exist
    // yet at this point, so 18 Td is not the right multiplier here.
    int64_t t2, td, a, b;
    if (!ck_mul(t, t, t2) || !ck_mul(t, d, td)) return false;
    if (!ck_mul(21, td, a) || !ck_mul(2, t2, b)) return false;
    if (!ck_add(a, b, out)) return false;
    return true;
}

static bool gpt_activation_estimate(const GPTConfig& c, int64_t& out) {
    const int64_t t = c.max_len, d = c.dim, b = c.blocks, v = c.vocab;
    int64_t persistent, fwd, bwd, transient, blocks_all, with_transient;
    int64_t tv, td, m4v, m3d, gpt_lvl;
    if (!block_persistent_act(t, d, persistent)) return false;
    if (!block_forward_transient(t, d, fwd)) return false;
    if (!block_backward_transient(t, d, bwd)) return false;
    // GLOBAL lifetime: every block's persistent caches stay live, but only
    // ONE block's transient peak exists at a time (forward/backward are
    // separate phases and each block call returns before the next begins).
    transient = bwd > fwd ? bwd : fwd;
    if (!ck_mul(b, persistent, blocks_all)) return false;
    if (!ck_add(blocks_all, transient, with_transient)) return false;
    // GPT-level: 4 Tv + 3 Td (output Dense caches + logits/dlogits/pe/h).
    if (!ck_mul(t, v, tv) || !ck_mul(t, d, td)) return false;
    if (!ck_mul(4, tv, m4v) || !ck_mul(3, td, m3d)) return false;
    if (!ck_add(m4v, m3d, gpt_lvl)) return false;
    if (!ck_add(with_transient, gpt_lvl, out)) return false;
    return true;
}

// Budget the whole model before any GPT object is constructed. Returns the
// expected parameter count so the reader can enforce it per tensor.
static bool gpt_config_budget(const GPTConfig& c, int64_t& out_params) {
    if (c.dim % 4 != 0) return false; // SelfAttention heads=4 and FFN need this
    if (c.dim > std::numeric_limits<i32>::max() / 4) return false;

    const int64_t d = c.dim, v = c.vocab, b = c.blocks;
    int64_t d2, vd, head, t9, blk, blk_all, params;
    int64_t cache_blk, cache_vd, caches, p2, model, load, act;
    if (!ck_mul(d, d, d2) || !ck_mul(v, d, vd)) return false;
    // params = 2*v*d + v + b*(12*d^2 + 9*d)
    if (!ck_mul(2, vd, head) || !ck_add(head, v, head)) return false;
    if (!ck_mul(12, d2, blk) || !ck_mul(9, d, t9) || !ck_add(blk, t9, blk)) return false;
    if (!ck_mul(b, blk, blk_all) || !ck_add(head, blk_all, params)) return false;
    // Dense transpose caches: ff1.wT + ff2.wT per block, output.wT once.
    if (!ck_mul(8, d2, cache_blk) || !ck_mul(b, cache_blk, cache_blk)) return false;
    if (!ck_mul(v, d, cache_vd) || !ck_add(cache_blk, cache_vd, caches)) return false;
    // Construction steady state: params + grads + caches.
    if (!ck_mul(2, params, p2) || !ck_add(p2, caches, model)) return false;
    // Checkpoint load peak: staged weights vector coexists with the new model.
    if (!ck_add(model, params, load)) return false;
    if (!gpt_activation_estimate(c, act)) return false;
    if (params <= 0 || model <= 0 || load <= 0 || act <= 0) return false;
    if (model > GPT_MAX_MODEL_ELEMS) return false;
    if (load > GPT_MAX_LOAD_ELEMS) return false;
    if (act > GPT_MAX_ACTIVATION_ELEMS) return false;
    out_params = params;
    return true;
}

// ---- checkpoint integrity: CRC-32 (IEEE 802.3, reflected poly 0xEDB88320) ----
// Detects accidental corruption, NOT deliberate tampering. Init 0xFFFFFFFF,
// final XOR 0xFFFFFFFF. The table is generated once, no external dependency.
static uint32_t gpt_crc32_update(uint32_t crc, const void* data, size_t n) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    const unsigned char* p = (const unsigned char*)data;
    for (size_t i = 0; i < n; i++) crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return crc;
}
static bool gpt_read_u32(FILE* f, uint32_t& v) {
    return Serialize::detail::read_exact(f, &v, sizeof(v));
}
static bool gpt_write_u32(FILE* f, uint32_t v) {
    return Serialize::detail::write_exact(f, &v, sizeof(v));
}

// Stream the whole file with a FIXED buffer, CRC-ing every byte except the
// 4-byte checksum field at GPT_CRC_OFFSET. No allocation depends on file size,
// so a hostile length cannot exhaust RAM. Leaves the file at EOF.
static bool gpt_compute_file_crc(FILE* f, uint32_t& out) {
    if (fseek(f, 0, SEEK_SET) != 0) return false;
    uint32_t crc = 0xFFFFFFFFu;
    unsigned char buf[65536];
    size_t pos = 0;
    for (;;) {
        size_t n = fread(buf, 1, sizeof(buf), f);
        if (n == 0) break;
        const size_t end = pos + n;
        if (pos < GPT_CRC_OFFSET) {
            size_t e = std::min(end, GPT_CRC_OFFSET);
            crc = gpt_crc32_update(crc, buf, e - pos);
        }
        if (end > GPT_CRC_OFFSET + 4) {
            size_t s = std::max(pos, GPT_CRC_OFFSET + 4);
            crc = gpt_crc32_update(crc, buf + (s - pos), end - s);
        }
        pos = end;
        if (n < sizeof(buf)) break;
    }
    if (ferror(f)) return false;
    out = crc ^ 0xFFFFFFFFu;
    return true;
}

static bool atomic_replace(const std::string& tmp, const std::string& dst) {
#ifdef _WIN32
    return MoveFileExA(tmp.c_str(), dst.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(tmp.c_str(), dst.c_str()) == 0;
#endif
}

bool save_gpt(const std::string& path, GPT& gpt, const Tokenizer& tok) {
    using namespace Serialize::detail;
    // Write to a sibling temp file first. The previous good checkpoint stays
    // untouched until the whole file, INCLUDING its checksum, is on disk and
    // the atomic rename succeeds.
    const std::string tmp = path + ".tmp";
    // Read/write mode: after writing the payload we read it back to compute
    // the checksum before patching the CRC field.
    FILE* f = fopen(tmp.c_str(), "w+b");
    if (!f) return false;
    bool ok = true;
    auto wi = [&](i32 v) { if (ok) ok = write_i32(f, v); };

    wi(GPT_MAGIC);
    wi(GPT_VERSION);
    if (ok) ok = gpt_write_u32(f, 0); // checksum placeholder, patched below
    wi(gpt.vocab_size);
    wi(gpt.dim);
    wi(gpt.num_blocks);
    wi(gpt.max_len);
    if (ok) ok = tok.write(f); // tokenizer vocabulary

    auto params = gpt.params();
    wi((i32)params.size());
    for (auto* p : params) {
        if (!ok) break;
        ok = write_tensor(f, *p);
    }

    if (ok) {
        uint32_t crc = 0;
        if (fflush(f) != 0 || !gpt_compute_file_crc(f, crc)) ok = false;
        if (ok && fseek(f, (long)GPT_CRC_OFFSET, SEEK_SET) != 0) ok = false;
        if (ok) ok = gpt_write_u32(f, crc);
    }
    if (fclose(f) != 0) ok = false;
    if (!ok || !atomic_replace(tmp, path)) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

bool read_gpt_checkpoint(const std::string& path, GPTConfig& cfg,
                         Tokenizer& tok, std::vector<Tensor>& weights) {
    using namespace Serialize::detail;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    // Nothing is written to the caller until EVERY stage succeeds.
    GPTConfig parsed{};
    Tokenizer parsed_tok;
    std::vector<Tensor> staged;
    bool ok = false;
    do {
        // 1-4: open, header, version.
        i32 magic, ver;
        if (!read_i32(f, magic) || !read_i32(f, ver)) break;
        if (magic != GPT_MAGIC) break;
        if (ver != GPT_VERSION_V1 && ver != GPT_VERSION_V2) break;

        // 5-6: v2 verifies the checksum BEFORE any payload is parsed. v1 has
        // no checksum field and keeps its original layout (config at offset 8).
        if (ver == GPT_VERSION_V2) {
            uint32_t stored_crc = 0, computed_crc = 0;
            if (!gpt_read_u32(f, stored_crc)) break;
            if (!gpt_compute_file_crc(f, computed_crc) || computed_crc != stored_crc) break;
            if (fseek(f, (long)(GPT_CRC_OFFSET + 4), SEEK_SET) != 0) break;
        }

        // 7: config, tokenizer, tensor shapes/count, all before staging data.
        if (!read_i32(f, parsed.vocab) || !read_i32(f, parsed.dim) ||
            !read_i32(f, parsed.blocks) || !read_i32(f, parsed.max_len)) break;
        if (parsed.vocab < Tokenizer::SPECIAL || parsed.vocab > 1000000) break;
        if (parsed.dim < 1 || parsed.dim > 65536) break;
        if (parsed.blocks < 1 || parsed.blocks > 1024) break;
        if (parsed.max_len < 1 || parsed.max_len > 1000000) break;

        // Reject the whole architecture before infer_gpt() constructs it:
        // params + gradients + caches + activations, not just file size.
        int64_t expected_params = 0;
        if (!gpt_config_budget(parsed, expected_params)) break;
        const auto expected = gpt_expected_shapes(parsed);

        if (!parsed_tok.read(f) || parsed_tok.vocab_size != parsed.vocab) break;

        i32 nt;
        if (!read_i32(f, nt) || nt != (i32)expected.size()) break;
        staged.reserve(expected.size());
        int64_t total_elems = 0;
        for (size_t i = 0; i < expected.size(); i++) {
            Tensor t;
            // Shape and the *remaining* parameter budget are both checked
            // inside read_tensor(), before it allocates the data buffer.
            const size_t remaining = (size_t)(expected_params - total_elems);
            if (!read_tensor(f, t, expected[i].rows, expected[i].cols, remaining)) break;
            total_elems += (int64_t)t.rows * t.cols;
            staged.push_back(std::move(t));
        }
        // 8: whole-file validation (no truncation and no trailing bytes).
        if (staged.size() != expected.size() || total_elems != expected_params
            || !at_eof(f)) break;
        ok = true;
    } while (false);
    fclose(f);

    // 9: commit only after every stage passed.
    if (!ok) return false;
    cfg = parsed;
    tok = std::move(parsed_tok);
    weights = std::move(staged);
    return true;
}

void train_gpt(const std::string& data_path, const std::string& model_path) {
    printf("=== GPT-2 Mini ===\n");
    srand(42);
    setbuf(stdout, NULL);

    // Load text
    std::ifstream file(data_path);
    if (!file.is_open()) { printf("ERROR: can't open %s\n", data_path.c_str()); return; }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Use first 100KB for speed
    if (text.size() > 100000) text = text.substr(0, 100000);
    printf("  loaded %d chars (%.1fKB)\n", (int)text.size(), text.size() / 1000.0f);

    // Build tokenizer
    Tokenizer tok;
    tok.build(text, 2000);
    printf("  vocab: %d tokens\n", tok.vocab_size);

    // Encode entire dataset
    auto data = tok.encode(text);
    printf("  encoded: %d tokens\n", (int)data.size());

    // Model config
    int dim = 64, num_blocks = 3, max_len = 32;
    GPT gpt(tok.vocab_size, dim, num_blocks, max_len);
    AdamW opt(0.001f, 0.0f);
    for (auto* p : gpt.params()) opt.add(*p);

    int total_params = 0;
    for (auto* p : gpt.params()) total_params += p->size();
    printf("  model: %d params, %d blocks, dim=%d, ctx=%d\n",
           total_params, num_blocks, dim, max_len);

    int seq_len = max_len, epochs = 12, batch_size = 8;
    int iters_since_step = 0;

    for (int ep = 0; ep < epochs; ep++) {
        auto pp = gpt.params();
        auto gp = gpt.grads();
        opt.zero_grad(gp);
        auto finish_step = [&]() {
            f32 scale = 1.0f / (f32)iters_since_step; // average accumulated micro-batches
            for (auto* g : gp)
                for (i32 k = 0; k < g->size(); k++) g->data[k] *= scale;
            gpt.clip_grads(1.0f);
            opt.step(pp, gp);
            opt.zero_grad(gp);
            iters_since_step = 0;
        };
        float tl = 0;
        int iters = 0;
        double t0 = clock();

        // Shuffle data order
        std::vector<int> indices;
        for (size_t i = 0; i + seq_len + 1 < data.size(); i += seq_len / 2)
            indices.push_back((int)i);
        for (int i = (int)indices.size() - 1; i > 0; i--)
            std::swap(indices[i], indices[rand() % (i + 1)]);

        for (int idx : indices) {
            Tensor ids(1, seq_len);
            Tensor targets(1, seq_len);
            for (int i = 0; i < seq_len; i++) {
                ids.data[i] = (float)data[idx + i];
                targets.data[i] = (float)data[idx + i + 1];
            }

            Tensor logits = gpt.forward(ids);
            Tensor dlogits;
            tl += gpt.compute_loss(logits, targets, dlogits);
            gpt.backward(dlogits, ids);
            iters++;
            iters_since_step++;

            if (iters_since_step >= batch_size) finish_step();
        }
        if (iters_since_step > 0) finish_step();

        double t1 = clock();
        float avg_loss = tl / fmaxf(1, iters);

        // Generate sample
        std::vector<int> seed = {Tokenizer::BOS};
        std::string gen = gpt.generate(seed, 40, 0.8f, 20, tok);
        printf("  ep %2d loss=%.4f [%.0fs] | %s\n",
               ep, avg_loss, (t1 - t0) / CLOCKS_PER_SEC, gen.c_str());
        fflush(stdout);
    }

    // Save a self-contained checkpoint (config + tokenizer + weights).
    if (!save_gpt(model_path, gpt, tok)) {
        printf("  ERROR: failed to save model to %s\n", model_path.c_str());
        return;
    }
    printf("  model saved to %s\n", model_path.c_str());

    printf("\n  --- final generation ---\n");
    const char* prompts[] = {"once upon a time", "the robot", "hello", nullptr};
    for (int i = 0; prompts[i]; i++) {
        auto ids = tok.encode(prompts[i]);
        std::string gen = gpt.generate(ids, 80, 0.7f, 20, tok);
        printf("  \"%s\" -> %s\n", prompts[i], gen.c_str());
    }
}

void infer_gpt(const std::string& model_path, const std::string& prompt_text) {
    printf("Loading %s ...\n", model_path.c_str());

    GPTConfig cfg;
    Tokenizer tok;
    std::vector<Tensor> weights;
    if (!read_gpt_checkpoint(model_path, cfg, tok, weights)) {
        printf("ERROR: cannot load checkpoint (missing, corrupt, or old-format file).\n");
        printf("       Train first with: shade.exe --train <data.txt> <model.bin>\n");
        return;
    }
    printf("  config: vocab=%d dim=%d blocks=%d ctx=%d\n",
           cfg.vocab, cfg.dim, cfg.blocks, cfg.max_len);

    auto gpt = std::make_unique<GPT>(cfg.vocab, cfg.dim, cfg.blocks, cfg.max_len);
    auto params = gpt->params();
    if ((i32)params.size() != (i32)weights.size()) {
        printf("ERROR: checkpoint weight count mismatch (%zu vs %zu)\n",
               params.size(), weights.size());
        return;
    }
    for (size_t i = 0; i < params.size(); i++) {
        if (params[i]->rows != weights[i].rows || params[i]->cols != weights[i].cols) {
            printf("ERROR: checkpoint weight %zu has shape %dx%d, expected %dx%d\n",
                   i, weights[i].rows, weights[i].cols, params[i]->rows, params[i]->cols);
            return;
        }
        params[i]->data = std::move(weights[i].data);
    }

    // Seed with BOS + prompt tokens. No EOS: the model should continue the text.
    std::vector<int> seed;
    seed.push_back(Tokenizer::BOS);
    for (const auto& t : tok.tokenize(prompt_text)) {
        auto it = tok.word2id.find(t);
        seed.push_back(it != tok.word2id.end() ? it->second : Tokenizer::UNK);
    }
    if (seed.size() == 1) seed.push_back(Tokenizer::UNK); // empty prompt fallback

    srand((unsigned)time(nullptr));
    std::string out = gpt->generate(seed, 120, 0.8f, 40, tok);
    printf("\n%s\n", out.c_str());
}

int main(int argc, char** argv) {
    std::string data_path = "data/tinystories_valid.txt";
    std::string model_path = "gpt_model.bin";
    std::string prompt;
    bool train = true;
    bool load_seen = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--train") {
            train = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') data_path = argv[++i];
            if (i + 1 < argc && argv[i + 1][0] != '-') model_path = argv[++i];
        } else if (a == "--load") {
            train = false;
            load_seen = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') model_path = argv[++i];
        } else if (a == "--prompt" || a == "--data" || a == "--model") {
            if (i + 1 < argc) {
                if (a == "--prompt") prompt = argv[++i];
                else if (a == "--data") data_path = argv[++i];
                else model_path = argv[++i];
            }
        } else if (!a.empty() && a[0] != '-' && load_seen && prompt.empty()) {
            prompt = a; // positional prompt, e.g. --load model.bin "hello"
        } else {
            printf("Unknown argument: %s\n"
                   "Usage: shade.exe [--train [data.txt] [model.bin] |\n"
                   "                  --load <model.bin> [--prompt] \"text\"]\n", a.c_str());
            return 1;
        }
    }

    if (train) {
        train_gpt(data_path, model_path);
    } else {
        infer_gpt(model_path, prompt);
    }
    return 0;
}
