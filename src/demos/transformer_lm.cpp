#include "../core/tensor.h"
#include "../core/loss.h"
#include "../layers/transformer.h"
#include "../layers/dense.h"
#include "../utils/embedding.h"
#include "../optimizer/adamw.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <stdexcept>

struct TransformerLM {
    i32 vocab_size, dim, num_blocks;
    Embedding embed;
    PositionalEncoding pos_enc;
    std::vector<TransformerBlock> blocks;
    Dense output;

    TransformerLM(i32 vs, i32 d, i32 nb)
        : vocab_size(vs), dim(d), num_blocks(nb),
          embed(vs, d), pos_enc(2048, d), output(d, vs, false) {
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

    // Mean cross-entropy over tokens; dlogits is its exact gradient. The demo
    // used to emit unscaled (sum) dlogits while printing the mean loss, so the
    // optimized and reported objectives differed (same issue fixed in CharRNN).
    float compute_loss(const Tensor& logits, const Tensor& targets, Tensor& dlogits) {
        // Same guards as CharRNN::compute_loss: a wrong logits width, a length
        // mismatch, or a non-integral / out-of-range target would otherwise
        // index past the row or divide by zero. Validate the raw float BEFORE
        // casting: float->int is undefined for NaN, infinities, and values
        // outside the int range.
        if (vocab_size <= 0)
            throw std::invalid_argument("TransformerLM::compute_loss: invalid vocab size");
        if (logits.cols != vocab_size)
            throw std::invalid_argument("TransformerLM::compute_loss: invalid logits width");
        if (logits.rows <= 0 || targets.size() != logits.rows)
            throw std::invalid_argument("TransformerLM::compute_loss: invalid sequence length");
        for (i32 i = 0; i < targets.size(); i++) {
            f32 raw = targets.data[i];
            if (!std::isfinite(raw) || raw < 0.0f ||
                raw >= (f32)vocab_size || std::trunc(raw) != raw)
                throw std::invalid_argument("TransformerLM::compute_loss: invalid target ID");
        }

        i32 T = logits.rows;
        dlogits = Tensor(T, vocab_size, 0.0f);
        float loss = 0;
        for (i32 i = 0; i < T; i++) {
            int tgt = (int)targets.data[i];
            const f32* lp = &logits.data[i * vocab_size];
            f32* dp = &dlogits.data[i * vocab_size];
            float maxv = lp[0];
            for (i32 j = 1; j < vocab_size; j++)
                if (lp[j] > maxv) maxv = lp[j];
            float sum = 0;
            for (i32 j = 0; j < vocab_size; j++) {
                float e = expf(lp[j] - maxv);
                dp[j] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            loss -= logf(fmaxf(dp[tgt] * inv, 1e-10f));
            float gscale = inv / T;
            for (i32 j = 0; j < vocab_size; j++)
                dp[j] = dp[j] * gscale - (j == tgt ? 1.0f / T : 0.0f);
        }
        return loss / T;
    }

    void clip_grads(f32 max_norm) {
        f32 norm = 0;
        for (auto* g : grads())
            for (i32 i = 0; i < g->size(); i++)
                norm += g->data[i] * g->data[i];
        norm = sqrtf(norm);
        if (norm > max_norm) {
            f32 scale = max_norm / norm;
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

    std::string generate(const std::vector<int>& seed, int n, const char* id2char) {
        std::vector<int> ids = seed;
        std::string result;
        for (size_t i = 0; i < seed.size(); i++) result += id2char[seed[i]];
        for (int step = 0; step < n; step++) {
            int ctx = std::min(128, (int)ids.size());
            int start = (int)ids.size() - ctx;
            Tensor input(1, ctx);
            for (int i = 0; i < ctx; i++) input.data[i] = (float)ids[start + i];
            Tensor logits = forward(input);
            int last = ctx - 1;
            float maxv = logits.at(last, 0);
            for (int j = 1; j < vocab_size; j++)
                if (logits.at(last, j) > maxv) maxv = logits.at(last, j);
            float sum = 0;
            float probs[256];
            for (int j = 0; j < vocab_size; j++) {
                probs[j] = expf(logits.at(last, j) - maxv);
                sum += probs[j];
            }
            float r = (float)rand() / RAND_MAX;
            float cum = 0;
            int chosen = 0;
            for (int j = 0; j < vocab_size; j++) {
                cum += probs[j] / sum;
                if (r < cum) { chosen = j; break; }
            }
            ids.push_back(chosen);
            result += id2char[chosen];
        }
        return result;
    }
};

void demo_transformer_lm() {
    printf("=== Transformer Language Model ===\n");
    srand(42);

    std::ifstream file("data/train.txt");
    if (!file.is_open()) {
        printf("ERROR: data/train.txt not found.\n");
        return;
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    printf("  loaded %d chars\n", (int)text.size());
    if (text.size() > 5000) text = text.substr(0, 5000);

    // Build vocab (char-level)
    char id2char[256];
    int char2id[256];
    for (int i = 0; i < 256; i++) char2id[i] = -1;
    std::vector<char> chars;
    for (char c : text) {
        unsigned char uc = (unsigned char)c;
        if (char2id[uc] < 0) {
            char2id[uc] = (int)chars.size();
            id2char[chars.size()] = c;
            chars.push_back(c);
        }
    }
    int vocab_size = (int)chars.size();
    printf("  vocab: %d chars\n", vocab_size);

    std::vector<int> data(text.size());
    for (size_t i = 0; i < text.size(); i++)
        data[i] = char2id[(unsigned char)text[i]];

    // Model
    int dim = 64, num_blocks = 2;
    TransformerLM lm(vocab_size, dim, num_blocks);
    AdamW opt(0.001f, 0.0f);
    for (auto* p : lm.params()) opt.add(*p);

    int seq_len = 32, epochs = 100;
    int total_params = 0;
    for (auto* p : lm.params()) total_params += p->size();
    printf("  model: %d params, %d blocks, dim=%d\n", total_params, num_blocks, dim);

    // Mini-batch training
    int batch_size = 8;          // accumulate this many iterations before step
    int iters_since_step = 0;

    for (int ep = 0; ep < epochs; ep++) {
        auto pp = lm.params();
        auto gp = lm.grads();
        opt.zero_grad(gp);
        // Each window contributes the gradient of its own mean loss; the
        // accumulated tensors hold their sum. Average before clipping so the
        // optimizer sees the gradient of the mean reported in `tl`.
        auto scale_accumulated = [&gp](f32 scale) {
            for (auto* g : gp)
                for (i32 i = 0; i < g->size(); i++)
                    g->data[i] *= scale;
        };
        float tl = 0;
        int iters = 0;

        for (size_t start = 0; start + seq_len + 1 < data.size(); start += seq_len, iters++) {
            Tensor ids(1, seq_len);
            Tensor targets(1, seq_len);
            for (int i = 0; i < seq_len; i++) {
                ids.data[i] = (float)data[start + i];
                targets.data[i] = (float)data[start + i + 1];
            }

            Tensor logits = lm.forward(ids);
            Tensor dlogits;
            float loss = lm.compute_loss(logits, targets, dlogits);
            tl += loss;
            lm.backward(dlogits, ids);
            iters_since_step++;

            if (iters_since_step >= batch_size) {
                scale_accumulated(1.0f / (f32)iters_since_step);
                lm.clip_grads(1.0f);
                opt.step(pp, gp);
                opt.zero_grad(gp);
                iters_since_step = 0;
            }
        }
        // final step: divide by the actual (possibly partial) group size, not
        // by the fixed batch_size.
        if (iters_since_step > 0) {
            scale_accumulated(1.0f / (f32)iters_since_step);
            lm.clip_grads(1.0f);
            opt.step(pp, gp);
            iters_since_step = 0;
        }

        if (ep % 5 == 0) {
            printf("  ep %d loss=%.4f\n", ep, tl / fmaxf(1, iters));
        }
    }

    printf("\n  --- generation ---\n");
    const char* seeds[] = {"the ", "hello", "once ", nullptr};
    for (int s = 0; seeds[s]; s++) {
        std::vector<int> seed;
        for (int i = 0; seeds[s][i]; i++) {
            int id = char2id[(unsigned char)seeds[s][i]];
            if (id < 0) id = 0;
            seed.push_back(id);
        }
        std::string gen = lm.generate(seed, 80, id2char);
        printf("  \"%s\" -> %s\n", seeds[s], gen.c_str());
    }
}

int main() {
    demo_transformer_lm();
    return 0;
}
