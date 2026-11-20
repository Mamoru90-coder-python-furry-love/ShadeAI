#include "../core/tensor.h"
#include "../layers/dense.h"
#include "../layers/gru.h"
#include "../utils/embedding.h"
#include "../optimizer/adamw.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

struct CharRNN {
    i32 vocab_size, embed_dim, hidden_dim;
    Embedding embed;
    GRU gru;
    Dense fc;

    CharRNN(i32 vs, i32 ed, i32 hd)
        : vocab_size(vs), embed_dim(ed), hidden_dim(hd),
          embed(vs, ed), gru(ed, hd), fc(hd, vs, false) {}

    Tensor forward(const Tensor& ids) {
        Tensor x = embed.forward(ids);
        Tensor h0(1, hidden_dim, 0.0f);
        Tensor h = gru.forward(x, h0);
        return fc.forward(h);
    }

    void backward(const Tensor& dlogits, const Tensor& ids) {
        Tensor dh = fc.backward(dlogits);
        Tensor dx = gru.backward(dh);
        embed.backward(dx, ids);
    }

    // Mean cross-entropy over the sequence. dlogits is scaled by 1/T, so the
    // gradient corresponds to the returned loss (same convention as
    // GPT::compute_loss). Callers accumulating several windows must then
    // average over the window count before clipping/stepping.
    float compute_loss(const Tensor& logits, const Tensor& targets, Tensor& dlogits) {
        i32 T = logits.rows;
        dlogits = Tensor(T, vocab_size, 0.0f);
        float loss = 0;
        for (i32 i = 0; i < T; i++) {
            int tgt = (int)targets.data[i];
            const float* lp = &logits.data[i * vocab_size];
            float maxv = lp[0];
            for (i32 j = 1; j < vocab_size; j++)
                if (lp[j] > maxv) maxv = lp[j];
            float sum = 0;
            float* dp = &dlogits.data[i * vocab_size];
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

    // Clip gradients to [-max_norm, max_norm]
    void clip_grads(float max_norm) {
        float norm = 0;
        for (auto* g : grads())
            for (int i = 0; i < g->size(); i++)
                norm += g->data[i] * g->data[i];
        norm = sqrtf(norm);
        if (norm > max_norm) {
            float scale = max_norm / norm;
            for (auto* g : grads())
                for (int i = 0; i < g->size(); i++)
                    g->data[i] *= scale;
        }
    }

    std::vector<Tensor*> params() {
        auto p = std::vector<Tensor*>{&embed.w, &fc.w, &fc.b};
        for (auto* gp : gru.params()) p.push_back(gp);
        return p;
    }
    std::vector<Tensor*> grads() {
        auto g = std::vector<Tensor*>{&embed.gw, &fc.gw, &fc.gb};
        for (auto* gp : gru.grads()) g.push_back(gp);
        return g;
    }

    std::string generate(const std::vector<int>& seed, int n, const char* id2char) {
        std::vector<int> ids = seed;
        std::string result;
        for (size_t i = 0; i < seed.size(); i++) result += id2char[seed[i]];
        for (int step = 0; step < n; step++) {
            int ctx = 16;
            int start = std::max(0, (int)ids.size() - ctx);
            int seq = (int)ids.size() - start;
            Tensor input(1, seq);
            for (int i = 0; i < seq; i++)
                input.data[i] = (float)ids[start + i];
            Tensor logits = forward(input);
            int last = seq - 1;
            float maxv = logits.at(last, 0);
            for (int j = 1; j < vocab_size; j++)
                if (logits.at(last, j) > maxv) maxv = logits.at(last, j);
            float sum = 0;
            float probs[128];
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

void demo_charrnn() {
    printf("\n=== Char-RNN (text generation) ===\n");
    srand(123);

    const char* text = "hello world this is a test of the neural network "
                       "it can learn to generate text char by char "
                       "the quick brown fox jumps over the lazy dog "
                       "machine learning is fun ";
    int text_len = (int)strlen(text);

    char id2char[128];
    int char2id[256];
    for (int i = 0; i < 256; i++) char2id[i] = -1;
    std::vector<char> chars;
    for (int i = 0; i < text_len; i++) {
        if (char2id[(int)text[i]] < 0) {
            char2id[(int)text[i]] = (int)chars.size();
            id2char[chars.size()] = text[i];
            chars.push_back(text[i]);
        }
    }
    int vocab_size = (int)chars.size();
    printf("  vocab: %d chars, text: %d chars\n", vocab_size, text_len);

    std::vector<int> data(text_len);
    for (int i = 0; i < text_len; i++) data[i] = char2id[(int)text[i]];

    CharRNN model(vocab_size, 16, 32);
    AdamW opt(0.002f, 0.0f);
    for (auto* p : model.params()) opt.add(*p);

    int seq_len = 8;
    int epochs = 300;

    for (int ep = 0; ep < epochs; ep++) {
        float total_loss = 0;
        auto pp = model.params();
        auto gp = model.grads();
        opt.zero_grad(gp);

        int iters = 0;
        for (int start = 0; start + seq_len + 1 < text_len; start += seq_len / 2, iters++) {
            Tensor ids(1, seq_len);
            Tensor targets(1, seq_len);
            for (int i = 0; i < seq_len; i++) {
                ids.data[i] = (float)data[start + i];
                targets.data[i] = (float)data[start + i + 1];
            }

            Tensor logits = model.forward(ids);
            Tensor dlogits;
            total_loss += model.compute_loss(logits, targets, dlogits);
            model.backward(dlogits, ids);
        }

        // compute_loss already scales each window to a mean over tokens;
        // average the accumulated gradients over the windows as well, so the
        // optimizer sees the gradient of the printed mean loss. Do this BEFORE
        // clipping: clipping a sum-of-tokens gradient changes the update.
        float grad_scale = 1.0f / (f32)std::max(1, iters);
        for (auto* g : gp)
            for (int i = 0; i < g->size(); i++)
                g->data[i] *= grad_scale;

        model.clip_grads(1.0f);
        opt.step(pp, gp);

        if (ep % 50 == 0) {
            std::vector<int> seed = {char2id[(int)'t'], char2id[(int)'h'], char2id[(int)'e'], char2id[(int)' ']};
            std::string gen = model.generate(seed, 50, id2char);
            printf("  ep %d loss=%.4f | %s\n", ep, total_loss / fmaxf(1, iters), gen.c_str());
        }
    }

    printf("\n  --- generation ---\n");
    const char* seeds[] = {"the ", "hello", "machine", "fox ", nullptr};
    for (int s = 0; seeds[s]; s++) {
        std::vector<int> seed;
        for (int i = 0; seeds[s][i]; i++) seed.push_back(char2id[(int)seeds[s][i]]);
        std::string gen = model.generate(seed, 80, id2char);
        printf("  \"%s\" -> %s\n", seeds[s], gen.c_str());
    }
}
