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
            Tensor dlogits(seq_len, vocab_size, 0.0f);
            float loss = 0;
            for (int i = 0; i < seq_len; i++) {
                int tgt = (int)targets.data[i];
                float maxv = logits.at(i, 0);
                for (int j = 1; j < vocab_size; j++)
                    if (logits.at(i, j) > maxv) maxv = logits.at(i, j);
                float sum = 0;
                for (int j = 0; j < vocab_size; j++) {
                    float e = expf(logits.at(i, j) - maxv);
                    dlogits.at(i, j) = e / 1.0f; // store softmax output scaled
                    sum += e;
                }
                for (int j = 0; j < vocab_size; j++) {
                    float p = expf(logits.at(i, j) - maxv) / sum;
                    dlogits.at(i, j) = p - (j == tgt ? 1.0f : 0.0f);
                }
                float p_correct = expf(logits.at(i, tgt) - maxv) / sum;
                loss -= logf(fmaxf(p_correct, 1e-10f));
            }
            total_loss += loss / seq_len;
            model.backward(dlogits, ids);
        }

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
