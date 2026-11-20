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
#include <memory>

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
        if (top_k > 0 && top_k < n) {
            float threshold = 0;
            std::vector<float> sorted(probs);
            std::sort(sorted.begin(), sorted.end(), std::greater<float>());
            threshold = sorted[std::min(top_k, n - 1)];
            float filtered_sum = 0;
            for (int j = 0; j < n; j++) {
                if (probs[j] < threshold) probs[j] = 0;
                filtered_sum += probs[j];
            }
            if (filtered_sum > 0) { sum = filtered_sum; }
        }
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
static constexpr i32 GPT_VERSION = 1;

struct GPTConfig { i32 vocab, dim, blocks, max_len; };

bool save_gpt(const std::string& path, GPT& gpt, const Tokenizer& tok) {
    using namespace Serialize::detail;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = true;
    auto wi = [&](i32 v) { if (ok) ok = write_i32(f, v); };

    wi(GPT_MAGIC);
    wi(GPT_VERSION);
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
    if (fclose(f) != 0) ok = false;
    return ok;
}

bool read_gpt_checkpoint(const std::string& path, GPTConfig& cfg,
                         Tokenizer& tok, std::vector<Tensor>& weights) {
    using namespace Serialize::detail;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = false;
    do {
        i32 magic, ver;
        if (!read_i32(f, magic) || !read_i32(f, ver)) break;
        if (magic != GPT_MAGIC || ver != GPT_VERSION) break;
        if (!read_i32(f, cfg.vocab) || !read_i32(f, cfg.dim) ||
            !read_i32(f, cfg.blocks) || !read_i32(f, cfg.max_len)) break;

        // Reject absurd values before they turn into allocations or loops.
        if (cfg.vocab < Tokenizer::SPECIAL || cfg.vocab > 1000000) break;
        if (cfg.dim < 1 || cfg.dim > 65536) break;
        if (cfg.blocks < 1 || cfg.blocks > 1024) break;
        if (cfg.max_len < 1 || cfg.max_len > 1000000) break;

        if (!tok.read(f) || tok.vocab_size != cfg.vocab) break;

        i32 nt;
        if (!read_i32(f, nt) || nt < 1 || nt > 10000) break;
        weights.clear();
        weights.reserve((size_t)nt);
        for (i32 i = 0; i < nt; i++) {
            Tensor t;
            if (!read_tensor(f, t)) break;
            weights.push_back(std::move(t));
        }
        if ((i32)weights.size() != nt || !at_eof(f)) break;
        ok = true;
    } while (false);
    fclose(f);
    return ok;
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
