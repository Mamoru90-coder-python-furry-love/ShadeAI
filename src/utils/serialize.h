#pragma once
#include "../core/tensor.h"
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include <mutex>

namespace Serialize {

struct ModelInfo { i32 dim, blocks, ctx; bool valid; };

// Returns true if any float element is NaN/Inf (a poisoned/exploded tensor).
inline bool has_naninf(const float* data, i32 n) {
    for (i32 i = 0; i < n; i++) {
        float v = data[i];
        if (v != v) return true;
        if (v > 1e30f || v < -1e30f) return true;   // treat as exploded/stale too
    }
    return false;
}
inline bool has_naninf_tensor(const Tensor& t) { return has_naninf(t.data.data(), (i32)t.size()); }

bool save(const std::string& path, const std::vector<Tensor*>& params) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    i32 n = (i32)params.size();
    fwrite(&n, sizeof(i32), 1, f);
    for (auto* p : params) {
        fwrite(&p->rows, sizeof(i32), 1, f);
        fwrite(&p->cols, sizeof(i32), 1, f);
        fwrite(p->data.data(), sizeof(f32), p->size(), f);
    }
    fclose(f);
    return true;
}

bool load(const std::string& path, const std::vector<Tensor*>& params) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    i32 n;
    if (fread(&n, sizeof(i32), 1, f) != 1 || n != (i32)params.size()) { fclose(f); return false; }
    for (i32 i = 0; i < n; i++) {
        i32 r, c;
        if (fread(&r, sizeof(i32), 1, f) != 1 || fread(&c, sizeof(i32), 1, f) != 1) {
            fclose(f);
            return false;
        }
        if (params[i]->rows != r || params[i]->cols != c) {
            fclose(f);
            return false;
        }
        if (fread(params[i]->data.data(), sizeof(f32), params[i]->size(), f) != (size_t)params[i]->size()) {
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}

bool save_model(const std::string& path, const std::vector<Tensor*>& params, i32 dim, i32 blocks, i32 ctx) {
    // Never persist a poisoned (NaN/Inf/exploded) model — it would permanently destroy the good one.
    for (auto* p : params) if (has_naninf_tensor(*p)) return false;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    i32 magic = 0x53484445; // "SHDE"
    i32 header[4] = { magic, dim, blocks, ctx };
    fwrite(header, sizeof(i32), 4, f);
    i32 n = (i32)params.size();
    fwrite(&n, sizeof(i32), 1, f);
    for (auto* p : params) {
        fwrite(&p->rows, sizeof(i32), 1, f);
        fwrite(&p->cols, sizeof(i32), 1, f);
        fwrite(p->data.data(), sizeof(f32), p->size(), f);
    }
    fclose(f);
    return true;
}

ModelInfo load_model(const std::string& path, const std::vector<Tensor*>& params) {
    ModelInfo info = { 0, 0, 0, false };
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return info;
    i32 header[4];
    if (fread(header, sizeof(i32), 4, f) != 4) { fclose(f); return info; }
    if (header[0] != 0x53484445) { fclose(f); return info; }
    info = { header[1], header[2], header[3], true };
    i32 n;
    if (fread(&n, sizeof(i32), 1, f) != 1 || n != (i32)params.size()) {
        fclose(f); info.valid = false; return info;
    }
    for (i32 i = 0; i < n; i++) {
        i32 r, c;
        if (fread(&r, sizeof(i32), 1, f) != 1 || fread(&c, sizeof(i32), 1, f) != 1 ||
            params[i]->rows != r || params[i]->cols != c) {
            fclose(f); info.valid = false; return info;
        }
        if (fread(params[i]->data.data(), sizeof(f32), params[i]->size(), f) != (size_t)params[i]->size()) {
            fclose(f); info.valid = false; return info;
        }
        if (has_naninf_tensor(*params[i])) { fclose(f); info.valid = false; return info; }   // poisoned checkpoint
    }
    fclose(f);
    return info;
}

ModelInfo read_model_info(const std::string& path) {
    ModelInfo info = { 0, 0, 0, false };
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return info;
    i32 header[4];
    if (fread(header, sizeof(i32), 4, f) != 4) { fclose(f); return info; }
    if (header[0] != 0x53484445) { fclose(f); return info; }
    info = { header[1], header[2], header[3], true };
    fclose(f);
    return info;
}

// ---- AdamW optimizer state (m, v moments) persistence for smooth resume ----
// Separate sidecar file so model files stay compatible. Layout mirrors params (rows/cols/data),
// one block for m's then one for v's, then the adam step counter t.
bool save_optimizer(const std::string& path,
                    const std::vector<Tensor*>& m, const std::vector<Tensor*>& v, i32 t) {
    if (m.size() != v.size()) return false;
    for (auto* p : m) if (has_naninf_tensor(*p)) return false;
    for (auto* p : v) if (has_naninf_tensor(*p)) return false;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    i32 n = (i32)m.size();
    fwrite(&n, sizeof(i32), 1, f);
    for (auto* p : m) { fwrite(&p->rows, sizeof(i32), 1, f); fwrite(&p->cols, sizeof(i32), 1, f); fwrite(p->data.data(), sizeof(f32), p->size(), f); }
    for (auto* p : v) { fwrite(&p->rows, sizeof(i32), 1, f); fwrite(&p->cols, sizeof(i32), 1, f); fwrite(p->data.data(), sizeof(f32), p->size(), f); }
    fwrite(&t, sizeof(i32), 1, f);
    fclose(f);
    return true;
}

bool load_optimizer(const std::string& path,
                    const std::vector<Tensor*>& m, const std::vector<Tensor*>& v, i32& t) {
    if (m.size() != v.size()) return false;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    i32 n;
    if (fread(&n, sizeof(i32), 1, f) != 1) { fclose(f); return false; }
    if ((i32)n != (i32)m.size()) { fclose(f); return false; }
    for (i32 i = 0; i < n; i++) {
        i32 r, c;
        if (fread(&r, sizeof(i32), 1, f) != 1 || fread(&c, sizeof(i32), 1, f) != 1) { fclose(f); return false; }
        if (m[i]->rows != r || m[i]->cols != c) { fclose(f); return false; }
        if (fread(m[i]->data.data(), sizeof(f32), m[i]->size(), f) != (size_t)m[i]->size()) { fclose(f); return false; }
        if (has_naninf_tensor(*m[i])) { fclose(f); return false; }   // poisoned optimizer state
    }
    for (i32 i = 0; i < n; i++) {
        i32 r, c;
        if (fread(&r, sizeof(i32), 1, f) != 1 || fread(&c, sizeof(i32), 1, f) != 1) { fclose(f); return false; }
        if (v[i]->rows != r || v[i]->cols != c) { fclose(f); return false; }
        if (fread(v[i]->data.data(), sizeof(f32), v[i]->size(), f) != (size_t)v[i]->size()) { fclose(f); return false; }
        if (has_naninf_tensor(*v[i])) { fclose(f); return false; }   // poisoned optimizer state
    }
    if (fread(&t, sizeof(i32), 1, f) != 1) { fclose(f); return false; }
    fclose(f);
    return true;
}

// ---- async (threaded) checkpoint writer ----
// Captures a deep copy of model + AdamW moments in the main thread (fast, ~tens of ms),
// then writes everything to disk on a detached background thread so the HDD fsync never
// blocks the training loop. A tiny mutex guards the "next snapshot" slot so saves never overlap.
struct AsyncSaver {
    struct Buf { std::vector<Tensor> model; std::vector<Tensor> m, v; i32 dim=0, blocks=0, ctx=0, t=0; };
    std::mutex mu;
    std::thread last;
    Buf pending;
    bool has_pending = false;

    void queue_model(const std::vector<Tensor*>& params, i32 dim, i32 blocks, i32 ctx) {
        std::lock_guard<std::mutex> lk(mu);
        pending.model.clear();
        for (auto* p : params) {
            Tensor cp(p->rows, p->cols, p->data); // deep copy
            pending.model.push_back(std::move(cp));
        }
        pending.dim = dim; pending.blocks = blocks; pending.ctx = ctx;
        has_pending = true;
    }

    void queue_opt(const std::vector<Tensor*>& om, const std::vector<Tensor*>& ov, i32 t) {
        std::lock_guard<std::mutex> lk(mu);
        pending.m.clear(); pending.v.clear();
        for (auto* p : om) { Tensor cp(p->rows, p->cols, p->data); pending.m.push_back(std::move(cp)); }
        for (auto* p : ov) { Tensor cp(p->rows, p->cols, p->data); pending.v.push_back(std::move(cp)); }
        pending.t = t;
    }

    // flush any still-running background writer before queueing the next one
    void join_prev() {
        std::lock_guard<std::mutex> lk(mu);
        if (last.joinable()) last.join();
    }

    // spawn writer for the current pending snapshot (swallows it) on a background thread
    void spawn() {
        std::lock_guard<std::mutex> lk(mu);
        if (has_pending) {
            Buf snap = std::move(pending);
            has_pending = false;
            std::string mp = "../model/gpt_model.bin", op = "../model/gpt_opt.bin";
            last = std::thread([mp, op, snap = std::move(snap)]() mutable {
                std::vector<Tensor*> ckpt;
                for (auto& t : snap.model) ckpt.push_back(&t);
                bool clean = true;
                for (auto* p : ckpt) if (has_naninf_tensor(*p)) { clean = false; break; }
                for (auto& t : snap.m) if (has_naninf_tensor(t)) { clean = false; break; }
                for (auto& t : snap.v) if (has_naninf_tensor(t)) { clean = false; break; }
                if (!clean) {
                    fprintf(stderr, "  WARNING: checkpoint contains NaN/Inf — skipping disk write (keeping last good state)\n");
                    fflush(stderr);
                    return;
                }
                Serialize::save_model(mp, ckpt, snap.dim, snap.blocks, snap.ctx);
                if (!snap.m.empty()) {
                    std::vector<Tensor*> om, ov;
                    for (auto& t : snap.m) om.push_back(&t);
                    for (auto& t : snap.v) ov.push_back(&t);
                    Serialize::save_optimizer(op, om, ov, snap.t);
                }
            });
        }
    }

    // wait for any in-flight background writer to finish (call before exiting)
    void flush() {
        std::lock_guard<std::mutex> lk(mu);
        if (last.joinable()) last.join();
    }
};

} // namespace Serialize
