#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include "../core/tensor.h"
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <fstream>
#include <thread>
#include <mutex>
#include <utility>

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

namespace detail {

inline bool valid_shape(i32 r, i32 c) {
    if (r < 0 || c < 0) return false;
    return r == 0 || c <= std::numeric_limits<i32>::max() / r;
}

inline bool read_exact(FILE* f, void* dst, size_t bytes) {
    return bytes == 0 || fread(dst, 1, bytes, f) == bytes;
}
inline bool write_exact(FILE* f, const void* src, size_t bytes) {
    return bytes == 0 || fwrite(src, 1, bytes, f) == bytes;
}

inline bool read_i32(FILE* f, i32& v) { return read_exact(f, &v, sizeof(v)); }
inline bool write_i32(FILE* f, i32 v) { return write_exact(f, &v, sizeof(v)); }

// Read one tensor block into a temporary. Nothing touches the model tensors.
// max_elems is checked before allocation so a corrupt shape cannot exhaust RAM.
inline bool read_tensor(FILE* f, Tensor& out, i32 expected_rows = -1, i32 expected_cols = -1,
                        size_t max_elems = std::numeric_limits<size_t>::max()) {
    i32 r, c;
    if (!read_i32(f, r) || !read_i32(f, c)) return false;
    if (!valid_shape(r, c)) return false;
    if (expected_rows >= 0 && (r != expected_rows || c != expected_cols)) return false;

    const size_t n = (size_t)r * (size_t)c;
    if (n > max_elems) return false;
    std::vector<f32> d(n);
    if (n && !read_exact(f, d.data(), n * sizeof(f32))) return false;
    if (has_naninf(d.data(), (i32)n)) return false;
    out = Tensor(r, c, std::move(d));
    return true;
}

inline bool write_tensor(FILE* f, const Tensor& t) {
    return write_i32(f, t.rows) && write_i32(f, t.cols)
        && write_exact(f, t.data.data(), (size_t)t.size() * sizeof(f32));
}

// Reject both truncation and trailing garbage after the expected payload.
inline bool at_eof(FILE* f) { return fgetc(f) == EOF && !ferror(f); }

// Atomic publish: write a sibling temp file, then replace the destination in
// one operation. The previous good file stays intact until the new payload is
// complete, so a crash/interruption cannot leave a truncated checkpoint.
inline bool atomic_replace(const std::string& tmp, const std::string& dst) {
#ifdef _WIN32
    return MoveFileExA(tmp.c_str(), dst.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(tmp.c_str(), dst.c_str()) == 0;
#endif
}

// writer returns the payload status. On any failure the temp file is removed
// and the destination is never touched.
template <typename Writer>
inline bool write_atomic(const std::string& path, Writer writer) {
    const std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "w+b");
    if (!f) return false;
    bool ok = writer(f);
    if (ok && fflush(f) != 0) ok = false;
    if (fclose(f) != 0) ok = false;
    if (!ok || !atomic_replace(tmp, path)) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

// Vector move assignment is noexcept for the default allocator, so this cannot
// partially update the model after every block has been validated.
inline void commit(std::vector<Tensor>& staged, const std::vector<Tensor*>& params) {
    for (size_t i = 0; i < params.size(); i++) params[i]->data = std::move(staged[i].data);
}

} // namespace detail

bool save(const std::string& path, const std::vector<Tensor*>& params) {
    return detail::write_atomic(path, [&](FILE* f) {
        bool ok = detail::write_i32(f, (i32)params.size());
        for (auto* p : params) {
            if (!ok) break;
            ok = detail::write_tensor(f, *p);
        }
        return ok;
    });
}

// Transactional: on any read/shape/NaN failure the target tensors are untouched.
bool load(const std::string& path, const std::vector<Tensor*>& params) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    bool ok = false;
    do {
        i32 n;
        if (!detail::read_i32(f, n) || n < 0 || n != (i32)params.size()) break;

        std::vector<Tensor> staged;
        staged.reserve(n);
        for (i32 i = 0; i < n; i++) {
            Tensor t;
            if (!detail::read_tensor(f, t, params[i]->rows, params[i]->cols)) break;
            staged.push_back(std::move(t));
        }
        if ((i32)staged.size() != n || !detail::at_eof(f)) break;

        detail::commit(staged, params);
        ok = true;
    } while (false);

    fclose(f);
    return ok;
}

bool save_model(const std::string& path, const std::vector<Tensor*>& params, i32 dim, i32 blocks, i32 ctx) {
    // Never persist a poisoned (NaN/Inf/exploded) model — it would permanently destroy the good one.
    if (dim <= 0 || blocks <= 0 || ctx <= 0) return false;
    for (auto* p : params) if (has_naninf_tensor(*p)) return false;

    return detail::write_atomic(path, [&](FILE* f) {
        const i32 header[4] = { 0x53484445, dim, blocks, ctx }; // "SHDE"
        bool ok = detail::write_exact(f, header, sizeof(header))
               && detail::write_i32(f, (i32)params.size());
        for (auto* p : params) {
            if (!ok) break;
            ok = detail::write_tensor(f, *p);
        }
        return ok;
    });
}

// Transactional: `info.valid` is false and no weights change unless the whole
// file (header, count, every block, EOF) validates.
ModelInfo load_model(const std::string& path, const std::vector<Tensor*>& params) {
    const ModelInfo bad = { 0, 0, 0, false };
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return bad;

    ModelInfo info = bad;
    do {
        i32 header[4];
        if (!detail::read_exact(f, header, sizeof(header)) || header[0] != 0x53484445) break;
        if (header[1] <= 0 || header[2] <= 0 || header[3] <= 0) break;

        i32 n;
        if (!detail::read_i32(f, n) || n < 0 || n != (i32)params.size()) break;

        std::vector<Tensor> staged;
        staged.reserve(n);
        for (i32 i = 0; i < n; i++) {
            Tensor t;
            if (!detail::read_tensor(f, t, params[i]->rows, params[i]->cols)) break;
            staged.push_back(std::move(t));
        }
        if ((i32)staged.size() != n || !detail::at_eof(f)) break;

        detail::commit(staged, params);
        info = { header[1], header[2], header[3], true };
    } while (false);

    fclose(f);
    return info;
}

ModelInfo read_model_info(const std::string& path) {
    const ModelInfo bad = { 0, 0, 0, false };
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return bad;
    i32 header[4];
    if (!detail::read_exact(f, header, sizeof(header)) || header[0] != 0x53484445) {
        fclose(f);
        return bad;
    }
    fclose(f);
    if (header[1] <= 0 || header[2] <= 0 || header[3] <= 0) return bad;
    return { header[1], header[2], header[3], true };
}

// ---- AdamW optimizer state (m, v moments) persistence for smooth resume ----
// Separate sidecar file so model files stay compatible. Layout mirrors params (rows/cols/data),
// one block for m's then one for v's, then the adam step counter t.
bool save_optimizer(const std::string& path,
                    const std::vector<Tensor*>& m, const std::vector<Tensor*>& v, i32 t) {
    if (m.size() != v.size() || t < 0) return false;
    for (auto* p : m) if (has_naninf_tensor(*p)) return false;
    for (auto* p : v) if (has_naninf_tensor(*p)) return false;

    return detail::write_atomic(path, [&](FILE* f) {
        bool ok = detail::write_i32(f, (i32)m.size());
        for (auto* p : m) { if (!ok) break; ok = detail::write_tensor(f, *p); }
        for (auto* p : v) { if (!ok) break; ok = detail::write_tensor(f, *p); }
        if (ok) ok = detail::write_i32(f, t);
        return ok;
    });
}

// Transactional: m, v and t are committed only after every block and EOF validate.
bool load_optimizer(const std::string& path,
                    const std::vector<Tensor*>& m, const std::vector<Tensor*>& v, i32& t) {
    if (m.size() != v.size()) return false;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    bool ok = false;
    do {
        i32 n;
        if (!detail::read_i32(f, n) || n < 0 || n != (i32)m.size()) break;

        std::vector<Tensor> sm, sv;
        sm.reserve(n); sv.reserve(n);
        bool blocks_ok = true;
        for (i32 i = 0; i < n && blocks_ok; i++) {
            Tensor tmp;
            blocks_ok = detail::read_tensor(f, tmp, m[i]->rows, m[i]->cols);
            if (blocks_ok) sm.push_back(std::move(tmp));
        }
        for (i32 i = 0; i < n && blocks_ok; i++) {
            Tensor tmp;
            blocks_ok = detail::read_tensor(f, tmp, v[i]->rows, v[i]->cols);
            if (blocks_ok) sv.push_back(std::move(tmp));
        }
        if (!blocks_ok) break;

        i32 new_t;
        if (!detail::read_i32(f, new_t) || new_t < 0 || !detail::at_eof(f)) break;

        detail::commit(sm, m);
        detail::commit(sv, v);
        t = new_t;
        ok = true;
    } while (false);

    fclose(f);
    return ok;
}

// ---- async (threaded) checkpoint writer ----
// NOTE: the files written here use the legacy SHDE layout from save_model /
// save_optimizer. They are NOT the self-contained v2 GPT checkpoint written by
// save_gpt() (SHDG magic + version + tokenizer + CRC). Do not point these paths
// at a GPT checkpoint unless its loader understands this format.
// Paths are configurable so callers (and tests) do not depend on a fixed ../model dir.
struct AsyncSaver {
    struct Buf { std::vector<Tensor> model; std::vector<Tensor> m, v; i32 dim=0, blocks=0, ctx=0, t=0; };
    const std::string model_path, opt_path;
    std::mutex mu;
    std::thread last;
    Buf pending;
    bool has_pending = false;

    AsyncSaver()
        : model_path("../model/gpt_model.bin"), opt_path("../model/gpt_opt.bin") {}
    AsyncSaver(std::string mp, std::string op)
        : model_path(std::move(mp)), opt_path(std::move(op)) {}

    void queue_model(const std::vector<Tensor*>& params, i32 dim, i32 blocks, i32 ctx) {
        std::lock_guard<std::mutex> lk(mu);
        pending = Buf(); // fresh cycle: call queue_model() before queue_opt()
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

    void join_prev() {
        std::lock_guard<std::mutex> lk(mu);
        if (last.joinable()) last.join();
    }

    // Spawn a writer for the current pending snapshot (swallows it). A still
    // running previous writer is joined here: assigning to a joinable
    // std::thread would terminate the process.
    void spawn() {
        std::lock_guard<std::mutex> lk(mu);
        if (!has_pending) return;
        if (last.joinable()) last.join();
        Buf snap = std::move(pending);
        has_pending = false;
        const std::string mp = model_path, op = opt_path;
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
            if (!Serialize::save_model(mp, ckpt, snap.dim, snap.blocks, snap.ctx)) {
                fprintf(stderr, "  WARNING: failed to save model checkpoint to %s\n", mp.c_str());
                fflush(stderr);
                return; // do not publish a newer optimizer against an older model
            }
            if (!snap.m.empty()) {
                std::vector<Tensor*> om, ov;
                for (auto& t : snap.m) om.push_back(&t);
                for (auto& t : snap.v) ov.push_back(&t);
                if (!Serialize::save_optimizer(op, om, ov, snap.t)) {
                    fprintf(stderr, "  WARNING: failed to save optimizer state to %s\n", op.c_str());
                    fflush(stderr);
                }
            }
        });
    }

    void flush() {
        std::lock_guard<std::mutex> lk(mu);
        if (last.joinable()) last.join();
    }
};

} // namespace Serialize
