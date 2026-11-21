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
#include <cerrno>
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

// Staged readers: validate the whole file and return tensors in `staged`.
// Nothing is written to live tensors until the caller commits (so a paired
// loader can commit model + optimizer together).
inline bool read_model_blocks(FILE* f, const std::vector<Tensor*>& params,
                              i32& dim, i32& blocks, i32& ctx, std::vector<Tensor>& staged) {
    i32 header[4];
    if (!read_exact(f, header, sizeof(header)) || header[0] != 0x53484445) return false;
    if (header[1] <= 0 || header[2] <= 0 || header[3] <= 0) return false;
    i32 n;
    if (!read_i32(f, n) || n < 0 || n != (i32)params.size()) return false;
    staged.clear();
    staged.reserve(n);
    for (i32 i = 0; i < n; i++) {
        Tensor t;
        if (!read_tensor(f, t, params[i]->rows, params[i]->cols)) return false;
        staged.push_back(std::move(t));
    }
    if ((i32)staged.size() != n || !at_eof(f)) return false;
    dim = header[1]; blocks = header[2]; ctx = header[3];
    return true;
}

inline bool read_optimizer_blocks(FILE* f, const std::vector<Tensor*>& m,
                                  const std::vector<Tensor*>& v, i32& t,
                                  std::vector<Tensor>& sm, std::vector<Tensor>& sv) {
    if (m.size() != v.size()) return false;
    i32 n;
    if (!read_i32(f, n) || n < 0 || n != (i32)m.size()) return false;
    sm.clear(); sv.clear();
    sm.reserve(n); sv.reserve(n);
    for (i32 i = 0; i < n; i++) {
        Tensor x;
        if (!read_tensor(f, x, m[i]->rows, m[i]->cols)) return false;
        sm.push_back(std::move(x));
    }
    for (i32 i = 0; i < n; i++) {
        Tensor x;
        if (!read_tensor(f, x, v[i]->rows, v[i]->cols)) return false;
        sv.push_back(std::move(x));
    }
    i32 new_t;
    if (!read_i32(f, new_t) || new_t < 0 || !at_eof(f)) return false;
    t = new_t;
    return true;
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
    std::vector<Tensor> staged;
    i32 d = 0, b = 0, c = 0;
    const bool ok = detail::read_model_blocks(f, params, d, b, c, staged);
    fclose(f);
    if (!ok) return bad;
    detail::commit(staged, params);
    return { d, b, c, true };
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
    std::vector<Tensor> sm, sv;
    i32 new_t = 0;
    const bool ok = detail::read_optimizer_blocks(f, m, v, new_t, sm, sv);
    fclose(f);
    if (!ok) return false;
    detail::commit(sm, m);
    detail::commit(sv, v);
    t = new_t;
    return true;
}

// ---- Paired checkpoint publishing (model + AdamW state as one generation) ----
// Two separately atomic files are not enough: a crash, or a failed optimizer
// write after a successful model write, leaves a new model beside an old
// optimizer — and both load_* calls succeed on that mix. Both snapshots are
// therefore written under a generation number and one small manifest is
// atomically replaced LAST. A reader sees either the old pair or the new pair,
// never a mix. Legacy plain files remain readable when no manifest exists.
namespace pair_detail {

static constexpr i32 MANIFEST_MAGIC = 0x52494150; // "PAIR" on disk (little-endian)
static constexpr i32 FLAG_HAS_OPT = 1;

// One lock for the whole manifest/generation protocol: publish is serialized
// (no two writers can claim the same gen), and load holds it from the manifest
// read through opening both files, so GC cannot delete a generation under a
// reader. Cross-process coordination is out of scope; this is in-process.
inline std::mutex& io_mutex() {
    static std::mutex m;
    return m;
}

inline std::string gen_path(const std::string& base, i32 gen) {
    return base + ".g" + std::to_string(gen);
}
inline std::string manifest_path(const std::string& model_path) {
    return model_path + ".manifest";
}

// `exists` separates "no manifest yet" (legacy fallback) from "manifest
// present but corrupt" (hard failure). gen/flags are only set on success.
inline bool read_manifest(const std::string& model_path, bool& exists, i32& gen, i32& flags) {
    errno = 0;
    FILE* f = fopen(manifest_path(model_path).c_str(), "rb");
    if (!f) {
        // Only ENOENT may fall back to the legacy pair. ENOTDIR is a path
        // configuration error (a component is a file), not an absent file.
        // MSVC's CRT reports that case as ENOENT/EINVAL and a directory as
        // EACCES, all of which are hard failures here as well.
        exists = (errno != ENOENT);
        return false;
    }
    exists = true;
    i32 h[3];
    const bool ok = detail::read_exact(f, h, sizeof(h)) && detail::at_eof(f);
    fclose(f);
    // Bound gen so a corrupt manifest cannot overflow gen+1 or drive the
    // garbage-collection loop over billions of missing files.
    if (!ok || h[0] != MANIFEST_MAGIC || h[1] <= 0 || h[1] > 1000000 ||
        (h[2] & ~FLAG_HAS_OPT) != 0)
        return false;
    gen = h[1]; flags = h[2];
    return true;
}

inline bool write_manifest(const std::string& model_path, i32 gen, i32 flags) {
    return detail::write_atomic(manifest_path(model_path), [&](FILE* f) {
        const i32 h[3] = { MANIFEST_MAGIC, gen, flags };
        return detail::write_exact(f, h, sizeof(h));
    });
}

} // namespace pair_detail

struct CheckpointPair {
    struct Result {
        bool valid;
        ModelInfo info;
        bool has_opt;
        i32 t;
    };

    // Publish model + optimizer as one generation. Any failure before the
    // manifest replace leaves readers on the previous generation.
    static bool publish(const std::string& model_path, const std::string& opt_path,
                        const std::vector<Tensor*>& params, i32 dim, i32 blocks, i32 ctx,
                        const std::vector<Tensor*>& m, const std::vector<Tensor*>& v, i32 t) {
        // Serialize the whole read-gen -> write -> manifest-swap -> GC cycle.
        // Otherwise two publishers can claim the same generation, and a GC can
        // delete a generation that a concurrent reader has already selected.
        std::lock_guard<std::mutex> lk(pair_detail::io_mutex());

        bool exists = false;
        i32 prev = 0, prev_flags = 0;
        if (pair_detail::read_manifest(model_path, exists, prev, prev_flags)) {
            // advance from the current generation
        } else if (exists) {
            return false; // corrupt manifest: never silently restart at g1
        }
        const i32 gen = prev + 1;
        const bool has_opt = !m.empty();
        // Check the pair shape unconditionally: v without m is a caller bug,
        // not a legitimate model-only publish.
        if (m.size() != v.size() || (has_opt && t < 0)) return false;

        const std::string mg = pair_detail::gen_path(model_path, gen);
        const std::string og = pair_detail::gen_path(opt_path, gen);
        if (!save_model(mg, params, dim, blocks, ctx)) return false;
        if (has_opt && !save_optimizer(og, m, v, t)) return false;

        // Verify the staged files before advertising them. write_atomic already
        // checked the write itself; this also catches a bad shape/config.
        const ModelInfo vi = read_model_info(mg);
        if (!vi.valid || vi.dim != dim || vi.blocks != blocks || vi.ctx != ctx) return false;
        if (has_opt) {
            std::vector<Tensor> tm, tv;
            std::vector<Tensor*> pm, pv;
            tm.reserve(m.size()); tv.reserve(v.size());
            for (auto* p : m) { tm.emplace_back(p->rows, p->cols); pm.push_back(&tm.back()); }
            for (auto* p : v) { tv.emplace_back(p->rows, p->cols); pv.push_back(&tv.back()); }
            i32 vt = -1;
            if (!load_optimizer(og, pm, pv, vt) || vt != t) return false;
        }

        if (!pair_detail::write_manifest(model_path, gen,
                                         has_opt ? pair_detail::FLAG_HAS_OPT : 0))
            return false;

        // Keep two newest generations: a reader may still hold the old manifest.
        for (i32 g = 1; g <= gen - 2; g++) {
            std::remove(pair_detail::gen_path(model_path, g).c_str());
            std::remove(pair_detail::gen_path(opt_path, g).c_str());
        }
        return true;
    }

    // Load the generation named by the manifest. Both files are staged before
    // either is committed, so a corrupt pair cannot half-update live tensors.
    static Result load(const std::string& model_path, const std::string& opt_path,
                       const std::vector<Tensor*>& params,
                       const std::vector<Tensor*>& m, const std::vector<Tensor*>& v) {
        // Hold the protocol lock from the manifest read until both files are
        // staged and committed. A publisher (and its GC) cannot advance past
        // the selected generation while this reader is still using it.
        std::lock_guard<std::mutex> lk(pair_detail::io_mutex());

        const Result bad = { false, { 0, 0, 0, false }, false, 0 };
        bool exists = false;
        i32 gen = 0, flags = 0;
        const bool manifest_ok = pair_detail::read_manifest(model_path, exists, gen, flags);

        if (!exists) { // legacy pair: plain files, no generation stamp
            FILE* f = fopen(model_path.c_str(), "rb");
            if (!f) return bad;
            std::vector<Tensor> sp;
            i32 d = 0, b = 0, c = 0;
            const bool ok = detail::read_model_blocks(f, params, d, b, c, sp);
            fclose(f);
            if (!ok) return bad;
            std::vector<Tensor> sm, sv;
            i32 t = 0;
            bool has_opt = false;
            // P3-A: check the pair shape unconditionally. A caller passing v
            // without m must fail even when there is no sidecar to read.
            if (m.size() != v.size()) return bad;
            if (!m.empty()) {
                errno = 0;
                FILE* fo = fopen(opt_path.c_str(), "rb");
                if (!fo) {
                    // P3-B: only ENOENT means "model-only legacy checkpoint".
                    // ENOTDIR is a broken path (file used as a directory), and
                    // EACCES/EINVAL are broken permissions/paths — all hard
                    // failures. Resuming with zero moments must never be the
                    // fallback for a configuration error.
                    if (errno != ENOENT) return bad;
                } else {
                    // A file that exists but fails validation is a hard error.
                    // The generation path already behaves this way; legacy
                    // must not silently degrade to a model-only load.
                    if (!detail::read_optimizer_blocks(fo, m, v, t, sm, sv)) {
                        fclose(fo);
                        return bad;
                    }
                    fclose(fo);
                    has_opt = true;
                }
            }
            detail::commit(sp, params);
            if (has_opt) { detail::commit(sm, m); detail::commit(sv, v); }
            return { true, { d, b, c, true }, has_opt, t };
        }
        if (!manifest_ok) return bad; // present but corrupt

        std::vector<Tensor> sp;
        i32 d = 0, b = 0, c = 0;
        {
            FILE* f = fopen(pair_detail::gen_path(model_path, gen).c_str(), "rb");
            if (!f) return bad;
            const bool ok = detail::read_model_blocks(f, params, d, b, c, sp);
            fclose(f);
            if (!ok) return bad;
        }

        const bool has_opt = (flags & pair_detail::FLAG_HAS_OPT) != 0;
        std::vector<Tensor> sm, sv;
        i32 t = 0;
        if (has_opt) {
            if (m.empty() || m.size() != v.size()) return bad;
            FILE* f = fopen(pair_detail::gen_path(opt_path, gen).c_str(), "rb");
            if (!f) return bad;
            const bool ok = detail::read_optimizer_blocks(f, m, v, t, sm, sv);
            fclose(f);
            if (!ok) return bad;
        }

        detail::commit(sp, params);
        if (has_opt) { detail::commit(sm, m); detail::commit(sv, v); }
        return { true, { d, b, c, true }, has_opt, t };
    }
};

// ---- async (threaded) checkpoint writer ----
// NOTE: this writes generation-stamped SHDE files via CheckpointPair, NOT the
// self-contained v2 GPT checkpoint from save_gpt() (SHDG + version + tokenizer
// + CRC). Do not point these paths at a GPT checkpoint. Use CheckpointPair::load
// to read what this class publishes; plain load_model/load_optimizer still read
// the legacy unsuffixed files.
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
    // A joinable std::thread at destruction would call std::terminate.
    ~AsyncSaver() { flush(); }

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
            std::vector<Tensor*> om, ov;
            for (auto& t : snap.m) om.push_back(&t);
            for (auto& t : snap.v) ov.push_back(&t);
            if (!Serialize::CheckpointPair::publish(mp, op, ckpt,
                                                    snap.dim, snap.blocks, snap.ctx,
                                                    om, ov, snap.t)) {
                fprintf(stderr, "  WARNING: failed to publish checkpoint pair %s / %s\n",
                        mp.c_str(), op.c_str());
                fflush(stderr);
            }
        });
    }

    void flush() {
        std::lock_guard<std::mutex> lk(mu);
        if (last.joinable()) last.join();
    }
};

} // namespace Serialize
