// End-to-end GPT resume: byte-exact continuity, transactional rejection of a
// corrupt resume section, config/tokenizer restore, and moment sensitivity.
// gpt.cpp's CLI main is renamed so the production functions can be driven.
#define main gpt_cli_main
#include "../src/gpt.cpp"
#undef main

#include <filesystem>
#include <fstream>
#include <iterator>

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
static bool same_bytes(const fs::path& a, const fs::path& b) {
    return read_all(a) == read_all(b);
}
// Recompute the v2/v3 checksum after a deliberate byte poke. This lets the
// controls test STRUCTURAL validation, not merely CRC detection.
static bool reseal(const fs::path& p) {
    FILE* f = fopen(p.string().c_str(), "r+b");
    if (!f) return false;
    uint32_t crc = 0;
    bool good = gpt_compute_file_crc(f, crc);
    if (good) good = fseek(f, (long)GPT_CRC_OFFSET, SEEK_SET) == 0;
    if (good) good = gpt_write_u32(f, crc);
    fclose(f);
    return good;
}
static bool poke_i32(const fs::path& p, size_t off, i32 v, bool seal) {
    auto b = read_all(p);
    if (off + sizeof(v) > b.size()) return false;
    std::memcpy(b.data() + off, &v, sizeof(v));
    if (!write_all(p, b)) return false;
    return seal ? reseal(p) : true;
}
// Offsets of the v3 resume section, derived from the file tail so the tests
// do not depend on tokenizer length. New writers append a 4-byte batch size
// after t; legacy v3 files omit it.
struct ResumeOffsets { size_t epoch, has_opt, n, first_rows, t, batch_size; };
static bool resume_offsets(const fs::path& p, const std::vector<Tensor>& m,
                           ResumeOffsets& out) {
    size_t blocks = 0;
    for (const auto& t : m) blocks += 8 + (size_t)t.rows * t.cols * 4;
    const size_t bytes = 16 + 4 + 2 * blocks + 4 + 4;
    const size_t size = read_all(p).size();
    if (size < bytes) return false;
    const size_t off = size - bytes;
    out.epoch = off;
    out.has_opt = off + 12;
    out.n = off + 16;
    out.first_rows = off + 20;
    out.t = off + 20 + 2 * blocks;
    out.batch_size = out.t + 4;
    return true;
}

int main() {
    const fs::path dir = fs::temp_directory_path();
    const fs::path data = dir / "shade_gpt_resume_data.txt";
    const fs::path full = dir / "shade_gpt_resume_full.bin";
    const fs::path part = dir / "shade_gpt_resume_part.bin";
    const fs::path bad = dir / "shade_gpt_resume_bad.bin";
    const fs::path v2 = dir / "shade_gpt_resume_v2.bin";
    const fs::path zero = dir / "shade_gpt_resume_zero.bin";
    const fs::path noopt = dir / "shade_gpt_resume_noopt.bin";
    const fs::path complete = dir / "shade_gpt_resume_complete.bin";
    const fs::path missing = dir / "shade_gpt_resume_absent.bin";
    const fs::path twoonly = dir / "shade_gpt_resume_twoonly.bin";
    const fs::path spaced = dir / "shade gpt resume complete.bin";
    const fs::path spaced_data = dir / "shade gpt resume data.txt";
    const fs::path other_data = dir / "shade_gpt_resume_other_data.txt";
    const fs::path changed = dir / "shade_gpt_resume_changed_data.bin";
    const fs::path shortened = dir / "shade_gpt_resume_shortened.bin";
    const fs::path arch = dir / "shade_gpt_resume_arch.bin";
    const fs::path maxcalls = dir / "shade_gpt_resume_maxcalls.bin";
    const fs::path tmaxsave = dir / "shade_gpt_resume_tmaxsave.bin";
    const fs::path nearmax = dir / "shade_gpt_resume_nearmax.bin";
    const fs::path otherbs = dir / "shade_gpt_resume_otherbs.bin";
    const fs::path legacybs = dir / "shade_gpt_resume_legacybs.bin";
    const fs::path badbs = dir / "shade_gpt_resume_badbs.bin";
    const fs::path badfresh = dir / "shade_gpt_resume_badfresh.bin";
    const fs::path rngbig = dir / "shade_gpt_resume_rngbig.bin";
    const fs::path rngbad = dir / "shade_gpt_resume_rngbad.bin";
    const fs::path badcfg = dir / "shade_gpt_resume_badcfg.bin";
    const fs::path maxlen1 = dir / "shade_gpt_resume_maxlen1.bin";
    for (const auto& p : { data, full, part, bad, v2, zero, noopt, complete, missing,
                           twoonly, spaced, spaced_data, other_data, changed,
                           shortened, arch, maxcalls, tmaxsave, nearmax, otherbs,
                           legacybs, badbs, badfresh, rngbig, rngbad, badcfg, maxlen1 })
        fs::remove(p);

    bool ok = true;
    auto expect = [&](bool cond, const char* what) {
        if (!cond) { std::fprintf(stderr, "FAIL: %s\n", what); ok = false; }
    };

    const std::string text =
        "the quick brown fox jumps over the lazy dog and the dog runs again "
        "into the forest where a slow robot waits for another quiet morning "
        "the story continues with a second line of ordinary training words";
    expect(write_all(data, std::vector<char>(text.begin(), text.end())), "write data");

    GPTTrainConfig small;
    small.dim = 8; small.blocks = 1; small.max_len = 4;
    small.epochs = 3; small.batch_size = 2; small.max_chars = 4096;
    small.verbose = false;
    GPTTrainConfig two = small;
    two.epochs = 2;

    // ---- 1. continuity: 3 epochs vs 2 epochs + resume, byte for byte ----
    train_gpt(data.string(), full.string(), false, small);
    train_gpt(data.string(), part.string(), false, two);
    train_gpt(data.string(), twoonly.string(), false, two);
    train_gpt(data.string(), part.string(), true, small);
    expect(fs::exists(full) && fs::exists(part), "checkpoints written");
    {
        GPTConfig ca{}; Tokenizer ta; std::vector<Tensor> wa; GPTResume ra;
        GPTConfig cb{}; Tokenizer tb; std::vector<Tensor> wb; GPTResume rb;
        bool oka = read_gpt_checkpoint(full.string(), ca, ta, wa, Tokenizer::MAX_BYTES, &ra);
        bool okb = read_gpt_checkpoint(part.string(), cb, tb, wb, Tokenizer::MAX_BYTES, &rb);
        float worst = 0;
        if (oka && okb && wa.size() == wb.size())
            for (size_t i = 0; i < wa.size(); ++i)
                for (size_t j = 0; j < wa[i].data.size() && j < wb[i].data.size(); ++j)
                    worst = std::fmax(worst, std::fabs(wa[i].data[j] - wb[i].data[j]));
        std::printf("  full: ep=%d t=%d calls=%d | 2+resume: ep=%d t=%d calls=%d | worst|dw|=%.9g\n",
                    ra.epoch, ra.t, ra.rand_calls,
                    rb.epoch, rb.t, rb.rand_calls, worst);
    }
    expect(same_bytes(full, part), "3 epochs == 2 + resume (byte-identical)");

    // The stored batch size must win over a caller-supplied override. Before
    // it was serialized, bs=3 here diverged from the uninterrupted bs=2 run.
    {
        GPTTrainConfig changed_bs = small;
        changed_bs.batch_size = 3;
        // part was resumed to epoch 3 above; twoonly is the untouched epoch-2
        // checkpoint, so this call really enters the training loop.
        expect(fs::copy_file(twoonly, otherbs), "stage checkpoint for batch-size override");
        expect(train_gpt(data.string(), otherbs.string(), true, changed_bs),
               "resume with ignored batch_size override succeeds");
        expect(same_bytes(full, otherbs), "stored batch_size beats caller override");
        GPTConfig co{}; Tokenizer to; std::vector<Tensor> wo; GPTResume ro;
        expect(read_gpt_checkpoint(otherbs.string(), co, to, wo,
                                   Tokenizer::MAX_BYTES, &ro) &&
               ro.batch_size == 2, "checkpoint stores the original batch_size");
    }

    GPTConfig cfg{}; Tokenizer tok; std::vector<Tensor> weights; GPTResume res;
    bool read_ok = read_gpt_checkpoint(full.string(), cfg, tok, weights,
                                       Tokenizer::MAX_BYTES, &res);
    expect(read_ok, "v3 checkpoint reads with resume state");
    if (read_ok) {
        expect(cfg.dim == 8 && cfg.blocks == 1 && cfg.max_len == 4,
               "architecture restored from checkpoint");
        expect(res.epoch == 3 && res.has_opt && res.t > 0 && res.rand_calls > 0,
               "progress, moments flag and RNG position restored");
        expect(res.m.size() == weights.size() && res.v.size() == weights.size(),
               "m/v tensor counts match weights");
        expect(tok.id2word.size() > 0, "tokenizer restored from checkpoint");

        // ---- 2. offset probe: proves the poke offsets below are correct ----
        expect(write_all(bad, read_all(full)), "stage bad checkpoint");
        ResumeOffsets o{};
        if (resume_offsets(full, res.m, o)) {
            GPTConfig c2{}; Tokenizer t2; std::vector<Tensor> w2; GPTResume r2;
            const bool poked = poke_i32(bad, o.epoch, 99, true);
            const bool reread = poked && read_gpt_checkpoint(bad.string(), c2, t2, w2,
                                                              Tokenizer::MAX_BYTES, &r2);
            // Confirms the structural pokes below hit the resume section.
            expect(poked && reread && r2.epoch == 99, "resume section offset probe");

            // ---- 3. transactional rejection; caller state must not move ----
            auto rejected = [&](const char* label, auto write_bad) {
                GPTConfig c3{}; Tokenizer t3;
                std::vector<Tensor> w3{ Tensor(1, 1, 123.0f) };
                GPTResume r3;
                r3.epoch = 777;
                const bool r = write_all(bad, read_all(full)) && write_bad() &&
                               read_gpt_checkpoint(bad.string(), c3, t3, w3,
                                                                 Tokenizer::MAX_BYTES, &r3);
                expect(!r, label);
                expect(w3.size() == 1 && w3[0].data[0] == 123.0f,
                       "rejected load leaves weights untouched");
            };
            auto bytes = read_all(full);
            // CRC failure inside the resume section (no reseal).
            rejected("CRC-damaged resume state rejected", [&] {
                return poke_i32(bad, o.epoch, 98, false); // no reseal: CRC must fail
            });
            // Structural failures with a VALID CRC — these exercise the new checks.
            rejected("has_opt=2 rejected", [&] { return poke_i32(bad, o.has_opt, 2, true); });
            rejected("negative epoch rejected", [&] { return poke_i32(bad, o.epoch, -1, true); });
            rejected("too-large rand_calls rejected", [&] {
                return poke_i32(bad, o.epoch + 8, 100000001, true);
            });
            rejected("wrong optimizer tensor count rejected", [&] {
                return poke_i32(bad, o.n, (i32)res.m.size() + 1, true);
            });
            rejected("wrong moment shape rejected", [&] {
                return poke_i32(bad, o.first_rows, res.m[0].rows + 1, true);
            });
            rejected("negative t rejected", [&] { return poke_i32(bad, o.t, -1, true); });
            rejected("t=INT32_MAX rejected", [&] {
                return poke_i32(bad, o.t, std::numeric_limits<i32>::max(), true);
            });
            rejected("batch_size=0 rejected", [&] {
                return poke_i32(bad, o.batch_size, 0, true);
            });
            // Removing only the trailing batch-size field is a LEGACY v3 file
            // and stays valid; cut deeper to truncate the actual resume state.
            rejected("truncated resume section rejected", [&] {
                auto b = bytes; b.resize(b.size() - 8); return write_all(bad, b);
            });

            // The writer must not publish a counter that the reader rejects.
            {
                GPT gmax(cfg.vocab, cfg.dim, cfg.blocks, cfg.max_len);
                auto gp = gmax.params();
                for (size_t i = 0; i < gp.size(); ++i) gp[i]->data = weights[i].data;
                GPTResume tmax = res;
                tmax.t = std::numeric_limits<i32>::max();
                expect(!save_gpt(tmaxsave.string(), gmax, tok, &tmax),
                       "save_gpt rejects t=INT32_MAX");
            }

            // A terminal counter is storable, but an explicit resume must not
            // silently start an epoch whose step() can no longer advance it.
            {
                GPT gnear(cfg.vocab, cfg.dim, cfg.blocks, cfg.max_len);
                auto gp = gnear.params();
                for (size_t i = 0; i < gp.size(); ++i) gp[i]->data = weights[i].data;
                GPTResume nearp = res;
                nearp.t = std::numeric_limits<i32>::max() - 1;
                expect(save_gpt(nearmax.string(), gnear, tok, &nearp),
                       "save_gpt stores terminal t=INT32_MAX-1");

                GPTConfig cn{}; Tokenizer tn; std::vector<Tensor> wn;
                GPTResume rn;
                expect(read_gpt_checkpoint(nearmax.string(), cn, tn, wn),
                       "terminal checkpoint readable for inference");
                expect(!read_gpt_checkpoint(nearmax.string(), cn, tn, wn,
                                            Tokenizer::MAX_BYTES, &rn),
                       "explicit resume rejects terminal t");
            }

            // The writer must not publish a counter the reader rejects — same
            // rule for the newly serialized batch size.
            {
                GPT gbad(cfg.vocab, cfg.dim, cfg.blocks, cfg.max_len);
                auto gp = gbad.params();
                for (size_t i = 0; i < gp.size(); ++i) gp[i]->data = weights[i].data;
                GPTResume bad = res;
                bad.batch_size = 0;
                expect(!save_gpt(badbs.string(), gbad, tok, &bad),
                       "save_gpt rejects batch_size=0");

                GPTTrainConfig invalid = small;
                invalid.batch_size = 0;
                expect(!train_gpt(data.string(), badfresh.string(), false, invalid),
                       "fresh run rejects batch_size=0");

                // The same wire contract must be enforced before a fresh model
                // is built: blocks=0 used to publish an unreadable checkpoint,
                // and dim=15 used to throw out of main instead of returning.
                GPTTrainConfig no_blocks = small;
                no_blocks.blocks = 0;
                fs::remove(badcfg);
                expect(!train_gpt(data.string(), badcfg.string(), false, no_blocks),
                       "fresh run rejects blocks=0");
                expect(!fs::exists(badcfg), "invalid architecture leaves no checkpoint");

                GPTTrainConfig bad_dim = small;
                bad_dim.dim = 15;
                fs::remove(badcfg);
                expect(!train_gpt(data.string(), badcfg.string(), false, bad_dim),
                       "fresh run rejects dim not divisible by 4");
                expect(!fs::exists(badcfg), "invalid dim leaves no checkpoint");
            }

            // max_len=1 is a legal one-token context. seq_len/2 was zero here,
            // so the shuffle loop never advanced and grew `indices` forever.
            {
                GPTTrainConfig one = small;
                one.max_len = 1;
                one.epochs = 1;
                expect(train_gpt(data.string(), maxlen1.string(), false, one),
                       "fresh run with max_len=1 completes");
                GPTConfig c1{}; Tokenizer t1; std::vector<Tensor> w1; GPTResume r1;
                expect(read_gpt_checkpoint(maxlen1.string(), c1, t1, w1,
                                           Tokenizer::MAX_BYTES, &r1) &&
                       c1.max_len == 1,
                       "max_len=1 checkpoint is readable");
            }

            // A v3 file written before the trailing field ended after t. It is
            // still a valid checkpoint and resumes with the historical default.
            {
                auto b = read_all(full);
                b.resize(b.size() - 4);
                expect(write_all(legacybs, b) && reseal(legacybs),
                       "stage legacy v3 without batch_size");
                GPTConfig cl{}; Tokenizer tl; std::vector<Tensor> wl; GPTResume rl;
                expect(read_gpt_checkpoint(legacybs.string(), cl, tl, wl,
                                           Tokenizer::MAX_BYTES, &rl),
                       "legacy v3 without batch_size still resumes");
                expect(rl.batch_size == GPT_DEFAULT_BATCH_SIZE,
                       "legacy v3 defaults batch_size to 8");
            }

            // ---- 4. sensitivity: zeroing m/v must change the continuation ----
            GPTResume zm = res;
            for (auto& t : zm.m) std::fill(t.data.begin(), t.data.end(), 0.0f);
            for (auto& t : zm.v) std::fill(t.data.begin(), t.data.end(), 0.0f);
            GPT rebuild(cfg.vocab, cfg.dim, cfg.blocks, cfg.max_len);
            auto rpp = rebuild.params();
            for (size_t i = 0; i < rpp.size(); ++i) rpp[i]->data = weights[i].data;
            expect(save_gpt(zero.string(), rebuild, tok, &zm),
                   "save zeroed-moment checkpoint");
            train_gpt(data.string(), zero.string(), true, small);
            expect(!same_bytes(full, zero), "resume without moments diverges (sensitivity)");
        }
    }

    // ---- 5. v1/v2 compatibility: inference still reads old files; explicit
    //         resume must refuse them instead of silently restarting. ----
    {
        GPTConfig c{}; Tokenizer t; std::vector<Tensor> w;
        expect(read_gpt_checkpoint(full.string(), c, t, w), "v3 readable without resume ptr");
        GPT g(c.vocab, c.dim, c.blocks, c.max_len);
        auto pp = g.params();
        for (size_t i = 0; i < pp.size(); ++i) pp[i]->data = std::move(w[i].data);
        expect(save_gpt(v2.string(), g, t), "write legacy v2 checkpoint");
        std::vector<Tensor> w2; GPTResume unused;
        expect(read_gpt_checkpoint(v2.string(), c, t, w2), "v2 still readable for inference");
        expect(!read_gpt_checkpoint(v2.string(), c, t, w2,
                                    Tokenizer::MAX_BYTES, &unused),
               "explicit resume rejects v2 (no resume state)");
    }

    // ---- 6. has_opt=0: valid for inference, invalid for EXPLICIT resume and
    //         rejected at the reader boundary (not deep inside train_gpt). ----
    {
        GPT g(cfg.vocab, cfg.dim, cfg.blocks, cfg.max_len);
        auto pp = g.params();
        for (size_t i = 0; i < pp.size(); ++i) pp[i]->data = weights[i].data;
        GPTResume bare = res;
        bare.has_opt = false;
        bare.m.clear();
        bare.v.clear();
        expect(save_gpt(noopt.string(), g, tok, &bare), "write has_opt=0 v3 checkpoint");
        GPTConfig c{}; Tokenizer t; std::vector<Tensor> w; GPTResume r;
        expect(read_gpt_checkpoint(noopt.string(), c, t, w), "has_opt=0 v3 readable for inference");
        expect(!read_gpt_checkpoint(noopt.string(), c, t, w,
                                    Tokenizer::MAX_BYTES, &r),
               "explicit resume rejects has_opt=0 at reader");
    }

    // ---- 7. CLI-level paths and failure exit codes. A checkpoint marked as
    //         complete returns immediately, so this stays fast. ----
    {
        GPT g(cfg.vocab, cfg.dim, cfg.blocks, cfg.max_len);
        auto pp = g.params();
        for (size_t i = 0; i < pp.size(); ++i) pp[i]->data = weights[i].data;
        GPTResume done = res;
        done.epoch = 12;
        expect(save_gpt(complete.string(), g, tok, &done), "write complete checkpoint");

        const fs::path stale = fs::path(complete.string() + ".tmp");
        expect(write_all(stale, std::vector<char>(37, 0x5a)), "create stale tmp file");
        expect(save_gpt(complete.string(), g, tok, &done), "save replaces stale tmp");
        expect(!fs::exists(stale), "stale tmp consumed by atomic replace");

        auto cli = [&](std::initializer_list<const char*> args) {
            std::vector<char*> av;
            for (const char* a : args) av.push_back(const_cast<char*>(a));
            return gpt_cli_main((int)av.size(), av.data());
        };
        // A completed checkpoint needs no dataset at all: progress is checked
        // before any data file is opened.
        expect(cli({ "shade", "--resume", complete.string().c_str() }) == 0,
               "--resume complete exits 0 without --data");
        expect(cli({ "shade", "--continue", complete.string().c_str(),
                     "--data", data.string().c_str() }) == 0,
               "--continue alias exits 0");
        expect(cli({ "shade", "--resume", missing.string().c_str(),
                     "--model", complete.string().c_str() }) == 0,
               "--model overrides earlier --resume path");
        expect(write_all(spaced_data, std::vector<char>(text.begin(), text.end())),
               "write data path with spaces");
        expect(save_gpt(spaced.string(), g, tok, &done), "save checkpoint path with spaces");
        expect(cli({ "shade", "--resume", spaced.string().c_str(),
                     "--data", spaced_data.string().c_str() }) == 0,
               "paths with spaces round-trip through CLI");
        expect(cli({ "shade", "--resume", complete.string().c_str(),
                     "--data", data.string().c_str() }) == 0,
               "--resume with --data exits 0 (dataset change allowed)");
        expect(cli({ "shade", "--resume", missing.string().c_str() }) == 1,
               "missing checkpoint exits 1");
        expect(cli({ "shade", "--resume", bad.string().c_str() }) == 1,
               "corrupt checkpoint exits 1");
        expect(cli({ "shade", "--resume", v2.string().c_str() }) == 1,
               "v2 checkpoint exits 1 on resume");
        expect(cli({ "shade", "--resume", noopt.string().c_str() }) == 1,
               "has_opt=0 checkpoint exits 1 on resume");
    }

    // ---- 8. Dataset identity, epoch limit and non-stored train params. ----
    // There is no dataset fingerprint in SHDG v3: resuming with different text
    // is an intentional fine-tuning path. The tokenizer/architecture still come
    // from the checkpoint, never from a freshly built one.
    {
        const std::string other_text = text +
            " a completely different continuation with fresh words for fine tuning";
        expect(write_all(other_data,
                         std::vector<char>(other_text.begin(), other_text.end())),
               "write alternate dataset");
        expect(fs::copy_file(twoonly, changed), "stage epoch-2 checkpoint");
        expect(train_gpt(other_data.string(), changed.string(), true, small),
               "resume with changed dataset continues");
        GPTConfig c{}; Tokenizer t; std::vector<Tensor> w; GPTResume r;
        expect(read_gpt_checkpoint(changed.string(), c, t, w,
                                   Tokenizer::MAX_BYTES, &r),
               "changed-dataset checkpoint still readable");
        expect(r.epoch == 3 && c.dim == cfg.dim && c.blocks == cfg.blocks &&
               c.max_len == cfg.max_len && t.id2word == tok.id2word,
               "changed dataset keeps checkpoint progress/tokenizer/architecture");
    }

    // Lowering the epoch limit below saved progress must fail without touching
    // the checkpoint. Raising it continues from the saved epoch.
    {
        expect(fs::copy_file(full, shortened), "stage complete checkpoint for epoch limit");
        GPTTrainConfig lower = small;
        lower.epochs = 2;
        expect(!train_gpt(data.string(), shortened.string(), true, lower),
               "resume rejects epoch limit below checkpoint progress");
        expect(same_bytes(full, shortened), "rejected epoch limit leaves bytes unchanged");

        GPTTrainConfig higher = small;
        higher.epochs = 4;
        expect(train_gpt(data.string(), shortened.string(), true, higher),
               "resume continues when epoch limit is raised");
        GPTConfig c{}; Tokenizer t; std::vector<Tensor> w; GPTResume r;
        expect(read_gpt_checkpoint(shortened.string(), c, t, w,
                                   Tokenizer::MAX_BYTES, &r) && r.epoch == 4,
               "raised epoch limit advances checkpoint to epoch 4");
    }

    // Architecture is not serialized as a train knob; it comes from the model
    // header. A caller that passes different dim/blocks/max_len still gets the
    // checkpoint's shapes. batch_size is now serialized separately above.
    {
        expect(fs::copy_file(full, arch), "stage architecture-override checkpoint");
        GPTTrainConfig override = small;
        override.dim = 16; override.blocks = 2; override.max_len = 8; override.epochs = 4;
        expect(train_gpt(data.string(), arch.string(), true, override),
               "resume ignores non-stored architecture overrides");
        GPTConfig c{}; Tokenizer t; std::vector<Tensor> w; GPTResume r;
        expect(read_gpt_checkpoint(arch.string(), c, t, w,
                                   Tokenizer::MAX_BYTES, &r) &&
               c.dim == 8 && c.blocks == 1 && c.max_len == 4 && r.epoch == 4,
               "checkpoint architecture survives caller override");
    }

    // Stored and replayable ranges are separate. The 100M boundary is still
    // resumeable at the reader level; actually replaying it is left to the
    // standalone control because it costs ~2s.
    {
        GPT g(cfg.vocab, cfg.dim, cfg.blocks, cfg.max_len);
        auto pp = g.params();
        for (size_t i = 0; i < pp.size(); ++i) pp[i]->data = weights[i].data;

        GPTResume maxr = res;
        maxr.rand_calls = GPT_MAX_RAND_REPLAY;
        expect(save_gpt(maxcalls.string(), g, tok, &maxr), "write max rand_calls checkpoint");
        GPTConfig c{}; Tokenizer t; std::vector<Tensor> w; GPTResume r;
        expect(read_gpt_checkpoint(maxcalls.string(), c, t, w,
                                   Tokenizer::MAX_BYTES, &r) &&
               r.rand_calls == GPT_MAX_RAND_REPLAY,
               "rand_calls boundary accepted by reader");
        GPTResume rr;
        expect(read_gpt_checkpoint(maxcalls.string(), c, t, w,
                                   Tokenizer::MAX_BYTES, &rr),
               "rand_calls boundary accepted for explicit resume");

        // A long run beyond the replay budget must still be able to publish a
        // checkpoint; that file is terminal for training but readable for
        // inference.
        GPTResume big = res;
        big.rand_calls = GPT_MAX_RAND_REPLAY + 1;
        expect(save_gpt(rngbig.string(), g, tok, &big),
               "save_gpt stores rand_calls past replay budget");
        GPTConfig cb{}; Tokenizer tb2; std::vector<Tensor> wb; GPTResume rb;
        expect(read_gpt_checkpoint(rngbig.string(), cb, tb2, wb),
               "past-budget checkpoint readable for inference");
        expect(!read_gpt_checkpoint(rngbig.string(), cb, tb2, wb,
                                    Tokenizer::MAX_BYTES, &rb),
               "explicit resume rejects past replay budget");

        GPTResume bad = res;
        bad.rand_calls = -1;
        expect(!save_gpt(rngbad.string(), g, tok, &bad),
               "save_gpt rejects negative rand_calls");
    }

    for (const auto& p : { data, full, part, bad, v2, zero, noopt, complete, missing,
                           twoonly, spaced, spaced_data, other_data, changed,
                           shortened, arch, maxcalls, tmaxsave, nearmax, otherbs,
                           legacybs, badbs, badfresh, rngbig, rngbad, badcfg, maxlen1 })
        fs::remove(p);
    if (!ok) { std::fprintf(stderr, "gpt_resume_checks: FAILED\n"); return 1; }
    std::puts("gpt_resume_checks: OK");
    return 0;
}
