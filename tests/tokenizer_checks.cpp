// BPE / Tokenizer audit: corrupt vocabulary, duplicate tokens, state reuse,
// deterministic vocab ordering, and compatibility with the Python BPE exporter.
#include "../src/core/tensor.h"
#include "../src/utils/tokenizer.h"
#include "../src/utils/tokenizer_bpe.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool write_text(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(s.data(), (std::streamsize)s.size());
    return f.good();
}

static void put_i32(std::string& out, i32 v) {
    const char* p = reinterpret_cast<const char*>(&v);
    out.append(p, sizeof(v));
}

struct LoadResult { bool ok = false; bool threw = false; };

static LoadResult try_load(BPETokenizer& t, const std::string& vocab,
                           const fs::path& vp, const fs::path& mp) {
    LoadResult r;
    if (!write_text(vp, vocab) || !write_text(mp, "#version: 0.2\na b\n")) return r;
    try { r.ok = t.load(vp.string(), mp.string()); }
    catch (...) { r.threw = true; }
    return r;
}

static int bpe_checks() {
    const fs::path dir = fs::temp_directory_path();
    const fs::path vp = dir / "shade_bpe_v.txt";
    const fs::path mp = dir / "shade_bpe_m.txt";
    bool ok = true;
    auto expect = [&](bool cond, const char* what) {
        if (!cond) { std::fprintf(stderr, "FAIL: %s\n", what); ok = false; }
    };

    const std::string good =
        "[UNK] 0\n[PAD] 1\n[BOS] 2\n[EOS] 3\n"
        "a 4\nb 5\nSPACE 6\nNEWLINE 7\nab 8\n";

    // 1. Baseline: a well-formed vocabulary loads without exceptions.
    {
        BPETokenizer t;
        auto r = try_load(t, good, vp, mp);
        expect(r.ok && !r.threw, "valid BPE vocabulary loads");
        expect(t.vocab.count("a") == 1 && t.vocab["a"] == 4, "valid BPE token id");
        expect(t.decode({ 4, 6, 5 }) == "a b", "valid BPE decode");
    }

    // 2. Corrupt id must be rejected, not throw out of load().
    {
        BPETokenizer t;
        auto r = try_load(t, good + "bad abc\n", vp, mp);
        expect(!r.ok && !r.threw, "non-numeric id rejected without exception");
    }

    // 3. Hostile id must not drive a huge id2token allocation.
    {
        BPETokenizer t;
        auto r = try_load(t, good + "hostile 2000000\n", vp, mp);
        expect(!r.ok && !r.threw, "out-of-range id rejected");
    }

    // 4. Negative ids are invalid (decode() cannot represent them).
    {
        BPETokenizer t;
        auto r = try_load(t, good + "negative -3\n", vp, mp);
        expect(!r.ok && !r.threw, "negative id rejected");
    }

    // 5. Two tokens claiming one id make decode() dependent on hash order.
    {
        BPETokenizer t;
        auto r = try_load(t, good + "collision 4\n", vp, mp);
        expect(!r.ok && !r.threw, "duplicate id rejected");
    }

    // 6. Same token written twice with different spellings ("SPACE" and " ")
    //    must not silently overwrite the first entry.
    {
        BPETokenizer t;
        auto r = try_load(t, good + "  6\n", vp, mp);
        expect(!r.ok && !r.threw, "duplicate token key rejected");
    }

    // 7. Reloading must replace state, not accumulate stale tokens.
    {
        BPETokenizer t;
        auto first = try_load(t, good, vp, mp);
        const std::string second =
            "[UNK] 0\n[PAD] 1\n[BOS] 2\n[EOS] 3\n"
            "z 4\nSPACE 5\nNEWLINE 6\n";
        auto reload = try_load(t, second, vp, mp);
        expect(first.ok && reload.ok && !reload.threw, "BPE reload succeeds");
        expect(t.vocab.count("a") == 0 && t.vocab.count("z") == 1,
               "reload replaces old vocabulary");
        expect((int)t.id2token.size() == 7, "reload replaces id2token");
    }

    // 8. A failed load must leave the previously good tokenizer usable.
    {
        BPETokenizer t;
        auto first = try_load(t, good, vp, mp);
        auto bad = try_load(t, good + "bad abc\n", vp, mp);
        expect(first.ok && !bad.ok, "failed BPE load reported");
        expect(t.vocab.count("a") == 1 && t.vocab.count("bad") == 0 &&
               (int)t.id2token.size() == 9,
               "failed load keeps previous state");
        auto ids = t.encode("a");
        expect(ids.size() == 3 && ids[1] == 4, "tokenizer still encodes after failed load");
    }

    // 9. HuggingFace exporter writes token " " as "  6" and token "\n" as a
    //    literal newline followed by " 7". The loader must recover "\n".
    {
        BPETokenizer t;
        const std::string exported =
            "[UNK] 0\n[PAD] 1\n[BOS] 2\n[EOS] 3\n"
            "a 4\nb 5\n"
            "  6\n"    // token " "
            "\n 7\n";  // token "\n" (leading newline -> empty getline, then " 7")
        auto r = try_load(t, exported, vp, mp);
        expect(r.ok && !r.threw, "exported vocabulary loads");
        expect(t.vocab.count("\n") == 1 && t.vocab["\n"] == 7,
               "newline token recovered from exporter format");
        auto ids = t.encode("a\nb");
        bool has_nl = false;
        for (int id : ids) if (id == 7) has_nl = true;
        expect(has_nl, "newline survives encode");
        auto strs = t.tokenize("a\nb");
        bool str_nl = false;
        for (const auto& s : strs) if (s == "\n") str_nl = true;
        expect(str_nl, "tokenize() and encode() agree on newlines");
    }

    fs::remove(vp); fs::remove(mp);
    if (!ok) { std::fprintf(stderr, "bpe_checks: FAILED\n"); return 1; }
    std::puts("bpe_checks: OK");
    return 0;
}

static int tokenizer_checks() {
    bool ok = true;
    auto expect = [&](bool cond, const char* what) {
        if (!cond) { std::fprintf(stderr, "FAIL: %s\n", what); ok = false; }
    };

    // 1. Equal-frequency words must be ordered by token, not by unordered_map
    //    iteration, so a rebuild on another STL produces the same checkpoint.
    {
        Tokenizer t;
        t.build("zeta eta theta iota kappa lambda");
        const char* want[] = { "eta", "iota", "kappa", "lambda", "theta", "zeta" };
        bool sorted = true;
        for (int i = 0; i < 6; i++)
            if (t.id2word[Tokenizer::SPECIAL + i] != want[i]) sorted = false;
        expect(sorted, "deterministic vocab order for equal frequencies");
    }

    // 2. An empty token in a checkpoint vocabulary cannot be produced by
    //    tokenize() and makes decode() ambiguous: reject it.
    {
        std::string bytes;
        put_i32(bytes, 5);
        const char* words[] = { "<PAD>", "<UNK>", "<BOS>", "<EOS>", "" };
        for (const char* w : words) {
            put_i32(bytes, (i32)std::strlen(w));
            bytes.append(w, std::strlen(w));
        }
        const fs::path p = fs::temp_directory_path() / "shade_tok_empty.bin";
        {
            std::ofstream f(p, std::ios::binary | std::ios::trunc);
            f.write(bytes.data(), (std::streamsize)bytes.size());
        }
        FILE* f = fopen(p.string().c_str(), "rb");
        Tokenizer t;
        expect(f && !t.read(f), "empty token in vocabulary rejected");
        if (f) fclose(f);
        fs::remove(p);
    }

    // 3. Regression: a normal vocabulary still round-trips through a file.
    {
        Tokenizer t;
        t.build("alpha beta alpha gamma");
        const fs::path p = fs::temp_directory_path() / "shade_tok_roundtrip.bin";
        expect(t.save(p.string()), "tokenizer save");
        Tokenizer r;
        expect(r.load(p.string()), "tokenizer load");
        expect(r.id2word == t.id2word && r.word2id == t.word2id,
               "tokenizer file round-trip");
        expect(r.encode("ALPHA beta") == t.encode("ALPHA beta"),
               "tokenizer encode stability");
        fs::remove(p);
    }

    if (!ok) { std::fprintf(stderr, "tokenizer_checks: FAILED\n"); return 1; }
    std::puts("tokenizer_checks: OK");
    return 0;
}

int main() {
    int rc = 0;
    rc |= bpe_checks();
    rc |= tokenizer_checks();
    if (rc) { std::fprintf(stderr, "tokenizer audit: FAILED\n"); return 1; }
    std::puts("tokenizer audit: OK");
    return 0;
}
