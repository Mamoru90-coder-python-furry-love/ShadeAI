#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <utility>

struct Tokenizer {
    std::unordered_map<std::string, int> word2id;
    std::vector<std::string> id2word;
    int vocab_size;
    static constexpr int PAD = 0, UNK = 1, BOS = 2, EOS = 3;
    static constexpr int SPECIAL = 4;

    Tokenizer() {}

    void build(const std::string& text, int max_vocab = 8000) {
        std::unordered_map<std::string, int> freq;
        auto tokens = tokenize(text);
        for (auto& t : tokens) freq[t]++;

        std::vector<std::pair<int, std::string>> sorted;
        for (auto& kv : freq)
            sorted.push_back({kv.second, kv.first});
        // Frequency descending, then token ascending. unordered_map iteration
        // order is not stable across STL implementations, so equal frequencies
        // must be broken by the token itself to keep checkpoints reproducible.
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) {
                      if (a.first != b.first) return a.first > b.first;
                      return a.second < b.second;
                  });

        id2word = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"};
        word2id = {{"<PAD>", PAD}, {"<UNK>", UNK}, {"<BOS>", BOS}, {"<EOS>", EOS}};
        int limit = std::min((int)sorted.size(), max_vocab - SPECIAL);
        for (int i = 0; i < limit; i++) {
            id2word.push_back(sorted[i].second);
            word2id[sorted[i].second] = i + SPECIAL;
        }
        vocab_size = (int)id2word.size();
    }

    static bool is_word_char(unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c >= 0x80 ||
               c == '\'' || c == '-' || c == '_';
    }
    static unsigned char to_lower(unsigned char c) {
        if (c >= 'A' && c <= 'Z') return c + 32;
        return c;
    }
    std::vector<std::string> tokenize(const std::string& text) const {
        std::vector<std::string> tokens;
        std::string word;
        for (char c : text) {
            unsigned char uc = (unsigned char)c;
            if (is_word_char(uc)) {
                word += to_lower(uc);
            } else {
                if (!word.empty()) { tokens.push_back(word); word.clear(); }
                if (!std::isspace(uc))
                    tokens.push_back(std::string(1, c));
            }
        }
        if (!word.empty()) tokens.push_back(word);
        return tokens;
    }

    std::vector<int> encode(const std::string& text) const {
        auto tokens = tokenize(text);
        std::vector<int> ids;
        ids.push_back(BOS);
        for (auto& t : tokens) {
            auto it = word2id.find(t);
            ids.push_back(it != word2id.end() ? it->second : UNK);
        }
        ids.push_back(EOS);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            if (id < 0 || id >= (int)id2word.size()) continue;
            auto& w = id2word[id];
            if (w == "<PAD>" || w == "<BOS>" || w == "<EOS>") continue;
            if (w == "<UNK>") { result += "?"; continue; }
            if (!result.empty() &&
                (is_word_char((unsigned char)w[0])))
                result += " ";
            result += w;
        }
        return result;
    }

    // Raw binary I/O so the vocabulary can be embedded in a model checkpoint.
    // Bounds are checked: a corrupt file must not trigger huge allocations.
    bool write(FILE* f) const {
        if (!f) return false;
        i32 n = (i32)id2word.size();
        if (fwrite(&n, sizeof(i32), 1, f) != 1) return false;
        for (const auto& w : id2word) {
            i32 len = (i32)w.size();
            if (fwrite(&len, sizeof(i32), 1, f) != 1) return false;
            if (len > 0 && fwrite(w.data(), 1, (size_t)len, f) != (size_t)len) return false;
        }
        return true;
    }

    // Upper bounds for metadata read from an untrusted checkpoint. The byte
    // budget covers BOTH containers: id2word (string object + data) and
    // word2id (node + duplicated key string + bucket). It is charged before
    // each allocation, so 1e6 words x 1 MiB cannot exhaust RAM.
    static constexpr i32 MAX_WORDS = 1000000;
    static constexpr i32 MAX_WORD_LEN = 1 << 20;
    static constexpr size_t MAX_BYTES = 64u * 1024u * 1024u;
    static constexpr size_t NODE_AND_BUCKET = 48; // unordered_map estimate
    static constexpr size_t BASE_PER_WORD = 2 * sizeof(std::string) + NODE_AND_BUCKET;

    // The checkpoint writer must not publish a vocabulary that read() rejects.
    // Mirror its structural checks here, including the default byte budget,
    // so a directly mutated Tokenizer cannot create an unloadable model.
    bool wire_valid() const {
        const size_t n = id2word.size();
        if (vocab_size < SPECIAL || vocab_size > MAX_WORDS || n != (size_t)vocab_size)
            return false;
        if (word2id.size() != n) return false;

        static const char* specials[SPECIAL] = { "<PAD>", "<UNK>", "<BOS>", "<EOS>" };
        for (i32 i = 0; i < SPECIAL; i++)
            if (id2word[(size_t)i] != specials[i]) return false;

        size_t used = 0;
        for (size_t i = 0; i < n; i++) {
            const std::string& w = id2word[i];
            if (w.empty() || w.size() > (size_t)MAX_WORD_LEN) return false;
            const size_t cost = BASE_PER_WORD + 2 * (w.size() + 1);
            if (cost > MAX_BYTES - used) return false;
            used += cost;
            auto it = word2id.find(w);
            if (it == word2id.end() || it->second != (i32)i) return false;
        }
        return true;
    }

    // max_bytes can be tightened by the caller (e.g. a checkpoint reader with
    // its own budget). Nothing is committed to *this until every check passes.
    bool read(FILE* f, size_t max_bytes = MAX_BYTES) {
        if (!f) return false;

        // Determine how many bytes actually remain so a declared length is
        // never used to allocate before the file is known to contain it.
        const long start = ftell(f);
        if (start < 0 || fseek(f, 0, SEEK_END) != 0) return false;
        const long end = ftell(f);
        if (end < start || fseek(f, start, SEEK_SET) != 0) return false;
        int64_t remaining = (int64_t)end - start;

        i32 n;
        if (remaining < (int64_t)sizeof(i32) || fread(&n, sizeof(i32), 1, f) != 1) return false;
        remaining -= sizeof(i32);
        if (n < SPECIAL || n > MAX_WORDS) return false;
        // Every word costs at least BASE_PER_WORD + 2 bytes (two NULs).
        if ((size_t)n > max_bytes / (BASE_PER_WORD + 2)) return false;

        std::vector<std::string> words;
        words.reserve((size_t)n);
        size_t used = 0;
        for (i32 i = 0; i < n; i++) {
            if (remaining < (int64_t)sizeof(i32)) return false;
            i32 len;
            if (fread(&len, sizeof(i32), 1, f) != 1) return false;
            remaining -= sizeof(i32);
            // An empty token can never be produced by tokenize() and makes
            // decode() ambiguous, so it is invalid in a checkpoint vocabulary.
            if (len <= 0 || len > MAX_WORD_LEN) return false;
            if ((int64_t)len > remaining) return false; // file must contain the payload
            remaining -= len;

            const size_t cost = BASE_PER_WORD + 2 * (size_t)(len + 1);
            if (cost > max_bytes - used) return false; // no underflow: used <= max_bytes
            used += cost;

            std::string w((size_t)len, '\0');
            if (len > 0 && fread(&w[0], 1, (size_t)len, f) != (size_t)len) return false;
            words.push_back(std::move(w));
        }

        // Generation indexes BOS/EOS/UNK directly, so their IDs must be intact.
        static const char* specials[SPECIAL] = { "<PAD>", "<UNK>", "<BOS>", "<EOS>" };
        for (i32 i = 0; i < SPECIAL; i++)
            if (words[(size_t)i] != specials[i]) return false;

        std::unordered_map<std::string, int> map;
        map.reserve((size_t)n);
        for (i32 i = 0; i < n; i++)
            if (!map.emplace(words[(size_t)i], i).second) return false; // duplicate token
        if ((i32)map.size() != n) return false;

        id2word = std::move(words);
        word2id = std::move(map);
        vocab_size = n;
        return true;
    }

    bool save(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb"); if (!f) return false;
        bool ok = write(f);
        return fclose(f) == 0 && ok;
    }

    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb"); if (!f) return false;
        bool ok = read(f);
        if (ok && fgetc(f) != EOF) ok = false; // reject trailing garbage
        return fclose(f) == 0 && ok;
    }
};
