#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <cstdio>

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
        std::sort(sorted.begin(), sorted.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });

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

    bool save(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb"); if (!f) return false;
        i32 n = (i32)id2word.size(); fwrite(&n, sizeof(i32), 1, f);
        for (auto& w : id2word) {
            i32 len = (i32)w.size(); fwrite(&len, sizeof(i32), 1, f);
            fwrite(w.data(), 1, len, f);
        }
        fclose(f); return true;
    }

    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb"); if (!f) return false;
        i32 n; fread(&n, sizeof(i32), 1, f);
        id2word.resize(n); word2id.clear();
        for (i32 i = 0; i < n; i++) {
            i32 len; fread(&len, sizeof(i32), 1, f);
            std::string w(len, '\0'); fread(&w[0], 1, len, f);
            id2word[i] = w; word2id[w] = i;
        }
        vocab_size = n; fclose(f); return true;
    }
};
