#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdio>

struct BPETokenizer {
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> id2token;
    std::vector<std::pair<std::string, std::string>> merges;
    std::unordered_map<std::string, int> merge_rank;

    int PAD_id = 1, UNK_id = 0, BOS_id = 2, EOS_id = 3;

    BPETokenizer() {}

    bool load(const std::string& vocab_path, const std::string& merges_path) {
        std::ifstream fvocab(vocab_path);
        if (!fvocab.is_open()) return false;
        std::string line;
        while (std::getline(fvocab, line)) {
            if (line.empty()) continue;
            size_t pos = line.rfind(' ');
            if (pos == std::string::npos) continue;
            std::string token = line.substr(0, pos);
            int id = std::stoi(line.substr(pos + 1));
            if (token == "SPACE") token = " ";
            else if (token == "NEWLINE") token = "\n";
            vocab[token] = id;
        }
        fvocab.close();

        int max_id = 0;
        for (auto& kv : vocab) max_id = std::max(max_id, kv.second);
        id2token.resize(max_id + 1);
        for (auto& kv : vocab) {
            if (kv.second >= 0 && kv.second <= max_id)
                id2token[kv.second] = kv.first;
        }

        // Read special token IDs from vocab
        auto it_pad = vocab.find("[PAD]");
        auto it_unk = vocab.find("[UNK]");
        auto it_bos = vocab.find("[BOS]");
        auto it_eos = vocab.find("[EOS]");
        if (it_pad != vocab.end()) PAD_id = it_pad->second;
        if (it_unk != vocab.end()) UNK_id = it_unk->second;
        if (it_bos != vocab.end()) BOS_id = it_bos->second;
        if (it_eos != vocab.end()) EOS_id = it_eos->second;

        std::ifstream fmerges(merges_path);
        if (!fmerges.is_open()) return false;
        while (std::getline(fmerges, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string a, b;
            if (!(iss >> a >> b)) continue;
            merges.emplace_back(a, b);
        }
        fmerges.close();

        for (int i = 0; i < (int)merges.size(); i++) {
            std::string key = merges[i].first + " " + merges[i].second;
            merge_rank[key] = i;
        }

        return true;
    }

    std::vector<std::string> utf8_chars(const std::string& s) const {
        std::vector<std::string> chars;
        for (size_t i = 0; i < s.size();) {
            unsigned char c = (unsigned char)s[i];
            size_t len = 1;
            if (c >= 0xF0) len = 4;
            else if (c >= 0xE0) len = 3;
            else if (c >= 0xC0) len = 2;
            if (i + len > s.size()) len = 1;
            chars.push_back(s.substr(i, len));
            i += len;
        }
        return chars;
    }

    std::vector<std::string> bpe_encode_word(const std::string& word) const {
        if (word.empty()) return {};
        auto chars = utf8_chars(word);
        std::vector<std::string> tokens = chars;

        bool changed = true;
        while (changed && tokens.size() > 1) {
            changed = false;
            int best_rank = INT_MAX;
            int best_idx = -1;
            for (int i = 0; i < (int)tokens.size() - 1; i++) {
                std::string pair = tokens[i] + " " + tokens[i + 1];
                auto it = merge_rank.find(pair);
                if (it != merge_rank.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                }
            }
            if (best_idx >= 0) {
                std::string merged = tokens[best_idx] + tokens[best_idx + 1];
                tokens[best_idx] = merged;
                tokens.erase(tokens.begin() + best_idx + 1);
                changed = true;
            }
        }
        return tokens;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        ids.push_back(BOS_id);

        // Split into words and whitespace, matching WhitespaceSplit behavior.
        // Between words: if the separator was a newline -> emit "\n" token,
        // otherwise emit " " (space) token. Consecutive spaces collapse to one.
        std::string word;
        std::string sep;      // whitespace that precedes the current word
        bool in_word = false;

        auto flush_word = [&]() {
            if (!word.empty()) {
                if (ids.back() != BOS_id) {
                    if (!sep.empty() && sep.find('\n') != std::string::npos) {
                        auto it = vocab.find("\n");
                        if (it != vocab.end()) ids.push_back(it->second);
                    } else if (!sep.empty()) {
                        auto it = vocab.find(" ");
                        if (it != vocab.end()) ids.push_back(it->second);
                    }
                }
                auto tokens = bpe_encode_word(word);
                for (auto& t : tokens) {
                    auto it = vocab.find(t);
                    ids.push_back(it != vocab.end() ? it->second : UNK_id);
                }
                word.clear();
            }
            sep.clear();
            in_word = false;
        };

        for (size_t i = 0; i < text.size(); i++) {
            char c = text[i];
            if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
                flush_word();
                sep += c;
            } else {
                word += c;
                in_word = true;
            }
        }
        flush_word();

        ids.push_back(EOS_id);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            if (id < 0 || id >= (int)id2token.size()) continue;
            const std::string& t = id2token[id];
            if (id == PAD_id || id == BOS_id || id == EOS_id) continue;
            if (id == UNK_id) { result += "?"; continue; }
            if (t == " ") {
                if (!result.empty() && result.back() != ' ') result += ' ';
            } else if (t == "\n") {
                result += '\n';
            } else if (t == "User:" || t == "Bot:" || t == "AI:") {
                if (!result.empty() && result.back() != ' ' && result.back() != '\n') result += ' ';
                result += t;
            } else {
                result += t;
            }
        }
        return result;
    }

    std::vector<std::string> tokenize(const std::string& text) const {
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string word;
        bool first = true;
        while (iss >> word) {
            if (!first) tokens.push_back(" ");
            first = false;
            auto t = bpe_encode_word(word);
            tokens.insert(tokens.end(), t.begin(), t.end());
        }
        return tokens;
    }
};