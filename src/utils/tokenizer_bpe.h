#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <unordered_set>

struct BPETokenizer {
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> id2token;
    std::vector<std::pair<std::string, std::string>> merges;
    std::unordered_map<std::string, int> merge_rank;

    int PAD_id = 1, UNK_id = 0, BOS_id = 2, EOS_id = 3;
    static constexpr int MAX_ID = 1000000; // bounds id2token against corrupt vocab

    BPETokenizer() {}

    // Parses into temporaries and commits only after BOTH files validate.
    // A failed reload therefore leaves the previous tokenizer intact.
    bool load(const std::string& vocab_path, const std::string& merges_path) {
        std::ifstream fvocab(vocab_path);
        if (!fvocab.is_open()) return false;

        std::unordered_map<std::string, int> new_vocab;
        std::string line;
        while (std::getline(fvocab, line)) {
            if (line.empty()) continue; // exporter emits an empty line for token "\n"
            const size_t pos = line.rfind(' ');
            if (pos == std::string::npos) return false; // corrupt entry: no id separator
            std::string token = line.substr(0, pos);
            const std::string id_str = line.substr(pos + 1);

            int id = 0;
            try {
                size_t used = 0;
                id = std::stoi(id_str, &used);
                if (used != id_str.size()) return false; // trailing junk
            } catch (...) {
                return false; // stoi throws on malformed / out-of-range ids
            }
            if (id < 0 || id > MAX_ID) return false;

            if (token == "SPACE") token = " ";
            else if (token == "NEWLINE") token = "\n";
            else if (token.empty()) token = "\n";
            // train_bpe.py writes token "\n" as a literal newline followed by
            // " <id>"; getline yields an empty token for that line. Recover it
            // instead of letting newlines silently disappear from encode().
            if (token.empty()) return false;
            if (!new_vocab.emplace(token, id).second) return false; // duplicate token
        }
        if (fvocab.bad()) return false; // I/O error must not look like a short valid file

        std::unordered_set<int> seen_ids;
        int max_id = -1;
        for (const auto& kv : new_vocab) {
            if (!seen_ids.insert(kv.second).second) return false; // duplicate id
            max_id = std::max(max_id, kv.second);
        }
        if (max_id < 0) return false;

        auto special_ok = [&](const char* name) {
            auto it = new_vocab.find(name);
            return it != new_vocab.end() && it->second >= 0 && it->second <= max_id;
        };
        if (!special_ok("[PAD]") || !special_ok("[UNK]") ||
            !special_ok("[BOS]") || !special_ok("[EOS]")) return false;

        std::ifstream fmerges(merges_path);
        if (!fmerges.is_open()) return false;
        std::vector<std::pair<std::string, std::string>> new_merges;
        while (std::getline(fmerges, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string a, b, extra;
            if (!(iss >> a >> b) || (iss >> extra)) return false; // malformed merge rule
            new_merges.emplace_back(a, b);
        }
        if (fmerges.bad()) return false; // same: no silent partial merge table

        std::unordered_map<std::string, int> new_rank;
        for (size_t i = 0; i < new_merges.size(); i++) {
            const std::string key = new_merges[i].first + " " + new_merges[i].second;
            new_rank.emplace(key, (int)i); // first definition wins, deterministically
        }

        std::vector<std::string> new_id2token((size_t)max_id + 1);
        for (const auto& kv : new_vocab) new_id2token[(size_t)kv.second] = kv.first;

        vocab = std::move(new_vocab);
        id2token = std::move(new_id2token);
        merges = std::move(new_merges);
        merge_rank = std::move(new_rank);
        PAD_id = vocab["[PAD]"];
        UNK_id = vocab["[UNK]"];
        BOS_id = vocab["[BOS]"];
        EOS_id = vocab["[EOS]"];
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

    // Same whitespace/newline rules as encode(), but returns token strings.
    // istringstream >> used to drop newlines here while encode() preserved them.
    std::vector<std::string> tokenize(const std::string& text) const {
        std::vector<std::string> tokens;
        std::string word, sep;
        auto flush = [&]() {
            if (!word.empty()) {
                if (!tokens.empty()) {
                    if (sep.find('\n') != std::string::npos) {
                        if (vocab.count("\n")) tokens.push_back("\n");
                    } else if (!sep.empty()) {
                        if (vocab.count(" ")) tokens.push_back(" ");
                    }
                }
                auto t = bpe_encode_word(word);
                tokens.insert(tokens.end(), t.begin(), t.end());
                word.clear();
            }
            sep.clear();
        };
        for (char c : text) {
            if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
                flush();
                sep += c;
            } else {
                word += c;
            }
        }
        flush();
        return tokens;
    }
};