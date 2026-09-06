#pragma once
#include "../core/tensor.h"
#include "../utils/tokenizer_bpe.h"
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>

// Simple RAG: splits a knowledge base into chunks, ranks chunks by
// token overlap with the query, and returns the top-k chunks.
struct SimpleRAG {
    struct Chunk { std::string text; std::vector<int> tokens; int n = 0; };

    BPETokenizer* tok = nullptr;
    std::vector<Chunk> chunks;
    int chunk_size = 64;       // tokens per chunk
    int max_results = 2;       // top-k chunks to retrieve

    SimpleRAG() {}
    SimpleRAG(BPETokenizer* t, int cs = 64, int mr = 2) : tok(t), chunk_size(cs), max_results(mr) {}

    // Load KB text file and split into overlapping chunks.
    void load(const std::string& path, int chunk_tokens = 64, int overlap = 8) {
        chunk_size = chunk_tokens;
        std::ifstream f(path);
        if (!f.is_open()) { chunks.clear(); return; }
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        f.close();

        std::vector<int> ids = tok->encode(text);
        int step = std::max(1, chunk_size - overlap);
        for (size_t i = 0; i < ids.size(); i += step) {
            int n = std::min(chunk_size, (int)ids.size() - (int)i);
            if (n < 8) break;
            Chunk c;
            c.tokens.assign(ids.begin() + i, ids.begin() + i + n);
            c.n = n;
            chunks.push_back(std::move(c));
        }
    }

    // Token overlap score (number of shared token ids, normalized).
    static float score(const std::vector<int>& q, const Chunk& c) {
        int match = 0;
        for (int t : q) {
            for (int ct : c.tokens) if (ct == t) { match++; break; }
        }
        return (float)match / (float)std::max(1, (int)q.size() + c.n / 2);
    }

    // Retrieve top-k chunks' text for a query.
    std::string retrieve(const std::string& query) {
        if (chunks.empty()) return "";
        std::vector<int> q = tok->encode(query);
        std::vector<std::pair<float, int>> scored;
        for (int i = 0; i < (int)chunks.size(); i++)
            scored.push_back({ score(q, chunks[i]), i });
        std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });

        std::string context;
        int got = 0;
        for (auto& s : scored) {
            if (s.first < 0.05f) break;
            std::string text = tok->decode(chunks[s.second].tokens);
            if (text.size() > 300) text = text.substr(0, 300);
            if (!context.empty()) context += "\n";
            context += text;
            if (++got >= max_results) break;
        }
        return context;
    }
};
