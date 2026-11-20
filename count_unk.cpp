#include "utils/tokenizer_bpe.h"
#include <cstdio>
#include <fstream>
#include <sstream>

int main() {
    BPETokenizer tok;
    if (!tok.load("../data/bpe_vocab.txt", "../data/bpe_merges.txt")) {
        printf("load failed\n"); return 1;
    }
    std::ifstream f("../data/train_full3.txt");
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    long long total = 0, unk = 0;
    const long long limit = 20000000; // 20M chars
    std::string chunk = all.substr(0, (size_t)(limit < (long long)all.size() ? limit : (long long)all.size()));
    auto ids = tok.encode(chunk);
    for (int id : ids) {
        total++;
        if (id == tok.UNK_id) unk++;
    }
    printf("chars=%zu total_tokens=%lld unk=%lld unk_rate=%.3f%%\n",
           chunk.size(), total, unk, 100.0 * (double)unk / (double)total);

    // show a few words that become UNK
    int shown = 0;
    std::istringstream iss(chunk);
    std::string w;
    while (iss >> w && shown < 15) {
        auto t = tok.bpe_encode_word(w);
        for (auto& x : t) {
            if (tok.vocab.find(x) == tok.vocab.end()) {
                printf("UNK from word '%s' -> char/merge '%s'\n", w.c_str(), x.c_str());
                shown++;
                break;
            }
        }
    }
    return 0;
}
