// One-time generator for tests/fixtures/shdg_v1_min.bin.
// Compile and run ONCE during development; the resulting binary is committed
// as a stable fixture. Tests never regenerate it from the current serializer.
//   cl /nologo /std:c++17 /O2 /EHsc /I ..\..\src gen_v1_fixture.cpp
//   gen_v1_fixture.exe shdg_v1_min.bin
#define main gpt_cli_main
#include "../../src/gpt.cpp"
#undef main
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    FILE* f = fopen(argv[1], "wb");
    if (!f) return 1;
    using namespace Serialize::detail;

    const GPTConfig c{ 8, 4, 1, 4 }; // smallest legal config: vocab>=SPECIAL, dim%4==0
    Tokenizer tok;
    tok.id2word = { "<PAD>", "<UNK>", "<BOS>", "<EOS>", "a", "b", "c", "d" };
    tok.word2id.clear();
    for (int i = 0; i < (int)tok.id2word.size(); i++) tok.word2id[tok.id2word[i]] = i;
    tok.vocab_size = 8;

    GPT gpt(c.vocab, c.dim, c.blocks, c.max_len);
    auto params = gpt.params();
    for (auto* p : params)
        for (i32 i = 0; i < p->size(); i++) p->data[i] = 0.01f * (f32)(i % 5);

    bool ok = true;
    auto wi = [&](i32 v) { if (ok) ok = write_i32(f, v); };
    wi(GPT_MAGIC);
    wi(1); // SHDG v1: no checksum field
    wi(c.vocab); wi(c.dim); wi(c.blocks); wi(c.max_len);
    if (ok) ok = tok.write(f);
    wi((i32)params.size());
    for (auto* p : params) { if (!ok) break; ok = write_tensor(f, *p); }
    if (fclose(f) != 0) ok = false;
    return ok ? 0 : 1;
}
