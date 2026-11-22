// Regression: GPT::sample() must not fall back to a top-k-excluded token.
// gpt.cpp is included with a test-only rand() override so the RAND_MAX endpoint
// can be forced deterministically. Production RNG semantics are untouched.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include "core/tensor.h"
#include "core/loss.h"
#include "layers/transformer.h"
#include "layers/dense.h"
#include "utils/embedding.h"
#include "utils/tokenizer.h"
#include "utils/serialize.h"
#include "optimizer/adamw.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <limits>
#include <memory>

static int g_forced_rand = -1;      // -1 means "use the CRT"
static int g_rand_calls_seen = 0;
static int shade_test_rand() {
    ++g_rand_calls_seen;
    if (g_forced_rand >= 0) {
        int v = g_forced_rand;
        g_forced_rand = -1;
        return v;
    }
    return std::rand();
}

#define main gpt_cli_main
#define rand shade_test_rand
#include "../src/gpt.cpp"
#undef rand
#undef main

int main() {
    bool ok = true;
    auto expect = [&](bool cond, const char* what) {
        if (!cond) { std::fprintf(stderr, "FAIL: %s\n", what); ok = false; }
    };

    Tokenizer tok;
    tok.build("the quick brown fox jumps over the lazy dog", 32);
    GPT gpt(tok.vocab_size, 8, 1, 4);

    // n=4, top_k=2 keeps indices 0 and 1; indices 2 and 3 are zeroed.
    const float logits[4] = { 3.0f, 2.0f, 1.0f, 0.0f };
    auto filtered = [&](int top_k) {
        std::vector<float> p(4);
        float mx = logits[0];
        for (int j = 1; j < 4; j++) mx = std::fmax(mx, logits[j]);
        for (int j = 0; j < 4; j++) p[j] = std::exp((logits[j] - mx) / 1.0f);
        keep_top_k(p, top_k);
        return p;
    };
    auto positive = [](int idx, const std::vector<float>& p) {
        return idx >= 0 && idx < (int)p.size() && p[idx] > 0.0f;
    };

    // 1. RAND_MAX endpoint: r == 1.0f, the cumulative walk cannot select a
    //    bucket. The old code returned n-1, which top_k had excluded.
    {
        const auto p = filtered(2);
        expect(p[3] == 0.0f, "top_k excludes the last index");
        g_forced_rand = RAND_MAX;
        g_rand_calls_seen = 0;
        const int got = gpt.sample(const_cast<float*>(logits), 4, 1.0f, 2);
        expect(got != 3, "RAND_MAX fallback skips excluded token");
        expect(positive(got, p), "RAND_MAX fallback has positive probability");
        expect(g_rand_calls_seen == 1, "RAND_MAX path consumes one RNG call");
    }

    // 2. Ordinary sampling still selects a kept token.
    {
        const auto p = filtered(2);
        g_forced_rand = 0;
        g_rand_calls_seen = 0;
        const int got = gpt.sample(const_cast<float*>(logits), 4, 1.0f, 2);
        expect(got == 0, "r=0 selects first kept token");
        expect(positive(got, p), "ordinary sample has positive probability");
        expect(g_rand_calls_seen == 1, "ordinary path consumes one RNG call");
    }

    // 3. With no filtering the last index is legitimate; behaviour is intact.
    {
        const auto p = filtered(10);
        g_forced_rand = RAND_MAX;
        g_rand_calls_seen = 0;
        const int got = gpt.sample(const_cast<float*>(logits), 4, 1.0f, 10);
        expect(positive(got, p), "unfiltered fallback has positive probability");
        expect(g_rand_calls_seen == 1, "unfiltered path consumes one RNG call");
    }

    // 4. Amid-range draw still walks the cumulative distribution normally.
    {
        const auto p = filtered(2);
        g_forced_rand = RAND_MAX / 2;
        g_rand_calls_seen = 0;
        const int got = gpt.sample(const_cast<float*>(logits), 4, 1.0f, 2);
        expect(positive(got, p), "mid-range sample has positive probability");
        expect(g_rand_calls_seen == 1, "mid-range path consumes one RNG call");
    }

    if (ok) std::printf("gpt_sample_check: OK\n");
    return ok ? 0 : 1;
}
