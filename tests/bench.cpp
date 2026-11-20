// Compare: cl /O2 /arch:AVX  vs  cl /O2  (scalar fallback)
#include "../src/core/tensor.h"
#include "../src/layers/attention.h"
#include "../src/optimizer/adamw.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

using Clock = std::chrono::steady_clock;

int main() {
    std::srand(7);
    {
        const i32 D = 256;
        Tensor A(D, D), B(D, D);
        for (i32 i = 0; i < A.size(); i++) A.data[i] = (f32)rand() / RAND_MAX - 0.5f;
        for (i32 i = 0; i < B.size(); i++) B.data[i] = (f32)rand() / RAND_MAX - 0.5f;
        const int reps = 50;
        auto t0 = Clock::now();
        for (int r = 0; r < reps; r++) { Tensor C = A.matmul(B); volatile f32 sink = C.data[r]; (void)sink; }
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        std::printf("matmul 256x256x256 : %8.2f ms / %d reps\n", ms, reps);
    }
    {
        SelfAttention a(256, 4, true);
        const i32 T = 64;
        Tensor x(T, 256);
        for (i32 i = 0; i < x.size(); i++) x.data[i] = (f32)rand() / RAND_MAX - 0.5f;
        const int reps = 20;
        auto t0 = Clock::now();
        for (int r = 0; r < reps; r++) { Tensor y = a.forward(x); volatile f32 sink = y.data[r]; (void)sink; }
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        std::printf("attention T=64 d=256 : %8.2f ms / %d reps\n", ms, reps);
    }
    {
        const i32 n = 1000000;
        Tensor p(1, n), g(1, n);
        for (i32 i = 0; i < n; i++) { p.data[i] = 0.01f; g.data[i] = 0.001f; }
        AdamW opt(1e-3f, 0.01f);
        opt.add(p);
        const int reps = 20;
        auto t0 = Clock::now();
        for (int r = 0; r < reps; r++) {
            std::vector<Tensor*> ps{ &p }; std::vector<Tensor*> gs{ &g };
            opt.step(ps, gs);
        }
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        std::printf("adamw 1M params      : %8.2f ms / %d reps\n", ms, reps);
    }
}
