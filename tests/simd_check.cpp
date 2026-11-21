// Self-check for the SIMD kernels: g++ -std=c++17 -O2 -march=native -fopenmp -Isrc tests/simd_check.cpp -o simd_check.exe
#include "../src/core/simd.h"
#include "../src/core/loss.h"
#include "../src/layers/attention.h"
#include "../src/layers/transformer.h"
#include "../src/optimizer/adamw.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

static bool close(f32 a, f32 b, f32 rel = 1e-4f) {
    return std::fabs(a - b) <= rel * (1.0f + std::fabs(b));
}

static Tensor scalar_matmul(const Tensor& A, const Tensor& B) {
    Tensor R(A.rows, B.cols, 0.0f);
    for (i32 i = 0; i < A.rows; i++)
        for (i32 k = 0; k < A.cols; k++)
            for (i32 j = 0; j < B.cols; j++)
                R.at(i, j) += A.at(i, k) * B.at(k, j);
    return R;
}

static Tensor attention_ref(const SelfAttention& a, const Tensor& x) {
    const i32 T = x.rows, D = a.head_dim, H = a.num_heads;
    Tensor Q = scalar_matmul(x, a.Wq), K = scalar_matmul(x, a.Wk), V = scalar_matmul(x, a.Wv);
    Tensor out(T, a.embed_dim, 0.0f);
    f32 scale = 1.0f / std::sqrt((f32)D);
    for (i32 h = 0; h < H; h++) {
        for (i32 i = 0; i < T; i++) {
            std::vector<f32> sm(T);
            f32 maxv = -1e30f;
            for (i32 j = 0; j < T; j++) {
                f32 s = 0;
                for (i32 k = 0; k < D; k++) s += Q.at(i, h * D + k) * K.at(j, h * D + k);
                s *= scale;
                if (a.causal && j > i) s = -1e30f;
                sm[j] = s;
                if (s > maxv) maxv = s;
            }
            f32 sum = 0;
            for (i32 j = 0; j < T; j++) { sm[j] = std::exp(sm[j] - maxv); sum += sm[j]; }
            for (i32 j = 0; j < T; j++) sm[j] /= sum;
            for (i32 c = 0; c < D; c++) {
                f32 s = 0;
                for (i32 k = 0; k < T; k++) s += sm[k] * V.at(k, h * D + c);
                out.at(i, h * D + c) = s;
            }
        }
    }
    return scalar_matmul(out, a.Wo);
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (i32 n : {1, 7, 8, 31, 32, 33, 64, 100}) {
        std::vector<f32> a(n), b(n);
        for (i32 i = 0; i < n; i++) { a[i] = dist(rng); b[i] = dist(rng); }
        f32 ref = 0;
        for (i32 i = 0; i < n; i++) ref += a[i] * b[i];
        assert(close(simd::dot(a.data(), b.data(), n), ref));
    }

#if defined(__AVX__)
    for (f32 x : {-80.f, -40.f, -10.f, -1.f, -0.3f, 0.f, 0.4f, 2.f}) {
        f32 got = _mm256_cvtss_f32(simd::exp8(_mm256_set1_ps(x)));
        assert(close(got, std::exp(x), 1e-5f));
    }
#endif

    for (i32 n : {1, 8, 9, 64, 100}) {
        std::vector<f32> row(n);
        for (i32 i = 0; i < n; i++) row[i] = dist(rng) * 8.0f;
        std::vector<f32> ref = row;
        f32 maxv = ref[0], sum = 0;
        for (i32 i = 1; i < n; i++) if (ref[i] > maxv) maxv = ref[i];
        for (i32 i = 0; i < n; i++) { ref[i] = std::exp(ref[i] - maxv); sum += ref[i]; }
        for (i32 i = 0; i < n; i++) ref[i] /= sum;
        simd::softmax_row(row.data(), n);
        for (i32 i = 0; i < n; i++) assert(close(row[i], ref[i], 2e-4f));
    }

    const i32 M = 13, N = 17, P = 23; // odd sizes exercise both tails
    Tensor A(M, N), B(N, P);
    for (i32 i = 0; i < A.size(); i++) A.data[i] = dist(rng);
    for (i32 i = 0; i < B.size(); i++) B.data[i] = dist(rng);
    Tensor got = A.matmul(B);
    for (i32 i = 0; i < M; i++) {
        for (i32 j = 0; j < P; j++) {
            f32 ref = 0;
            for (i32 k = 0; k < N; k++) ref += A.at(i, k) * B.at(k, j);
            assert(close(got.at(i, j), ref, 2e-4f));
        }
    }

    {
        SelfAttention attn(8, 2, true);
        for (i32 i = 0; i < attn.Wq.size(); i++) attn.Wq.data[i] = std::sin(i * 0.31f);
        for (i32 i = 0; i < attn.Wk.size(); i++) attn.Wk.data[i] = std::cos(i * 0.27f);
        for (i32 i = 0; i < attn.Wv.size(); i++) attn.Wv.data[i] = std::sin(i * 0.17f + 1.0f);
        for (i32 i = 0; i < attn.Wo.size(); i++) attn.Wo.data[i] = std::cos(i * 0.11f + 2.0f);
        Tensor x(5, 8);
        for (i32 i = 0; i < x.size(); i++) x.data[i] = dist(rng);
        Tensor got = attn.forward(x);
        Tensor ref = attention_ref(attn, x);
        for (i32 i = 0; i < got.size(); i++) assert(close(got.data[i], ref.data[i], 3e-3f));
    }

    {
        const i32 n = 3, C = 19;
        Tensor logits(n, C), target(n, C, 0.0f);
        for (i32 i = 0; i < logits.size(); i++) logits.data[i] = dist(rng) * 4.0f;
        for (i32 i = 0; i < n; i++) target.at(i, (i * 5) % C) = 1.0f;
        Tensor d;
        f32 got = CrossEntropyLoss::forward(logits, target, d);

        Tensor rd(n, C);
        f32 ref = 0;
        for (i32 i = 0; i < n; i++) {
            f32 mx = logits.at(i, 0);
            for (i32 j = 1; j < C; j++) mx = std::max(mx, logits.at(i, j));
            f32 sum = 0;
            for (i32 j = 0; j < C; j++) { rd.at(i, j) = std::exp(logits.at(i, j) - mx); sum += rd.at(i, j); }
            for (i32 j = 0; j < C; j++) {
                f32 p = rd.at(i, j) / sum;
                rd.at(i, j) = (p - target.at(i, j)) / n;
                ref -= target.at(i, j) * std::log(std::max(p, 1e-7f));
            }
        }
        ref /= n;
        assert(close(got, ref, 1e-4f));
        for (i32 i = 0; i < d.size(); i++) assert(close(d.data[i], rd.data[i], 1e-4f));
    }

    {
        // Independent AdamW reference: m_hat/(sqrt(v_hat)+eps) + decoupled decay.
        const i32 n = 37;
        const f32 b1 = 0.9f, b2 = 0.999f, wd = 0.1f, lr = 0.01f, eps = 1e-8f;
        Tensor p(1, n), g(1, n);
        std::vector<f32> pref(n), m(n, 0.0f), v(n, 0.0f);
        for (i32 i = 0; i < n; i++) {
            p.data[i] = pref[i] = dist(rng) * 0.1f;
            g.data[i] = dist(rng);
        }
        AdamW opt(0.01f, 0.1f);
        opt.add(p);
        std::vector<Tensor*> ps{ &p }, gs{ &g };
        for (int step = 1; step <= 3; step++) {
            opt.step(ps, gs);
            f32 ib1 = 1.0f / (1.0f - std::pow(b1, step));
            f32 ib2 = 1.0f / (1.0f - std::pow(b2, step));
            for (i32 j = 0; j < n; j++) {
                m[j] = b1 * m[j] + (1.0f - b1) * g.data[j];
                v[j] = b2 * v[j] + (1.0f - b2) * g.data[j] * g.data[j];
                // True AdamW: decay the OLD parameter, then subtract the step.
                // The previous reference decayed `e` (post-step), mirroring the
                // production ordering bug and making this check blind to it.
                f32 e = pref[j] - lr * wd * pref[j];
                e -= lr * (m[j] * ib1) / (std::sqrt(v[j] * ib2) + eps);
                pref[j] = e;
                assert(close(p.data[j], e, 1e-4f));
            }
        }
    }

    {
        const i32 dim = 8, pos = 3;
        Tensor pe = PositionalEncoding(10, dim).forward(pos + 1);
        for (i32 j = 0; j < dim; j++) {
            f32 angle = (f32)pos / std::pow(10000.0f, (f32)(j & ~1) / dim);
            f32 expected = (j % 2 == 0) ? std::sin(angle) : std::cos(angle);
            assert(close(pe.at(pos, j), expected, 1e-5f));
        }
    }

    std::puts("simd_check: OK");
    return 0;
}
