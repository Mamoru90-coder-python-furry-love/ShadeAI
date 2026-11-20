#pragma once
#include "tensor.h"
#include "simd.h"
#include <cmath>
#include <stdexcept>

struct CrossEntropyLoss {
    // softmax + cross entropy, returns dlogits for backward
    static f32 forward(const Tensor& logits, const Tensor& target, Tensor& dlogits) {
        const i32 n = logits.rows, C = logits.cols;
        // target is a one-hot tensor with the same shape as logits. Without
        // these guards an empty batch divides by zero and a mismatched target
        // reads past its storage at row i * C + j.
        if (n <= 0 || C <= 0)
            throw std::invalid_argument("CrossEntropyLoss::forward: empty logits");
        if (target.rows != n || target.cols != C)
            throw std::invalid_argument("CrossEntropyLoss::forward: target shape mismatch");
        dlogits = Tensor(n, C);
        f32 loss = 0;
        for (i32 i = 0; i < n; i++) {
            const f32* lp = logits.data.data() + (size_t)i * C;
            const f32* tp = target.data.data() + (size_t)i * C;
            f32* dp = dlogits.data.data() + (size_t)i * C;
            f32 maxv = lp[0];
            for (i32 j = 1; j < C; j++) if (lp[j] > maxv) maxv = lp[j];

            f32 sum = 0;
            i32 j = 0;
#if defined(__AVX__)
            __m256 mv = simd::set8(maxv);
            __m256 sumv = simd::set8(0);
            for (; j + 8 <= C; j += 8) {
                __m256 e = simd::exp8(_mm256_sub_ps(simd::load8(lp + j), mv));
                simd::store8(dp + j, e);
                sumv = _mm256_add_ps(sumv, e);
            }
            sum = simd::hsum(sumv);
#endif
            for (; j < C; j++) {
                f32 e = std::exp(lp[j] - maxv);
                dp[j] = e;
                sum += e;
            }

            // dlogits must match the returned mean loss: p - target, then / batch.
            const f32 inv = 1.0f / (sum * n);
            j = 0;
#if defined(__AVX__)
            __m256 inv8 = simd::set8(inv), tn8 = simd::set8(1.0f / n);
            for (; j + 8 <= C; j += 8)
                simd::store8(dp + j, _mm256_sub_ps(_mm256_mul_ps(simd::load8(dp + j), inv8),
                                                   _mm256_mul_ps(simd::load8(tp + j), tn8)));
#endif
            for (; j < C; j++) dp[j] = dp[j] * inv - tp[j] / n;

            // one-hot in practice: skip 12000 logs per row that multiply by zero
            for (j = 0; j < C; j++) {
                if (tp[j] != 0.0f) {
                    f32 p = std::exp(lp[j] - maxv) / sum;
                    loss -= tp[j] * std::log(std::max(p, 1e-7f));
                }
            }
        }
        return loss / n;
    }
};
