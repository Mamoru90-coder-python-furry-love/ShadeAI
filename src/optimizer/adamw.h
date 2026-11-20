#pragma once
#include "../core/tensor.h"
#include "../core/simd.h"
#include <vector>

struct AdamW {
    f32 lr, beta1, beta2, eps, wd;
    i32 t;
    std::vector<Tensor> m, v;

    AdamW(f32 lr = 1e-3f, f32 wd = 0.01f)
        : lr(lr), beta1(0.9f), beta2(0.999f), eps(1e-8f), wd(wd), t(0) {}

    void add(Tensor& p) {
        m.push_back(Tensor(p.rows, p.cols, 0.0f));
        v.push_back(Tensor(p.rows, p.cols, 0.0f));
    }

    void step(const std::vector<Tensor*>& params, const std::vector<Tensor*>& grads) {
        t++;
        // bias-correction factors + fused step coefficients — computed ONCE per step
        f32 p1t = std::pow(beta1, t);
        f32 p2t = std::pow(beta2, t);
        f32 sb2 = std::sqrt(1.0f - p2t);
        f32 ib1 = 1.0f / (1.0f - p1t);
        f32 ib2 = 1.0f / (1.0f - p2t);
        f32 km = lr * sb2 * ib1 * ib1;   // update multiplier
        f32 kw = lr * sb2 * ib1 * wd;    // weight-decay multiplier
        for (size_t i = 0; i < params.size(); i++) {
            const f32* g = grads[i]->data.data();
            f32* p = params[i]->data.data();
            f32* pm = m[i].data.data();
            f32* pv = v[i].data.data();
            const i32 n = params[i]->size();
            i32 j = 0;
#if defined(__AVX__)
            const __m256 b1 = simd::set8(beta1), b2 = simd::set8(beta2);
            const __m256 c1 = simd::set8(1.0f - beta1), c2 = simd::set8(1.0f - beta2);
            const __m256 kmv = simd::set8(km), kwv = simd::set8(kw);
            const __m256 ib2v = simd::set8(ib2), epsv = simd::set8(eps);
            for (; j + 8 <= n; j += 8) {
                __m256 gv = simd::load8(g + j);
                __m256 mv = simd::load8(pm + j);
                __m256 vv = simd::load8(pv + j);
                __m256 pv_ = simd::load8(p + j);
                __m256 mi = _mm256_add_ps(_mm256_mul_ps(b1, mv), _mm256_mul_ps(c1, gv));
                __m256 vi = _mm256_add_ps(_mm256_mul_ps(b2, vv),
                                          _mm256_mul_ps(c2, _mm256_mul_ps(gv, gv)));
                simd::store8(pm + j, mi);
                simd::store8(pv + j, vi);
                __m256 den = _mm256_add_ps(_mm256_sqrt_ps(_mm256_mul_ps(vi, ib2v)), epsv);
                __m256 np = _mm256_sub_ps(pv_, _mm256_div_ps(_mm256_mul_ps(kmv, mi), den));
                np = _mm256_sub_ps(np, _mm256_mul_ps(kwv, np));
                simd::store8(p + j, np);
            }
#endif
            for (; j < n; j++) {
                f32 gi = g[j];
                f32 mi = beta1 * pm[j] + (1 - beta1) * gi;
                f32 vi = beta2 * pv[j] + (1 - beta2) * gi * gi;
                pm[j] = mi; pv[j] = vi;
                p[j] -= km * mi / (std::sqrt(vi * ib2) + eps);
                p[j] -= kw * p[j];
            }
        }
    }

    void zero_grad(const std::vector<Tensor*>& grads) {
        for (auto* g : grads) g->fill(0.0f);
    }
};
