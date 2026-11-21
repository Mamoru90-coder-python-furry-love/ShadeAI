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
        // Caller contract: validate the WHOLE set before touching anything.
        // An invalid combination is a no-op — no parameter, moment or step
        // counter is modified. Previously grads[i]/m[i]/v[i] were indexed by
        // params.size() with no checks, so a short vector meant out-of-bounds
        // access (undefined behaviour) and a long one silently ignored data.
        const size_t np = params.size();
        if (np == 0 || grads.size() != np || m.size() != np || v.size() != np) return;
        // rows/cols are public and so is data — a caller can resize data
        // without updating the shape. The kernel iterates params[i]->size()
        // (i.e. data.size()), so shape agreement alone is not enough.
        auto data_matches_shape = [](const Tensor& t) {
            if (t.rows < 0 || t.cols < 0) return false;
            return t.data.size() == static_cast<size_t>(t.rows) * static_cast<size_t>(t.cols);
        };
        for (size_t i = 0; i < np; ++i) {
            const Tensor* p = params[i];
            const Tensor* g = grads[i];
            if (!p || !g) return;
            if (p->rows != g->rows || p->cols != g->cols) return;
            if (p->rows != m[i].rows || p->cols != m[i].cols) return;
            if (p->rows != v[i].rows || p->cols != v[i].cols) return;
            if (!data_matches_shape(*p) || !data_matches_shape(*g) ||
                !data_matches_shape(m[i]) || !data_matches_shape(v[i])) return;
            const size_t want = p->data.size();
            if (!fits_i32(want)) return; // element loops use i32; reject before t++
            if (g->data.size() != want || m[i].data.size() != want ||
                v[i].data.size() != want) return;
        }
        t++;
        // bias-correction factors + fused step coefficients — computed ONCE per step
        f32 p1t = std::pow(beta1, t);
        f32 p2t = std::pow(beta2, t);
        f32 ib1 = 1.0f / (1.0f - p1t);
        f32 ib2 = 1.0f / (1.0f - p2t);
        // m_hat = m*ib1, v_hat = v*ib2; decay is decoupled and never bias-corrected.
        // CRITICAL ORDER: decay multiplies the OLD parameter p_t; the adaptive
        // step is subtracted afterwards and must NOT be scaled by (1 - lr*wd).
        // Applying decay to (p_t - step) shrinks the gradient step itself and is
        // not AdamW (negative control: tests/adamw_checks.cpp).
        f32 km = lr * ib1;
        f32 kw = lr * wd;
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
                __m256 np = _mm256_sub_ps(pv_, _mm256_mul_ps(kwv, pv_)); // decay p_t
                np = _mm256_sub_ps(np, _mm256_div_ps(_mm256_mul_ps(kmv, mi), den));
                simd::store8(p + j, np);
            }
#endif
            for (; j < n; j++) {
                f32 gi = g[j];
                f32 mi = beta1 * pm[j] + (1 - beta1) * gi;
                f32 vi = beta2 * pv[j] + (1 - beta2) * gi * gi;
                pm[j] = mi; pv[j] = vi;
                const f32 old_p = p[j];
                p[j] = old_p - kw * old_p; // decay old parameter first
                p[j] -= km * mi / (std::sqrt(vi * ib2) + eps);
            }
        }
    }

    void zero_grad(const std::vector<Tensor*>& grads) {
        for (auto* g : grads) g->fill(0.0f);
    }
};
