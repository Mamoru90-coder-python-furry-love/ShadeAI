#pragma once
#include "../core/tensor.h"

struct LayerNorm {
    i32 dim;
    Tensor gamma, beta, ggamma, gbeta;
    Tensor last_x, last_mean, last_var;
    f32 eps;

    LayerNorm(i32 d, f32 ep = 1e-5f) : dim(d), gamma(1, d, 1.0f), beta(1, d, 0.0f),
        ggamma(1, d, 0.0f), gbeta(1, d, 0.0f), eps(ep) {
        if (d <= 0 || ep <= 0.0f) throw std::invalid_argument("LayerNorm: dim and eps must be positive");
    }

    Tensor forward(const Tensor& x) {
        if (x.cols != dim) throw std::invalid_argument("LayerNorm::forward: input width != dim");
        last_x = x;
        i32 B = x.rows;
        last_mean = Tensor(B, 1, 0.0f);
        last_var = Tensor(B, 1, 0.0f);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (i32 i = 0; i < B; i++) {
            const f32* xp = &x.data[i * dim];
            f32 sum = 0;
            for (i32 j = 0; j < dim; j++) sum += xp[j];
            f32 mean = sum / dim;
            last_mean.data[i] = mean;
            f32 var = 0;
            for (i32 j = 0; j < dim; j++) { f32 d = xp[j] - mean; var += d * d; }
            last_var.data[i] = var / dim;
        }
        Tensor out(B, dim);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (i32 i = 0; i < B; i++) {
            const f32* xp = &x.data[i * dim];
            f32* op = out.data.data() + i * dim;
            f32 inv_std = 1.0f / sqrtf(last_var.data[i] + eps);
            f32 mean = last_mean.data[i];
            i32 j = 0;
#if defined(__AVX__) || defined(__AVX2__)
            i32 jlim = dim & ~7;
            for (; j < jlim; j += 8) {
                __m256 xv = _mm256_loadu_ps(xp + j);
                __m256 gv = _mm256_loadu_ps(gamma.data.data() + j);
                __m256 bv = _mm256_loadu_ps(beta.data.data() + j);
                __m256 mv = _mm256_set1_ps(mean);
                __m256 iv = _mm256_set1_ps(inv_std);
                __m256 norm = _mm256_mul_ps(_mm256_sub_ps(xv, mv), iv);
                __m256 res = _mm256_add_ps(_mm256_mul_ps(gv, norm), bv);
                _mm256_storeu_ps(op + j, res);
            }
#endif
            for (; j < dim; j++)
                op[j] = gamma.data[j] * (xp[j] - mean) * inv_std + beta.data[j];
        }
        return out;
    }

    Tensor backward(const Tensor& dout) {
        if (dout.rows != last_x.rows || dout.cols != dim)
            throw std::invalid_argument("LayerNorm::backward: dout shape does not match forward output");
        i32 B = dout.rows;
        ggamma.fill(0.0f); gbeta.fill(0.0f);
        Tensor dx(B, dim, 0.0f);
        for (i32 i = 0; i < B; i++) {
            const f32* dp = &dout.data[i * dim];
            const f32* xp = &last_x.data[i * dim];
            f32* dxp = &dx.data[i * dim];
            f32 std = sqrtf(last_var.data[i] + eps);
            f32 mean = last_mean.data[i];
            // dL/dx = (gamma/std) * (dout - mean(dout*gamma) - xhat*mean(dout*gamma*xhat)).
            f32 gsum = 0, gdsum = 0;
            for (i32 j = 0; j < dim; j++) {
                f32 x_hat = (xp[j] - mean) / std;
                gbeta.data[j] += dp[j];
                ggamma.data[j] += dp[j] * x_hat;
                gsum += dp[j] * gamma.data[j];
                gdsum += dp[j] * gamma.data[j] * x_hat;
            }
            f32 a = gsum / dim, b = gdsum / dim;
            for (i32 j = 0; j < dim; j++) {
                f32 x_hat = (xp[j] - mean) / std;
                dxp[j] = (gamma.data[j] * dp[j] - a - x_hat * b) / std;
            }
        }
        return dx;
    }
};
