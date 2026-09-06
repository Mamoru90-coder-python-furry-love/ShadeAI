#pragma once
#include "../core/tensor.h"

struct BatchNorm {
    i32 dim;
    Tensor gamma, beta;
    Tensor ggamma, gbeta;
    Tensor last_x, last_norm, last_var;
    f32 momentum, eps;

    BatchNorm(i32 d, f32 mom = 0.9f) : dim(d), gamma(1, d, 1.0f), beta(1, d, 0.0f),
        ggamma(1, d, 0.0f), gbeta(1, d, 0.0f), momentum(mom), eps(1e-5f) {}

    Tensor forward(const Tensor& x) {
        last_x = x;
        i32 N = x.rows;
        // mean
        Tensor mean(1, dim, 0.0f);
        for (i32 i = 0; i < N; i++)
            for (i32 j = 0; j < dim; j++)
                mean.data[j] += x.at(i, j);
        for (i32 j = 0; j < dim; j++) mean.data[j] /= N;
        // var
        Tensor var(1, dim, 0.0f);
        for (i32 i = 0; i < N; i++)
            for (i32 j = 0; j < dim; j++) {
                f32 d = x.at(i, j) - mean.data[j];
                var.data[j] += d * d;
            }
        for (i32 j = 0; j < dim; j++) var.data[j] /= N;
        // normalize
        last_var = Tensor(1, dim);
        last_norm = Tensor(N, dim);
        for (i32 i = 0; i < N; i++)
            for (i32 j = 0; j < dim; j++) {
                last_var.data[j] = std::sqrt(var.data[j] + eps);
                last_norm.at(i, j) = (x.at(i, j) - mean.data[j]) / last_var.data[j];
            }
        // scale + shift
        Tensor out(N, dim);
        for (i32 i = 0; i < N; i++)
            for (i32 j = 0; j < dim; j++)
                out.at(i, j) = gamma.data[j] * last_norm.at(i, j) + beta.data[j];
        return out;
    }

    Tensor backward(const Tensor& dout) {
        i32 N = dout.rows;
        // dbeta, dgamma (accumulate)
        for (i32 i = 0; i < N; i++)
            for (i32 j = 0; j < dim; j++) {
                gbeta.data[j] += dout.at(i, j);
                ggamma.data[j] += dout.at(i, j) * last_norm.at(i, j);
            }
        // dx = (1/N) * gamma / var * (N * dout - sum(dout) - norm * sum(dout * norm))
        Tensor dx(N, dim);
        for (i32 i = 0; i < N; i++)
            for (i32 j = 0; j < dim; j++) {
                f32 sum_d = 0, sum_dn = 0;
                for (i32 k = 0; k < N; k++) {
                    sum_d += dout.at(k, j);
                    sum_dn += dout.at(k, j) * last_norm.at(k, j);
                }
                f32 val = dout.at(i, j) - sum_d / N - last_norm.at(i, j) * sum_dn / N;
                dx.at(i, j) = gamma.data[j] * val / last_var.data[j] / N;
            }
        return dx;
    }
};
