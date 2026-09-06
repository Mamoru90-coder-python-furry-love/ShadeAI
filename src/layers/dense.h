#pragma once
#include "../core/tensor.h"

struct Dense {
    i32 in_dim, out_dim;
    bool use_relu;
    Tensor w, b;       // weights
    Tensor gw, gb;     // gradients
    Tensor wT;         // cached transpose for backward
    Tensor last_input, last_output;

    Dense(i32 in_d, i32 out_d, bool relu = false)
        : in_dim(in_d), out_dim(out_d), use_relu(relu),
          w(in_d, out_d), b(1, out_d, 0.0f),
          gw(in_d, out_d, 0.0f), gb(1, out_d, 0.0f),
          wT(out_d, in_d, 0.0f) {
        // Xavier init
        f32 scale = std::sqrt(2.0f / (in_d + out_d));
        for (i32 i = 0; i < w.size(); i++)
            w.data[i] = ((f32)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        for (i32 i = 0; i < in_d; i++)
            for (i32 j = 0; j < out_d; j++)
                wT.data[j * in_d + i] = w.data[i * out_d + j];
    }

    Tensor forward(const Tensor& x) {
        last_input = x;
        last_output = x.matmul(w).add(b);
        if (use_relu)
            for (i32 i = 0; i < last_output.size(); i++)
                if (last_output.data[i] < 0) last_output.data[i] = 0;
        return last_output;
    }

    Tensor backward(const Tensor& dout) {
        // refresh cached transpose from current w (w is not modified during backward)
        for (i32 i = 0; i < in_dim; i++) {
            const f32* wp = &w.data[i * out_dim];
            for (i32 j = 0; j < out_dim; j++)
                wT.data[j * in_dim + i] = wp[j];
        }
        Tensor d = dout;
        if (use_relu)
            for (i32 i = 0; i < d.size(); i++)
                if (last_output.data[i] <= 0) d.data[i] = 0;

        // gw = last_input.T * d (accumulate)
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (i32 i = 0; i < in_dim; i++)
            for (i32 j = 0; j < out_dim; j++) {
                f32 sum = 0;
                const f32* lp = &last_input.data[i];
                const f32* dp = &d.data[j];
                for (i32 b = 0; b < last_input.rows; b++)
                    sum += lp[b * in_dim] * dp[b * out_dim];
                gw.data[i * out_dim + j] += sum;
            }
        // gb = sum over batch
        {
            const i32 B = d.rows;
            for (i32 j = 0; j < out_dim; j++) {
                f32 s = 0;
                const f32* dp = &d.data[j];
                for (i32 b = 0; b < B; b++) s += dp[b * out_dim];
                gb.data[j] += s;
            }
        }

        // dx = d * w.T (uses cached transpose, no per-call allocation)
        return d.matmul(wT);
    }
};
