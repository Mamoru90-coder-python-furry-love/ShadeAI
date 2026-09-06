#pragma once
#include "tensor.h"
#include <cmath>

struct CrossEntropyLoss {
    // softmax + cross entropy, returns dlogits for backward
    static f32 forward(const Tensor& logits, const Tensor& target, Tensor& dlogits) {
        i32 n = logits.rows;
        dlogits = Tensor(logits.rows, logits.cols);
        f32 loss = 0;
        for (i32 i = 0; i < n; i++) {
            f32 maxv = logits.data[i * logits.cols];
            for (i32 j = 1; j < logits.cols; j++)
                if (logits.data[i * logits.cols + j] > maxv)
                    maxv = logits.data[i * logits.cols + j];
            f32 sum = 0;
            for (i32 j = 0; j < logits.cols; j++) {
                f32 e = std::exp(logits.data[i * logits.cols + j] - maxv);
                dlogits.data[i * logits.cols + j] = e;
                sum += e;
            }
            for (i32 j = 0; j < logits.cols; j++) {
                f32 p = dlogits.data[i * logits.cols + j] / sum;
                dlogits.data[i * logits.cols + j] = p - target.data[i * logits.cols + j];
                loss -= target.data[i * logits.cols + j] * std::log(std::max(p, 1e-7f));
            }
        }
        return loss / n;
    }
};
