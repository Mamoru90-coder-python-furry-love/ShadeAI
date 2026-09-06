#pragma once
#include "../core/tensor.h"

struct Dropout {
    f32 rate;
    Tensor mask;
    bool training;

    Dropout(f32 r = 0.3f) : rate(r), training(true) {}

    void train(bool t) { training = t; }

    Tensor forward(const Tensor& x) {
        if (!training || rate <= 0.0f) return x;
        f32 scale = 1.0f / (1.0f - rate);
        mask = Tensor(x.rows, x.cols);
        for (i32 i = 0; i < mask.size(); i++)
            mask.data[i] = ((f32)rand() / RAND_MAX >= rate) ? scale : 0.0f;
        return x.emul(mask);
    }

    Tensor backward(const Tensor& dout) {
        if (!training || rate <= 0.0f) return dout;
        return dout.emul(mask);
    }
};
