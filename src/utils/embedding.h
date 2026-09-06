#pragma once
#include "../core/tensor.h"

struct Embedding {
    i32 vocab_size, dim;
    Tensor w, gw;

    Embedding(i32 vs, i32 d) : vocab_size(vs), dim(d), w(vs, d), gw(vs, d, 0.0f) {
        f32 scale = 0.1f;
        for (i32 i = 0; i < w.size(); i++)
            w.data[i] = ((f32)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }

    // ids: (seq_len,) - row vector of token indices
    // returns: (seq_len, dim)
    Tensor forward(const Tensor& ids) {
        i32 seq = ids.cols; // ids is (1, seq_len)
        Tensor out(seq, dim);
        for (i32 i = 0; i < seq; i++) {
            i32 id = (i32)ids.data[i];
            for (i32 j = 0; j < dim; j++)
                out.at(i, j) = w.at(id, j);
        }
        return out;
    }

    // dout: (seq_len, dim)
    // accumulates gradients into gw
    Tensor backward(const Tensor& dout, const Tensor& ids) {
        i32 seq = ids.cols;
        for (i32 i = 0; i < seq; i++) {
            i32 id = (i32)ids.data[i];
            for (i32 j = 0; j < dim; j++)
                gw.at(id, j) += dout.at(i, j);
        }
        // no gradient to pass further (input is discrete indices)
        return Tensor(0, 0);
    }
};
