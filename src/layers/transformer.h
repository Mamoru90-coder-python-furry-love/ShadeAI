#pragma once
#include "../core/tensor.h"
#include "attention.h"
#include "layernorm.h"
#include "dense.h"
#include "activations.h"

struct PositionalEncoding {
    i32 max_len, dim;

    PositionalEncoding(i32 ml, i32 d) : max_len(ml), dim(d) {}

    Tensor forward(i32 T) {
        Tensor pe(T, dim);
        for (i32 i = 0; i < T; i++) {
            for (i32 j = 0; j < dim; j++) {
                f32 angle = (f32)i / std::pow(10000.0f, (f32)(j % 2) / dim);
                if (j % 2 == 0)
                    pe.at(i, j) = std::sin(angle);
                else
                    pe.at(i, j) = std::cos(angle);
            }
        }
        return pe;
    }
};

struct TransformerBlock {
    i32 embed_dim;
    SelfAttention attn;
    LayerNorm ln1, ln2;
    Dense ff1, ff2;
    Act act;
    Tensor last_x, last_attn_out, last_x1, last_y, last_ffn_out, last_x2;

    TransformerBlock(i32 emb)
        : embed_dim(emb), attn(emb, 4, true), ln1(emb), ln2(emb),
          ff1(emb, emb * 4, false), ff2(emb * 4, emb, false), act(Activation::GELU) {}

    Tensor forward(const Tensor& x) {
        last_x = x;
        last_attn_out = attn.forward(x);
        last_x1 = x.add(last_attn_out);
        last_y = ln1.forward(last_x1);

        Tensor ff1_out = ff1.forward(last_y);
        Tensor ff1_act = act.forward(ff1_out);
        last_ffn_out = ff2.forward(ff1_act);
        last_x2 = last_y.add(last_ffn_out);
        return ln2.forward(last_x2);
    }

    Tensor backward(const Tensor& dout) {
        Tensor dx2 = ln2.backward(dout);
        Tensor dy_skip = dx2;
        Tensor dffn = dx2;

        Tensor dffn2 = ff2.backward(dffn);
        Tensor dffn1_act = act.backward(dffn2);
        Tensor dy_ffn = ff1.backward(dffn1_act);

        Tensor dy_total(last_y.rows, last_y.cols);
        for (i32 i = 0; i < dy_total.size(); i++)
            dy_total.data[i] = dy_skip.data[i] + dy_ffn.data[i];

        Tensor dx1 = ln1.backward(dy_total);
        Tensor dx_skip = dx1;
        Tensor dattn = dx1;
        Tensor d_from_attn = attn.backward(dattn);

        Tensor dx_total(last_x.rows, last_x.cols);
        for (i32 i = 0; i < dx_total.size(); i++)
            dx_total.data[i] = dx_skip.data[i] + d_from_attn.data[i];
        return dx_total;
    }

    std::vector<Tensor*> params() {
        auto p = attn.params();
        p.push_back(&ln1.gamma); p.push_back(&ln1.beta);
        p.push_back(&ff1.w); p.push_back(&ff1.b);
        p.push_back(&ff2.w); p.push_back(&ff2.b);
        p.push_back(&ln2.gamma); p.push_back(&ln2.beta);
        return p;
    }
    std::vector<Tensor*> grads() {
        auto g = attn.grads();
        g.push_back(&ln1.ggamma); g.push_back(&ln1.gbeta);
        g.push_back(&ff1.gw); g.push_back(&ff1.gb);
        g.push_back(&ff2.gw); g.push_back(&ff2.gb);
        g.push_back(&ln2.ggamma); g.push_back(&ln2.gbeta);
        return g;
    }
};
