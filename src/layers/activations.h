#pragma once
#include "../core/tensor.h"
#include <cmath>

enum class Activation { NONE, RELU, LEAKY_RELU, SIGMOID, TANH, GELU };

struct Act {
    Activation type;
    Tensor last_in;

    Act(Activation t = Activation::RELU) : type(t) {}

    Tensor forward(const Tensor& x) {
        last_in = x;
        Tensor out(x.rows, x.cols);
        const i32 n = out.size();
        switch (type) {
            case Activation::NONE:
                for (i32 i = 0; i < n; i++) out.data[i] = x.data[i];
                break;
            case Activation::RELU:
                for (i32 i = 0; i < n; i++) out.data[i] = x.data[i] < 0 ? 0 : x.data[i];
                break;
            case Activation::LEAKY_RELU:
                for (i32 i = 0; i < n; i++) out.data[i] = x.data[i] < 0 ? 0.01f * x.data[i] : x.data[i];
                break;
            case Activation::SIGMOID:
                for (i32 i = 0; i < n; i++) out.data[i] = 1.0f / (1.0f + std::exp(-x.data[i]));
                break;
            case Activation::TANH:
                for (i32 i = 0; i < n; i++) out.data[i] = std::tanh(x.data[i]);
                break;
            case Activation::GELU:
                for (i32 i = 0; i < n; i++) {
                    f32 v = x.data[i];
                    out.data[i] = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
                }
                break;
        }
        return out;
    }

    Tensor backward(const Tensor& dout) {
        Tensor dx(dout.rows, dout.cols);
        const i32 n = dx.size();
        const f32* li = last_in.data.data();
        const f32* do_ = dout.data.data();
        f32* dxp = dx.data.data();
        switch (type) {
            case Activation::NONE:
                for (i32 i = 0; i < n; i++) dxp[i] = do_[i];
                break;
            case Activation::RELU:
                for (i32 i = 0; i < n; i++) dxp[i] = li[i] <= 0 ? 0 : do_[i];
                break;
            case Activation::LEAKY_RELU:
                for (i32 i = 0; i < n; i++) dxp[i] = li[i] < 0 ? 0.01f * do_[i] : do_[i];
                break;
            case Activation::SIGMOID: {
                for (i32 i = 0; i < n; i++) {
                    f32 s = 1.0f / (1.0f + std::exp(-li[i]));
                    dxp[i] = do_[i] * s * (1 - s);
                }
                break;
            }
            case Activation::TANH: {
                for (i32 i = 0; i < n; i++) {
                    f32 t = std::tanh(li[i]);
                    dxp[i] = do_[i] * (1 - t * t);
                }
                break;
            }
            case Activation::GELU: {
                const f32 c = 0.7978845608f;
                for (i32 i = 0; i < n; i++) {
                    f32 v = li[i];
                    f32 v3 = v * v * v;
                    f32 tanh_v = std::tanh(c * (v + 0.044715f * v3));
                    f32 sech2 = 1.0f - tanh_v * tanh_v;
                    f32 dg = 0.5f * (1.0f + tanh_v) + 0.5f * v * sech2 * c * (1.0f + 0.134145f * v * v);
                    dxp[i] = do_[i] * dg;
                }
                break;
            }
        }
        return dx;
    }
};
