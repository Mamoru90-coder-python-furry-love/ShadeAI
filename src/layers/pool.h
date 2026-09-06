#pragma once
#include "../core/tensor.h"

// MaxPool2D: (N, in_c*H*W) → (N, in_c*oH*oW)
struct MaxPool2D {
    i32 kH, kW, stride, pad;
    Tensor last_x, last_mask;
    i32 last_N, last_C, last_H, last_W, oH, oW;

    MaxPool2D(i32 kh = 2, i32 kw = 2, i32 s = 2, i32 p = 0)
        : kH(kh), kW(kw), stride(s), pad(p) {}

    Tensor forward(const Tensor& x) {
        last_x = x;
        last_N = x.rows;
        // infer C, H, W from x.cols (must be set externally)
        oH = (last_H + 2 * pad - kH) / stride + 1;
        oW = (last_W + 2 * pad - kW) / stride + 1;
        Tensor out(last_N, last_C * oH * oW, 0.0f);
        last_mask = Tensor(x.rows, x.cols, -1.0f); // stores which pool region max came from

        for (i32 n = 0; n < last_N; n++) {
            for (i32 c = 0; c < last_C; c++) {
                for (i32 oy = 0; oy < oH; oy++) {
                    for (i32 ox = 0; ox < oW; ox++) {
                        f32 maxv = -1e30f;
                        i32 max_iy = 0, max_ix = 0;
                        for (i32 ky = 0; ky < kH; ky++) {
                            for (i32 kx = 0; kx < kW; kx++) {
                                i32 iy = oy * stride + ky - pad;
                                i32 ix = ox * stride + kx - pad;
                                if (iy < 0 || iy >= last_H || ix < 0 || ix >= last_W) continue;
                                f32 v = x.at(n, c * last_H * last_W + iy * last_W + ix);
                                if (v > maxv) { maxv = v; max_iy = iy; max_ix = ix; }
                            }
                        }
                        out.at(n, c * oH * oW + oy * oW + ox) = maxv;
                        last_mask.at(n, c * last_H * last_W + max_iy * last_W + max_ix) = 1.0f;
                    }
                }
            }
        }
        return out;
    }

    // dout: (N, C*oH*oW), returns: (N, C*H*W)
    Tensor backward(const Tensor& dout) {
        Tensor dx(last_N, last_C * last_H * last_W, 0.0f);
        for (i32 n = 0; n < last_N; n++) {
            for (i32 c = 0; c < last_C; c++) {
                for (i32 oy = 0; oy < oH; oy++) {
                    for (i32 ox = 0; ox < oW; ox++) {
                        f32 d = dout.at(n, c * oH * oW + oy * oW + ox);
                        for (i32 ky = 0; ky < kH; ky++) {
                            for (i32 kx = 0; kx < kW; kx++) {
                                i32 iy = oy * stride + ky - pad;
                                i32 ix = ox * stride + kx - pad;
                                if (iy < 0 || iy >= last_H || ix < 0 || ix >= last_W) continue;
                                if (last_mask.at(n, c * last_H * last_W + iy * last_W + ix) == 1.0f)
                                    dx.at(n, c * last_H * last_W + iy * last_W + ix) = d;
                            }
                        }
                    }
                }
            }
        }
        return dx;
    }

    void set_spatial(i32 C, i32 H, i32 W) { last_C = C; last_H = H; last_W = W; }
};
