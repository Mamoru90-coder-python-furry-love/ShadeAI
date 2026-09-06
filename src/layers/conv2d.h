#pragma once
#include "../core/tensor.h"
#include <cmath>

struct Conv2D {
    i32 in_c, out_c, kH, kW, stride, pad;
    Tensor w, b, gw, gb;
    i32 oH, oW;
    Tensor last_x, last_col;
    i32 last_N, last_H, last_W;

    Conv2D(i32 ic, i32 oc, i32 kh, i32 kw, i32 s = 1, i32 p = 0)
        : in_c(ic), out_c(oc), kH(kh), kW(kw), stride(s), pad(p),
          w(oc, ic * kh * kw), b(1, oc, 0.0f),
          gw(oc, ic * kh * kw, 0.0f), gb(1, oc, 0.0f) {
        f32 scale = sqrtf(2.0f / (ic * kh * kw));
        for (i32 i = 0; i < w.size(); i++)
            w.data[i] = ((f32)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }

    void out_shape(i32 H, i32 W, i32& oh, i32& ow) const {
        oh = (H + 2 * pad - kH) / stride + 1;
        ow = (W + 2 * pad - kW) / stride + 1;
    }

    // im2col: x(N, C*H*W) → col(N*oH*oW, C*kH*kW)
    void do_im2col(const Tensor& x, Tensor& col, i32 N, i32 C, i32 H, i32 W, i32 oh, i32 ow) const {
        i32 K = C * kH * kW;
        i32 M = N * oh * ow;
        col = Tensor(M, K);
        for (i32 n = 0; n < N; n++) {
            for (i32 oy = 0; oy < oh; oy++) {
                for (i32 ox = 0; ox < ow; ox++) {
                    i32 row = (n * oh + oy) * ow + ox;
                    i32 ci = 0;
                    for (i32 c = 0; c < C; c++) {
                        for (i32 ky = 0; ky < kH; ky++) {
                            for (i32 kx = 0; kx < kW; kx++) {
                                i32 iy = oy * stride + ky - pad;
                                i32 ix = ox * stride + kx - pad;
                                f32 v = 0.0f;
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                    v = x.at(n, c * H * W + iy * W + ix);
                                col.at(row, ci++) = v;
                            }
                        }
                    }
                }
            }
        }
    }

    // col2im: col(M, C*kH*kW) → dx(N, C*H*W)
    void do_col2im(const Tensor& col, Tensor& dx, i32 N, i32 C, i32 H, i32 W, i32 oh, i32 ow) const {
        i32 K = C * kH * kW;
        dx = Tensor(N, C * H * W, 0.0f);
        for (i32 n = 0; n < N; n++) {
            for (i32 oy = 0; oy < oh; oy++) {
                for (i32 ox = 0; ox < ow; ox++) {
                    i32 row = (n * oh + oy) * ow + ox;
                    i32 ci = 0;
                    for (i32 c = 0; c < C; c++) {
                        for (i32 ky = 0; ky < kH; ky++) {
                            for (i32 kx = 0; kx < kW; kx++) {
                                i32 iy = oy * stride + ky - pad;
                                i32 ix = ox * stride + kx - pad;
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                    dx.at(n, c * H * W + iy * W + ix) += col.at(row, ci);
                                ci++;
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor forward(const Tensor& x) {
        last_x = x;
        last_N = x.rows;
        if (last_H == 0) { printf("ERROR: Conv2D: set spatial dims\n"); return x; }
        i32 oh, ow;
        out_shape(last_H, last_W, oh, ow);
        oH = oh; oW = ow;
        i32 K = in_c * kH * kW;
        i32 M = last_N * oh * ow;

        do_im2col(x, last_col, last_N, in_c, last_H, last_W, oh, ow);
        // w: (out_c, K)  w^T: (K, out_c)
        // col(M,K) * wT(K, out_c) = out_2d(M, out_c)
        Tensor wT = w.T();
        Tensor out_2d = last_col.matmul(wT);

        Tensor out(last_N, out_c * oh * ow);
        for (i32 n = 0; n < last_N; n++)
            for (i32 oc = 0; oc < out_c; oc++)
                for (i32 oy = 0; oy < oh; oy++)
                    for (i32 ox = 0; ox < ow; ox++)
                        out.at(n, oc * oh * ow + oy * ow + ox) =
                            out_2d.at(n * oh * ow + oy * ow + ox, oc) + b.data[oc];
        return out;
    }

    Tensor backward(const Tensor& dout) {
        i32 oh = oH, ow = oW;
        i32 K = in_c * kH * kW;
        i32 M = last_N * oh * ow;
        i32 N = last_N;

        // reshape dout(N, out_c*oh*ow) → dout_2d(M, out_c)
        Tensor dout_2d(M, out_c);
        for (i32 n = 0; n < N; n++)
            for (i32 oc = 0; oc < out_c; oc++)
                for (i32 oy = 0; oy < oh; oy++)
                    for (i32 ox = 0; ox < ow; ox++)
                        dout_2d.at(n * oh * ow + oy * ow + ox, oc) =
                            dout.at(n, oc * oh * ow + oy * ow + ox);

        // gb: sum dout_2d over all rows (n, oy, ox)
        for (i32 oc = 0; oc < out_c; oc++) {
            f32 s = 0;
            for (i32 i = 0; i < M; i++) s += dout_2d.at(i, oc);
            gb.data[oc] += s;
        }

        // gw: dout_2d^T * last_col → (out_c, M) * (M, K) = (out_c, K)
        Tensor dout_T = dout_2d.T();
        Tensor gw_new = dout_T.matmul(last_col);
        for (i32 i = 0; i < gw.size(); i++)
            gw.data[i] += gw_new.data[i];

        // dx: dout_2d * w → (M, out_c) * (out_c, K) = (M, K) → col2im
        Tensor dx_2d = dout_2d.matmul(w);
        Tensor dx;
        do_col2im(dx_2d, dx, N, in_c, last_H, last_W, oh, ow);
        return dx;
    }

    void set_spatial(i32 H, i32 W) { last_H = H; last_W = W; }

    std::vector<Tensor*> params() { return {&w, &b}; }
    std::vector<Tensor*> grads() { return {&gw, &gb}; }
};
