#pragma once
#include "../core/tensor.h"

struct GRU {
    i32 in_dim, hidden_dim;
    // weight matrices: W = input weights, U = hidden weights
    // z = sigmoid(Wz * x + Uz * h), r = sigmoid(Wr * x + Ur * h), n = tanh(Wn * x + r * (Un * h))
    Tensor Wz, Wr, Wn, Uz, Ur, Un;
    Tensor bz, br, bn; // biases
    Tensor gWz, gWr, gWn, gUz, gUr, gUn, gbz, gbr, gbn;

    // cache for backward
    Tensor last_x, last_h0;
    std::vector<Tensor> xs, zs, rs, ns, hs;

    GRU(i32 in_d, i32 hid_d)
        : in_dim(in_d), hidden_dim(hid_d),
          Wz(in_d, hid_d), Wr(in_d, hid_d), Wn(in_d, hid_d),
          Uz(hid_d, hid_d), Ur(hid_d, hid_d), Un(hid_d, hid_d),
          bz(1, hid_d, 0.0f), br(1, hid_d, 0.0f), bn(1, hid_d, 0.0f),
          gWz(in_d, hid_d, 0.0f), gWr(in_d, hid_d, 0.0f), gWn(in_d, hid_d, 0.0f),
          gUz(hid_d, hid_d, 0.0f), gUr(hid_d, hid_d, 0.0f), gUn(hid_d, hid_d, 0.0f),
          gbz(1, hid_d, 0.0f), gbr(1, hid_d, 0.0f), gbn(1, hid_d, 0.0f) {
        f32 scale = std::sqrt(2.0f / (in_d + hid_d));
        auto init = [&](Tensor& t) {
            for (i32 i = 0; i < t.size(); i++)
                t.data[i] = ((f32)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        };
        init(Wz); init(Wr); init(Wn);
        init(Uz); init(Ur); init(Un);
    }

    Tensor forward(const Tensor& x, const Tensor& h0) {
        last_x = x; last_h0 = h0;
        i32 T = x.rows;
        xs.clear(); zs.clear(); rs.clear(); ns.clear(); hs.clear();
        hs.push_back(h0);

        for (i32 t = 0; t < T; t++) {
            // x_t is a single row, but our matmul works on full matrix — extract row
            Tensor xt(1, in_dim);
            for (i32 j = 0; j < in_dim; j++)
                xt.data[j] = x.at(t, j);
            xs.push_back(xt);

            Tensor h_prev = hs.back();

            // z = sigmoid(Wz * xt + Uz * h_prev + bz)
            Tensor zt = xt.matmul(Wz).add(h_prev.matmul(Uz)).add(bz);
            for (i32 j = 0; j < hidden_dim; j++)
                zt.data[j] = 1.0f / (1.0f + std::exp(-zt.data[j]));
            zs.push_back(zt);

            // r = sigmoid(Wr * xt + Ur * h_prev + br)
            Tensor rt = xt.matmul(Wr).add(h_prev.matmul(Ur)).add(br);
            for (i32 j = 0; j < hidden_dim; j++)
                rt.data[j] = 1.0f / (1.0f + std::exp(-rt.data[j]));
            rs.push_back(rt);

            // n = tanh(Wn * xt + rt * (Un * h_prev) + bn)
            Tensor nt = xt.matmul(Wn).add(rt.emul(h_prev.matmul(Un))).add(bn);
            for (i32 j = 0; j < hidden_dim; j++)
                nt.data[j] = std::tanh(nt.data[j]);
            ns.push_back(nt);

            // h = (1 - z) * n + z * h_prev
            Tensor ht(1, hidden_dim);
            for (i32 j = 0; j < hidden_dim; j++)
                ht.data[j] = (1.0f - zt.data[j]) * nt.data[j] + zt.data[j] * h_prev.data[j];
            hs.push_back(ht);
        }
        // return all hidden states as T x hidden_dim
        Tensor out(T, hidden_dim);
        for (i32 t = 0; t < T; t++)
            for (i32 j = 0; j < hidden_dim; j++)
                out.at(t, j) = hs[t+1].data[j];
        return out;
    }

    Tensor backward(const Tensor& dh) {
        // dh: T x hidden_dim, gradients for each output timestep
        // Gradients accumulate (zero_grad before epoch)

        i32 T = dh.rows;
        Tensor dh_next(1, hidden_dim, 0.0f);
        Tensor dx(T, in_dim, 0.0f);

        for (i32 t = T - 1; t >= 0; t--) {
            // dh_t from upstream + from next timestep
            Tensor dht(1, hidden_dim);
            for (i32 j = 0; j < hidden_dim; j++)
                dht.data[j] = dh.at(t, j) + dh_next.data[j];

            Tensor h_prev = hs[t];  // h_{t-1}
            Tensor zt = zs[t], rt = rs[t], nt = ns[t], ht = hs[t+1];

            // dh = (1 - z) * dn + z * dh_prev + dh * (h_prev - n) * dz + dh * prev * dr
            // but simplified:
            // dh/dn = 1-z,  dh/dh_prev = z,  dh/dz = h_prev - n
            // d/dh_prev:
            // dht = dh/dh_prev * dht + dh/dz * dht * dz/dh_prev + dh/dn * dht * dn/dh_prev + dh/dr * dht * dr/dh_prev
            //
            // Let's do step by step:

            // dz = dht * (h_prev - n) * z * (1 - z)
            Tensor dz(1, hidden_dim);
            for (i32 j = 0; j < hidden_dim; j++)
                dz.data[j] = dht.data[j] * (h_prev.data[j] - nt.data[j])
                            * zt.data[j] * (1.0f - zt.data[j]);

            // dr = dht * (1 - z) * (1 - n^2) * (U * h_prev) * r * (1 - r)
            // First, Un * h_prev
            Tensor Uh(1, hidden_dim);
            for (i32 j = 0; j < hidden_dim; j++) {
                f32 s = 0;
                for (i32 k = 0; k < hidden_dim; k++)
                    s += Un.at(k, j) * h_prev.data[k];
                Uh.data[j] = s;
            }
            Tensor dr(1, hidden_dim);
            for (i32 j = 0; j < hidden_dim; j++)
                dr.data[j] = dht.data[j] * (1.0f - zt.data[j]) * (1.0f - nt.data[j] * nt.data[j])
                            * Uh.data[j] * rt.data[j] * (1.0f - rt.data[j]);

            // dn = dht * (1 - z) * (1 - n^2)
            Tensor dn(1, hidden_dim);
            for (i32 j = 0; j < hidden_dim; j++)
                dn.data[j] = dht.data[j] * (1.0f - zt.data[j]) * (1.0f - nt.data[j] * nt.data[j]);

            // gradients for gates
            // Wz, Uz, bz
            for (i32 j = 0; j < hidden_dim; j++)
                for (i32 k = 0; k < in_dim; k++)
                    gWz.at(k, j) += dz.data[j] * xs[t].data[k];
            for (i32 j = 0; j < hidden_dim; j++)
                for (i32 k = 0; k < hidden_dim; k++)
                    gUz.at(k, j) += dz.data[j] * h_prev.data[k];
            for (i32 j = 0; j < hidden_dim; j++)
                gbz.data[j] += dz.data[j];

            // Wr, Ur, br
            for (i32 j = 0; j < hidden_dim; j++)
                for (i32 k = 0; k < in_dim; k++)
                    gWr.at(k, j) += dr.data[j] * xs[t].data[k];
            for (i32 j = 0; j < hidden_dim; j++)
                for (i32 k = 0; k < hidden_dim; k++)
                    gUr.at(k, j) += dr.data[j] * h_prev.data[k];
            for (i32 j = 0; j < hidden_dim; j++)
                gbr.data[j] += dr.data[j];

            // Wn, Un, bn
            for (i32 j = 0; j < hidden_dim; j++)
                for (i32 k = 0; k < in_dim; k++)
                    gWn.at(k, j) += dn.data[j] * xs[t].data[k];
            for (i32 j = 0; j < hidden_dim; j++)
                for (i32 k = 0; k < hidden_dim; k++)
                    gUn.at(k, j) += dn.data[j] * rt.data[j] * h_prev.data[k];
            for (i32 j = 0; j < hidden_dim; j++)
                gbn.data[j] += dn.data[j];

            // dh_next for next iteration
            for (i32 j = 0; j < hidden_dim; j++) {
                dh_next.data[j] = dht.data[j] * zt.data[j]; // from z * h_prev
                // + from dz * Wz ...
                f32 sum_z = 0, sum_r = 0, sum_n = 0;
                for (i32 k = 0; k < hidden_dim; k++) {
                    sum_z += dz.data[k] * Uz.at(j, k);
                    sum_r += dr.data[k] * Ur.at(j, k);
                    sum_n += dn.data[k] * Un.at(j, k) * rt.data[k];
                }
                dh_next.data[j] += sum_z + sum_r + sum_n;
            }

            // dx for this timestep
            for (i32 j = 0; j < in_dim; j++) {
                f32 s = 0;
                for (i32 k = 0; k < hidden_dim; k++)
                    s += dz.data[k] * Wz.at(j, k) + dr.data[k] * Wr.at(j, k) + dn.data[k] * Wn.at(j, k);
                dx.at(t, j) = s;
            }
        }

        return dx;
    }

    std::vector<Tensor*> params() {
        return {&Wz, &Wr, &Wn, &Uz, &Ur, &Un, &bz, &br, &bn};
    }

    std::vector<Tensor*> grads() {
        return {&gWz, &gWr, &gWn, &gUz, &gUr, &gUn, &gbz, &gbr, &gbn};
    }
};
