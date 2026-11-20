#pragma once
#include "../core/tensor.h"
#include "../core/simd.h"
#include <cmath>

struct SelfAttention {
    i32 embed_dim, num_heads, head_dim;
    bool causal;
    Tensor Wq, Wk, Wv, Wo;
    Tensor gWq, gWk, gWv, gWo;
    Tensor last_x, last_q, last_k, last_v, last_scores, last_softmax, last_out;
    std::vector<Tensor> head_out, head_softmax;

    SelfAttention(i32 emb, i32 heads, bool causal_ = false)
        : embed_dim(emb), num_heads(heads), head_dim(emb / heads), causal(causal_),
          Wq(emb, emb), Wk(emb, emb), Wv(emb, emb), Wo(emb, emb),
          gWq(emb, emb, 0.0f), gWk(emb, emb, 0.0f), gWv(emb, emb, 0.0f), gWo(emb, emb, 0.0f) {
        f32 scale = std::sqrt(2.0f / emb);
        auto init = [&](Tensor& t) {
            for (i32 i = 0; i < t.size(); i++)
                t.data[i] = ((f32)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        };
        init(Wq); init(Wk); init(Wv); init(Wo);
        for (i32 h = 0; h < num_heads; h++) { head_out.push_back(Tensor()); head_softmax.push_back(Tensor()); }
    }

    Tensor forward(const Tensor& x) {
        last_x = x;
        i32 T = x.rows;
        last_q = x.matmul(Wq);
        last_k = x.matmul(Wk);
        last_v = x.matmul(Wv);

        f32 scale = 1.0f / std::sqrt((f32)head_dim);
        last_scores = Tensor(T, T, 0.0f);
        last_softmax = Tensor(T, T, 0.0f);
        last_out = Tensor(T, embed_dim, 0.0f);

        for (i32 h = 0; h < num_heads; h++) {
            const f32* Qh = last_q.data.data() + (size_t)h * head_dim;
            const f32* Kh = last_k.data.data() + (size_t)h * head_dim;
            const f32* Vh = last_v.data.data() + (size_t)h * head_dim;
            f32* oh = last_out.data.data() + (size_t)h * head_dim;
            Tensor sm(T, T, 0.0f);
            f32* smp = sm.data.data();

            // K as T x head_dim and V^T as head_dim x T: both dot products become contiguous
            Tensor Kc(T, head_dim), Vt(head_dim, T);
            f32* kd = Kc.data.data();
            f32* vd = Vt.data.data();
            for (i32 i = 0; i < T; i++) {
                const f32* kr = Kh + (size_t)i * embed_dim;
                const f32* vr = Vh + (size_t)i * embed_dim;
                for (i32 c = 0; c < head_dim; c++) {
                    kd[(size_t)i * head_dim + c] = kr[c];
                    vd[(size_t)c * T + i] = vr[c];
                }
            }

#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (i32 i = 0; i < T; i++) {
                const f32* qrow = Qh + (size_t)i * embed_dim;
                f32* row = smp + (size_t)i * T;
                for (i32 j = 0; j < T; j++) {
                    f32 s = (causal && j > i) ? -1e30f
                          : simd::dot(qrow, kd + (size_t)j * head_dim, head_dim) * scale;
                    row[j] = s;
                    if (h == 0) last_scores.data[(size_t)i * T + j] = s;
                }
                simd::softmax_row(row, T);
                for (i32 c = 0; c < head_dim; c++)
                    oh[(size_t)i * embed_dim + c] = simd::dot(row, vd + (size_t)c * T, T);
            }
            head_softmax[h] = sm;
            if (h == 0) last_softmax = sm;
            head_out[h] = Tensor(T, head_dim);
            for (i32 i = 0; i < T; i++)
                for (i32 j = 0; j < head_dim; j++)
                    head_out[h].at(i, j) = oh[i * embed_dim + j];
        }

        sum_softmax();
        return last_out.matmul(Wo);
    }

    Tensor backward(const Tensor& dout) {
        i32 T = dout.rows;
        // gWo = last_out^T @ dout  (outer product, parallelized over i)
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (i32 i = 0; i < embed_dim; i++)
            for (i32 j = 0; j < embed_dim; j++) {
                f32 s = 0;
                for (i32 b = 0; b < T; b++)
                    s += last_out.data[b * embed_dim + i] * dout.data[b * embed_dim + j];
                gWo.data[i * embed_dim + j] += s;
            }

        // dO = dout @ Wo^T
        Tensor dO(T, embed_dim, 0.0f);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (i32 i = 0; i < T; i++)
            for (i32 j = 0; j < embed_dim; j++) {
                f32 s = 0;
                for (i32 k = 0; k < embed_dim; k++)
                    s += dout.data[i * embed_dim + k] * Wo.data[j * embed_dim + k];
                dO.data[i * embed_dim + j] = s;
            }

        Tensor dQ(T, embed_dim, 0.0f), dK(T, embed_dim, 0.0f), dV(T, embed_dim, 0.0f);
        f32 scale = 1.0f / std::sqrt((f32)head_dim);

        for (i32 h = 0; h < num_heads; h++) {
            const Tensor& sm = head_softmax[h];
            const f32* smp = sm.data.data();

            // dH = sm @ dO_per_head
            Tensor dH(T, embed_dim, 0.0f);
            for (i32 i = 0; i < T; i++)
                for (i32 c = 0; c < head_dim; c++) {
                    f32 s = 0;
                    i32 j = h * head_dim + c;
                    for (i32 k = 0; k < T; k++)
                        s += smp[i * T + k] * dO.data[k * embed_dim + j];
                    dH.data[i * embed_dim + j] = s;
                }

            // dS = dH @ V^T_per_head
            Tensor dS(T, T, 0.0f);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (i32 i = 0; i < T; i++)
                for (i32 j = 0; j < T; j++) {
                    f32 s = 0;
                    for (i32 k = 0; k < head_dim; k++)
                        s += dH.data[i * embed_dim + h * head_dim + k] * last_v.data[j * embed_dim + h * head_dim + k];
                    dS.data[i * T + j] = s;
                }

            // backprop through softmax (masked for causal)
            Tensor dScores(T, T, 0.0f);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (i32 i = 0; i < T; i++) {
                f32 dot = 0;
                i32 limit = causal ? i + 1 : T;
                for (i32 j = 0; j < limit; j++)
                    dot += smp[i * T + j] * dS.data[i * T + j];
                for (i32 j = 0; j < limit; j++)
                    dScores.data[i * T + j] = smp[i * T + j] * (dS.data[i * T + j] - dot);
            }

            // dQ += dScores @ K * scale,  dK += dScores^T @ Q * scale
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (i32 i = 0; i < T; i++)
                for (i32 c = 0; c < head_dim; c++) {
                    i32 j = h * head_dim + c;
                    f32 dq = 0, dk = 0;
                    for (i32 k = 0; k < T; k++) {
                        dq += dScores.data[i * T + k] * last_k.data[k * embed_dim + j];
                        dk += dScores.data[k * T + i] * last_q.data[k * embed_dim + j];
                    }
                    dQ.data[i * embed_dim + j] += dq * scale;
                    dK.data[i * embed_dim + j] += dk * scale;
                }

            // dV += sm^T @ dH_per_head
            Tensor dVh(T, head_dim, 0.0f);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (i32 i = 0; i < T; i++)
                for (i32 c = 0; c < head_dim; c++) {
                    f32 s = 0;
                    i32 j = h * head_dim + c;
                    for (i32 k = 0; k < T; k++)
                        s += smp[k * T + i] * dH.data[k * embed_dim + j];
                    dVh.data[i * head_dim + c] = s;
                }
            for (i32 i = 0; i < T; i++)
                for (i32 c = 0; c < head_dim; c++)
                    dV.data[i * embed_dim + h * head_dim + c] = dVh.data[i * head_dim + c];
        }

        // gWq = last_x^T @ dQ, gWk = last_x^T @ dK, gWv = last_x^T @ dV
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (i32 i = 0; i < embed_dim; i++)
            for (i32 j = 0; j < embed_dim; j++) {
                f32 sq = 0, sk = 0, sv = 0;
                for (i32 b = 0; b < T; b++) {
                    f32 xb = last_x.data[b * embed_dim + i];
                    sq += xb * dQ.data[b * embed_dim + j];
                    sk += xb * dK.data[b * embed_dim + j];
                    sv += xb * dV.data[b * embed_dim + j];
                }
                gWq.data[i * embed_dim + j] += sq;
                gWk.data[i * embed_dim + j] += sk;
                gWv.data[i * embed_dim + j] += sv;
            }

        // dx = dQ @ Wq^T + dK @ Wk^T + dV @ Wv^T
        Tensor dx(T, embed_dim, 0.0f);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (i32 i = 0; i < T; i++)
            for (i32 j = 0; j < embed_dim; j++) {
                f32 s = 0;
                for (i32 k = 0; k < embed_dim; k++)
                    s += dQ.data[i * embed_dim + k] * Wq.data[j * embed_dim + k]
                       + dK.data[i * embed_dim + k] * Wk.data[j * embed_dim + k]
                       + dV.data[i * embed_dim + k] * Wv.data[j * embed_dim + k];
                dx.data[i * embed_dim + j] = s;
            }
        return dx;
    }

    // debug: aggregated attention across all heads (for dashboard display)
    void sum_softmax() {
        i32 T = last_softmax.rows;
        if (T == 0) return;
        for (i32 i = 0; i < T; i++)
            for (i32 j = 0; j < T; j++)
                last_softmax.at(i, j) = 0.0f;
        for (i32 h = 0; h < num_heads; h++) {
            const Tensor& sm = head_softmax[h];
            for (i32 i = 0; i < T; i++)
                for (i32 j = 0; j < T; j++)
                    last_softmax.at(i, j) += sm.at(i, j);
        }
        for (i32 i = 0; i < T; i++)
            for (i32 j = 0; j < T; j++)
                last_softmax.at(i, j) /= (f32)num_heads;
    }

    std::vector<Tensor*> params() { return {&Wq, &Wk, &Wv, &Wo}; }
    std::vector<Tensor*> grads() { return {&gWq, &gWk, &gWv, &gWo}; }
};