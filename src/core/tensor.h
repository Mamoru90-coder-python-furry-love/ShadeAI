#pragma once
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <algorithm>
#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

using f32 = float;
using i32 = int32_t;

struct Tensor {
    i32 rows, cols;
    std::vector<f32> data;

    Tensor() : rows(0), cols(0) {}
    Tensor(i32 r, i32 c, f32 v = 0.0f) : rows(r), cols(c), data(r * c, v) {}
    Tensor(i32 r, i32 c, const std::vector<f32>& d) : rows(r), cols(c), data(d) {}

    f32& at(i32 r, i32 c) { return data[r * cols + c]; }
    f32  at(i32 r, i32 c) const { return data[r * cols + c]; }
    i32 size() const { return (i32)data.size(); }

    void fill(f32 v) { std::fill(data.begin(), data.end(), v); }
    Tensor clone() const { return Tensor(rows, cols, data); }
    Tensor slice(i32 start, i32 count) const {
        Tensor r(count, cols);
        for (i32 i = 0; i < count; i++)
            for (i32 j = 0; j < cols; j++)
                r.at(i, j) = at(start + i, j);
        return r;
    }

    void rand_uniform() {
        for (i32 i = 0; i < size(); i++)
            data[i] = ((f32)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    Tensor emul(const Tensor& o) const {
        Tensor r(rows, cols);
        if (o.rows == rows && o.cols == cols) {
            for (i32 i = 0; i < size(); i++) r.data[i] = data[i] * o.data[i];
        } else if (o.rows == 1 && o.cols == cols) {
            for (i32 i = 0; i < rows; i++)
                for (i32 j = 0; j < cols; j++)
                    r.at(i, j) = at(i, j) * o.data[j];
        } else if (o.rows == rows && o.cols == 1) {
            for (i32 i = 0; i < rows; i++)
                for (i32 j = 0; j < cols; j++)
                    r.at(i, j) = at(i, j) * o.data[i];
        } else {
            printf("ERROR: emul shape mismatch (%dx%d * %dx%d)\n", rows, cols, o.rows, o.cols);
        }
        return r;
    }

    Tensor add(const Tensor& o) const {
        Tensor r(rows, cols);
        if (o.rows == rows && o.cols == cols) {
            for (i32 i = 0; i < size(); i++) r.data[i] = data[i] + o.data[i];
        } else if (o.rows == 1 && o.cols == cols) {
            for (i32 i = 0; i < rows; i++)
                for (i32 j = 0; j < cols; j++)
                    r.at(i, j) = at(i, j) + o.data[j];
        } else if (o.rows == rows && o.cols == 1) {
            for (i32 i = 0; i < rows; i++)
                for (i32 j = 0; j < cols; j++)
                    r.at(i, j) = at(i, j) + o.data[i];
        } else {
            printf("ERROR: add shape mismatch (%dx%d + %dx%d)\n", rows, cols, o.rows, o.cols);
        }
        return r;
    }

    Tensor sub(const Tensor& o) const {
        Tensor r(rows, cols);
        if (o.rows == rows && o.cols == cols) {
            for (i32 i = 0; i < size(); i++) r.data[i] = data[i] - o.data[i];
        } else if (o.rows == 1 && o.cols == cols) {
            for (i32 i = 0; i < rows; i++)
                for (i32 j = 0; j < cols; j++)
                    r.at(i, j) = at(i, j) - o.data[j];
        } else if (o.rows == rows && o.cols == 1) {
            for (i32 i = 0; i < rows; i++)
                for (i32 j = 0; j < cols; j++)
                    r.at(i, j) = at(i, j) - o.data[i];
        } else {
            printf("ERROR: sub shape mismatch (%dx%d - %dx%d)\n", rows, cols, o.rows, o.cols);
        }
        return r;
    }

    Tensor mul(f32 s) const {
        Tensor r(rows, cols);
        for (i32 i = 0; i < size(); i++) r.data[i] = data[i] * s;
        return r;
    }

    Tensor matmul(const Tensor& o) const {
        i32 M = rows, N = cols, P = o.cols;
        Tensor r(M, P, 0.0f);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (i32 i = 0; i < M; i++) {
            f32* rp = &r.data[i * P];
            const f32* dp = &data[i * N];
            for (i32 k = 0; k < N; k++) {
                f32 v = dp[k];
                if (v == 0.0f) continue;
                const f32* op = &o.data[k * P];
#if defined(__AVX__) || defined(__AVX2__)
                i32 j = 0;
                // vectorized tail-free dot-product accumulation over P in 8-wide AVX chunks
                i32 jlim = P & ~7;
                for (; j < jlim; j += 8) {
                    __m256 bv = _mm256_loadu_ps(op + j);
                    __m256 cv = _mm256_loadu_ps(rp + j);
                    __m256 pv = _mm256_mul_ps(_mm256_set1_ps(v), bv);
                    _mm256_storeu_ps(rp + j, _mm256_add_ps(cv, pv));
                }
                for (; j < P; j++)
                    rp[j] += v * op[j];
#else
                for (i32 j = 0; j < P; j++)
                    rp[j] += v * op[j];
#endif
            }
        }
        return r;
    }

    Tensor T() const {
        Tensor r(cols, rows);
        for (i32 i = 0; i < rows; i++)
            for (i32 j = 0; j < cols; j++)
                r.at(j, i) = at(i, j);
        return r;
    }

    void print(i32 m = 6) const {
        i32 sr = std::min(rows, m), sc = std::min(cols, m);
        std::cout << std::fixed << std::setprecision(4);
        for (i32 i = 0; i < sr; i++) {
            std::cout << "  [";
            for (i32 j = 0; j < sc; j++)
                std::cout << at(i, j) << (j + 1 < sc ? ", " : "");
            if (cols > m) std::cout << "...";
            std::cout << "]\n";
        }
        if (rows > m) std::cout << "  ... (" << rows << "x" << cols << ")\n";
    }
};
