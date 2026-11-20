#pragma once
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

using f32 = float;
using i32 = int32_t;

// Checks stay active in Release (NDEBUG does not disable them).
static inline i32 checked_size(i32 r, i32 c) {
    if (r < 0 || c < 0) throw std::invalid_argument("Tensor: negative dimension");
    if (r != 0 && c > std::numeric_limits<i32>::max() / r)
        throw std::length_error("Tensor: rows * cols overflows i32");
    return r * c;
}

struct Tensor {
    i32 rows, cols;
    std::vector<f32> data;

    Tensor() : rows(0), cols(0) {}
    Tensor(i32 r, i32 c, f32 v = 0.0f) : rows(r), cols(c), data(checked_size(r, c), v) {}
    Tensor(i32 r, i32 c, const std::vector<f32>& d) : rows(r), cols(c), data(d) {
        if (d.size() != (size_t)checked_size(r, c))
            throw std::invalid_argument("Tensor: data size does not match rows * cols");
    }

    void check_bounds(i32 r, i32 c) const {
        if (r < 0 || r >= rows || c < 0 || c >= cols)
            throw std::out_of_range("Tensor::at: index out of range");
    }
    f32& at(i32 r, i32 c) { check_bounds(r, c); return data[r * cols + c]; }
    f32  at(i32 r, i32 c) const { check_bounds(r, c); return data[r * cols + c]; }
    i32 size() const { return (i32)data.size(); }

    void fill(f32 v) { std::fill(data.begin(), data.end(), v); }
    Tensor clone() const { return Tensor(rows, cols, data); }
    Tensor slice(i32 start, i32 count) const {
        if (start < 0 || count < 0 || start + count > rows)
            throw std::out_of_range("Tensor::slice: range out of bounds");
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
            throw std::invalid_argument("Tensor::emul: unsupported shape combination");
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
            throw std::invalid_argument("Tensor::add: unsupported shape combination");
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
            throw std::invalid_argument("Tensor::sub: unsupported shape combination");
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
        if (o.rows != N) throw std::invalid_argument("Tensor::matmul: incompatible dimensions");
        Tensor r(M, P, 0.0f);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (i32 i = 0; i < M; i++) {
            f32* rp = r.data.data() + (size_t)i * P;
            const f32* dp = data.data() + (size_t)i * N;
            i32 k = 0;
#if defined(__AVX__) || defined(__AVX2__)
            const i32 jlim = P & ~7;
            // 4 independent accumulators: 4x fewer load/store of the output row
            for (; k + 4 <= N; k += 4) {
                const f32* b0 = o.data.data() + (size_t)k * P;
                const f32* b1 = b0 + P;
                const f32* b2 = b1 + P;
                const f32* b3 = b2 + P;
                __m256 a0 = _mm256_set1_ps(dp[k]);
                __m256 a1 = _mm256_set1_ps(dp[k + 1]);
                __m256 a2 = _mm256_set1_ps(dp[k + 2]);
                __m256 a3 = _mm256_set1_ps(dp[k + 3]);
                i32 j = 0;
                for (; j < jlim; j += 8) {
                    __m256 acc = _mm256_mul_ps(a0, _mm256_loadu_ps(b0 + j));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(a1, _mm256_loadu_ps(b1 + j)));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(a2, _mm256_loadu_ps(b2 + j)));
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(a3, _mm256_loadu_ps(b3 + j)));
                    _mm256_storeu_ps(rp + j, _mm256_add_ps(_mm256_loadu_ps(rp + j), acc));
                }
                for (; j < P; j++)
                    rp[j] += dp[k] * b0[j] + dp[k + 1] * b1[j]
                           + dp[k + 2] * b2[j] + dp[k + 3] * b3[j];
            }
#endif
            for (; k < N; k++) {
                f32 v = dp[k];
                if (v == 0.0f) continue;
                const f32* bp = o.data.data() + (size_t)k * P;
                i32 j = 0;
#if defined(__AVX__) || defined(__AVX2__)
                for (; j < jlim; j += 8) {
                    __m256 bv = _mm256_loadu_ps(bp + j);
                    __m256 cv = _mm256_loadu_ps(rp + j);
                    _mm256_storeu_ps(rp + j, _mm256_add_ps(cv, _mm256_mul_ps(_mm256_set1_ps(v), bv)));
                }
#endif
                for (; j < P; j++) rp[j] += v * bp[j];
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
