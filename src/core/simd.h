#pragma once
#include "tensor.h"

// AVX intrinsics compile to the same VEX instructions as hand-written asm.
// Every entry point has a scalar fallback, so builds without -mavx keep working.

namespace simd {

#if defined(__AVX__)

inline __m256 load8(const f32* p) { return _mm256_loadu_ps(p); }
inline void store8(f32* p, __m256 v) { _mm256_storeu_ps(p, v); }

inline __m256 set8(f32 v) { return _mm256_set1_ps(v); }

inline f32 hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

inline f32 hmax(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0x4E));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0xB1));
    return _mm_cvtss_f32(lo);
}

// exp2(r) Taylor on [-0.5, 0.5]; relative error < 2e-8.
inline __m256 exp8(__m256 x) {
    x = _mm256_max_ps(x, set8(-80.0f)); // softmax tails: exp(-80) is already 0 in f32 terms
    __m256 y = _mm256_mul_ps(x, set8(1.4426950408889634f));
    __m256 k = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_sub_ps(y, k);
    __m256 p = set8(0.00015403530393381676f);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), set8(0.0013333558146428443f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), set8(0.009618129107628477f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), set8(0.05550410866482158f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), set8(0.2402265069591007f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), set8(0.6931471805599453f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), set8(1.0f));
    // AVX1 has no 256-bit integer ops: bias+convert in 128-bit halves.
    const __m128 bias = _mm_set1_ps(127.0f);
    __m128i elo = _mm_slli_epi32(_mm_cvtps_epi32(_mm_add_ps(_mm256_castps256_ps128(k), bias)), 23);
    __m128i ehi = _mm_slli_epi32(_mm_cvtps_epi32(_mm_add_ps(_mm256_extractf128_ps(k, 1), bias)), 23);
    __m256 scale = _mm256_castsi256_ps(_mm256_set_m128i(ehi, elo));
    return _mm256_mul_ps(p, scale);
}

inline f32 dot(const f32* a, const f32* b, i32 n) {
    __m256 s0 = set8(0), s1 = set8(0), s2 = set8(0), s3 = set8(0);
    i32 i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(load8(a + i),      load8(b + i)));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(load8(a + i + 8),  load8(b + i + 8)));
        s2 = _mm256_add_ps(s2, _mm256_mul_ps(load8(a + i + 16), load8(b + i + 16)));
        s3 = _mm256_add_ps(s3, _mm256_mul_ps(load8(a + i + 24), load8(b + i + 24)));
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    f32 r = hsum(s);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
}

// Stable softmax over one row; row is overwritten in place.
inline void softmax_row(f32* row, i32 n) {
    __m256 mx = set8(-1e30f);
    i32 j = 0;
    for (; j + 8 <= n; j += 8) mx = _mm256_max_ps(mx, load8(row + j));
    f32 maxv = hmax(mx);
    for (; j < n; j++) if (row[j] > maxv) maxv = row[j];

    __m256 sumv = set8(0);
    j = 0;
    for (; j + 8 <= n; j += 8) {
        __m256 e = exp8(_mm256_sub_ps(load8(row + j), set8(maxv)));
        store8(row + j, e);
        sumv = _mm256_add_ps(sumv, e);
    }
    f32 sum = hsum(sumv);
    for (; j < n; j++) {
        f32 e = std::exp(row[j] - maxv);
        row[j] = e;
        sum += e;
    }
    __m256 inv = set8(1.0f / sum);
    j = 0;
    for (; j + 8 <= n; j += 8) store8(row + j, _mm256_mul_ps(load8(row + j), inv));
    for (; j < n; j++) row[j] *= 1.0f / sum;
}

#else

inline f32 dot(const f32* a, const f32* b, i32 n) {
    f32 r = 0;
    for (i32 i = 0; i < n; i++) r += a[i] * b[i];
    return r;
}

inline void softmax_row(f32* row, i32 n) {
    f32 maxv = row[0];
    for (i32 j = 1; j < n; j++) if (row[j] > maxv) maxv = row[j];
    f32 sum = 0;
    for (i32 j = 0; j < n; j++) { row[j] = std::exp(row[j] - maxv); sum += row[j]; }
    for (i32 j = 0; j < n; j++) row[j] /= sum;
}

#endif

} // namespace simd
