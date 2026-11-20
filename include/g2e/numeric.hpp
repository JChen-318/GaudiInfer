// Host-side number formats: BF16 and Gaudi2 FP8 (fp8_143 = E4M3 with IEEE-style specials, exponent bias 7 by
// default, so the largest finite value is 240).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace g2e {

inline uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;           // NaN
    u += 0x7fffu + ((u >> 16) & 1u);                                // round to nearest even
    return static_cast<uint16_t>(u >> 16);
}
inline float from_bf16(uint16_t h) {
    uint32_t u = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// E4M3, exponent field 15 reserved (inf/NaN), bias `bias`: finite max = 2^(14-bias) * 1.875.
inline float from_fp8_143(uint8_t v, int bias = 7) {
    const int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    float x;
    if (e == 15) x = m ? NAN : INFINITY;
    else if (e == 0) x = std::ldexp(float(m) / 8.0f, 1 - bias);
    else x = std::ldexp(1.0f + float(m) / 8.0f, e - bias);
    return s ? -x : x;
}
// Round to nearest even, saturating to the largest finite value.
inline uint8_t to_fp8_143(float f, int bias = 7) {
    if (std::isnan(f)) return 0x7f;
    const uint8_t sign = std::signbit(f) ? 0x80 : 0;
    float a = std::fabs(f);
    const float maxv = std::ldexp(1.875f, 14 - bias);
    if (a >= maxv) return sign | 0x77;  // e=14, m=7: the largest finite value
    uint8_t best = 0;
    float best_err = INFINITY;
    for (int code = 0; code < 0x78; ++code) {  // all finite non-negative codes (e < 15)
        const float v = from_fp8_143(uint8_t(code), bias);
        const float err = std::fabs(v - a);
        if (err < best_err || (err == best_err && (code & 1) == 0)) { best_err = err; best = uint8_t(code); }
    }
    return sign | best;
}

}  // namespace g2e
