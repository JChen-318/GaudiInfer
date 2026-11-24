// SPDX-License-Identifier: Apache-2.0
// LUT-free transcendental functions for the Inkling-2 kernels (F32, 64 lanes). The Direct engine never programs the
// TPC lookup-table registers: a kernel using v_sigmoid / v_tanh / v_exp / v_reciprocal there faults the device, so
// every kernel that may run under Direct uses these instead (range reduction + polynomial, Newton reciprocal).
#pragma once

// 1 / x on the scalar unit without CALC_FP_SPECIAL (its recip / div read the special-function tables, which only
// Synapse sets up: under Direct they page-fault). Bit-trick estimate + 3 Newton steps (~1 ulp).
static inline float ink_srecip(float x) {
    float r = as_float(0x7ef311c3 - as_int(x));
    r = r * (2.0f - x * r);
    r = r * (2.0f - x * r);
    r = r * (2.0f - x * r);
    return r;
}

// e^x for x in [-87, 88] (clamped): 2^(x log2 e) = 2^n * p(f), f in [-0.5, 0.5], degree-5 polynomial (~1e-7)
static inline float64 ink_exp(float64 x) {
    const float64 xc = v_f32_min_b(v_f32_max_b(x, -87.0f), 88.0f);
    const float64 t = xc * 1.4426950408889634f;
    const int64 n = v_convert_f32_to_i32_b(t, SW_RHNE);
    const float64 f = t - v_convert_i32_to_f32_b(n, 0);
    float64 p = 0.0013333558f;
    p = v_f32_mac_b(p, f, 0.0096181291f);
    p = v_f32_mac_b(p, f, 0.0555041087f);
    p = v_f32_mac_b(p, f, 0.2402265070f);
    p = v_f32_mac_b(p, f, 0.6931471806f);
    p = v_f32_mac_b(p, f, 1.0f);
    const int64 bits = (n + 127) << 23;
    return p * as_float64(bits);
}

// 1 / x for positive finite x: magic-constant estimate + 3 Newton steps
static inline float64 ink_recip(float64 x) {
    float64 r = as_float64((int64)0x7ef311c3 - as_int64(x));
    r = r * (2.0f - x * r);
    r = r * (2.0f - x * r);
    r = r * (2.0f - x * r);
    return r;
}

// sigmoid(x) = 1 / (1 + e^-x), from e = e^-|x| in (0, 1] (no overflow): x >= 0 -> 1/(1+e), else e/(1+e)
static inline float64 ink_sigmoid(float64 x) {
    const float64 e = ink_exp(-v_f32_abs_b(x));
    const float64 r = ink_recip(1.0f + e);
    return v_f32_sel_less_f32_b(x, 0.0f, e * r, r);
}

// tanh(x) = sign(x) (1 - e) / (1 + e), e = e^-2|x|
static inline float64 ink_tanh(float64 x) {
    const float64 e = ink_exp(-2.0f * v_f32_abs_b(x));
    const float64 t = (1.0f - e) * ink_recip(1.0f + e);
    return v_f32_sel_less_f32_b(x, 0.0f, -t, t);
}

// tanh(x) by the [7/6] Pade approximant on |x| <= 4.97 (clamped; max abs error 9e-5 at the clamp, ~1e-7 below 2) with
// a two-step Newton reciprocal: 16 VPU instructions against ~27 of ink_tanh (exp + reciprocal). For the prefill
// kernels whose results are rounded to BF16 / FP8 anyway (MoE activation, message gates).
static inline float64 ink_tanh_pade(float64 x) {
    const float64 xc = v_f32_min_b(v_f32_max_b(x, -4.97f), 4.97f);
    const float64 x2 = xc * xc;
    float64 n = x2 + 378.0f;
    n = v_f32_mac_b(n, x2, 17325.0f);
    n = v_f32_mac_b(n, x2, 135135.0f);
    float64 d = v_f32_mac_b(x2, 28.0f, 3150.0f);
    d = v_f32_mac_b(d, x2, 62370.0f);
    d = v_f32_mac_b(d, x2, 135135.0f);
    float64 r = as_float64((int64)0x7ef311c3 - as_int64(d));
    r = r * (2.0f - d * r);
    r = r * (2.0f - d * r);
    return v_f32_min_b(v_f32_max_b((n * xc) * r, -1.0f), 1.0f);
}

// e^x - 1 for x in [-87, 88] (clamped). n = round(x log2 e), r = x - n ln 2 in two steps (ln 2 = 0.693359375 -
// 2.1219444e-4: n times the first is exact), e^r - 1 = r (1 + r/2 + ... + r^6/5040) on |r| <= 0.347 (~1e-8), then
// 2^n (e^r - 1) + (2^n - 1): no cancellation near 0 (n = 0 gives the polynomial itself).
static inline float64 ink_expm1(float64 x) {
    const float64 xc = v_f32_min_b(v_f32_max_b(x, -87.0f), 88.0f);
    const int64 n = v_convert_f32_to_i32_b(xc * 1.4426950408889634f, SW_RHNE);
    const float64 nf = v_convert_i32_to_f32_b(n, 0);
    float64 r = v_f32_mac_b(nf, -0.693359375f, xc);
    r = v_f32_mac_b(nf, 2.12194440e-4f, r);
    float64 p = 1.984126984e-4f;
    p = v_f32_mac_b(p, r, 1.388888889e-3f);
    p = v_f32_mac_b(p, r, 8.333333333e-3f);
    p = v_f32_mac_b(p, r, 4.166666667e-2f);
    p = v_f32_mac_b(p, r, 1.666666667e-1f);
    p = v_f32_mac_b(p, r, 0.5f);
    p = v_f32_mac_b(p, r, 1.0f);
    p = p * r;
    const float64 s = as_float64((n + 127) << 23);
    return v_f32_mac_b(s, p, s - 1.0f);
}

// The softcapped SwiGLU of the experts for a gate softcap of 4, from the gate / up pre-activations:
//   act = (4 tanh(g / 4)) sigmoid(g) (cu tanh(u / cu))
// With a = e^(-g/2) and b = e^(-2u/cu): tanh(g / 4) = (1 - a) / (1 + a), sigmoid(g) = 1 / (1 + a^2),
// tanh(u / cu) = (1 - b) / (1 + b), so
//   act = 4 cu (a - 1)(b - 1) / ((1 + a)(1 + a^2)(1 + b))
// from two ink_expm1 and one reciprocal (~1e-7 relative, also for small g and u: a - 1 and b - 1 come without
// cancellation). g is clamped to [-44, 174] and 2u/cu to [-18.2, 18.2] (the product of the denominators stays in
// range; beyond them the factors are at their limits to 1e-8 or the activation is below 1e-17).
static inline float64 ink_swiglu4(float64 g, float64 u, float icu2, float c4cu) {
    const float64 ea = ink_expm1(v_f32_min_b(v_f32_max_b(g * -0.5f, -87.0f), 22.0f));
    const float64 eb = ink_expm1(v_f32_min_b(v_f32_max_b(u * icu2, -18.2f), 18.2f));
    const float64 a = ea + 1.0f;
    const float64 d = ((ea + 2.0f) * v_f32_mac_b(a, a, 1.0f)) * (eb + 2.0f);
    return ((ea * eb) * c4cu) * ink_recip(d);
}
