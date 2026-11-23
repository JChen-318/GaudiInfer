// Image decoding (stb_image) and the Qwen2-VL image processor (smart resize, bicubic, normalise, patchify).
#include "g2e/vision.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <png.h>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_FAILURE_USERMSG
#include "stb/stb_image.h"

namespace g2e {
namespace {

// Python's round(): to nearest, ties to even
double py_round(double x) { return std::nearbyint(x); }

// transformers smart_resize: multiples of `factor`, pixel count within [min_pixels, max_pixels], aspect kept
void smart_resize(int height, int width, int factor, uint64_t min_pixels, uint64_t max_pixels, int& out_h, int& out_w) {
    if (double(std::max(height, width)) / double(std::min(height, width)) > 200)
        throw std::invalid_argument("image aspect ratio must be smaller than 200");
    double h_bar = py_round(double(height) / factor) * factor, w_bar = py_round(double(width) / factor) * factor;
    if (h_bar * w_bar > double(max_pixels)) {
        const double beta = std::sqrt(double(height) * width / double(max_pixels));
        h_bar = std::max<double>(factor, std::floor(height / beta / factor) * factor);
        w_bar = std::max<double>(factor, std::floor(width / beta / factor) * factor);
    } else if (h_bar * w_bar < double(min_pixels)) {
        const double beta = std::sqrt(double(min_pixels) / (double(height) * width));
        h_bar = std::ceil(height * beta / factor) * factor;
        w_bar = std::ceil(width * beta / factor) * factor;
    }
    out_h = int(h_bar);
    out_w = int(w_bar);
}

double bicubic(double x) {  // a = -0.5 (PIL)
    constexpr double a = -0.5;
    x = std::fabs(x);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
}

// PIL ImagingResample coefficients for one axis (antialiased: the filter widens by the scale when shrinking),
// as 22-bit fixed point for 8-bit images
struct Coeffs {
    int ksize = 0;
    std::vector<int> start, size;
    std::vector<int32_t> k;  // [out, ksize]
};
Coeffs coefficients(int in, int out) {
    constexpr int kBits = 32 - 8 - 2;
    const double scale = double(in) / out, filterscale = std::max(scale, 1.0), support = 2.0 * filterscale;
    Coeffs c;
    c.ksize = int(std::ceil(support)) * 2 + 1;
    c.start.resize(out);
    c.size.resize(out);
    c.k.assign(size_t(out) * c.ksize, 0);
    std::vector<double> w(c.ksize);
    for (int i = 0; i < out; ++i) {
        const double center = (i + 0.5) * scale;
        int xmin = int(center - support + 0.5), xmax = int(center + support + 0.5);
        xmin = std::max(xmin, 0);
        xmax = std::min(xmax, in) - xmin;
        double total = 0;
        for (int x = 0; x < xmax; ++x) total += w[x] = bicubic((x + xmin - center + 0.5) / filterscale);
        for (int x = 0; x < xmax; ++x) {
            const double v = total != 0 ? w[x] / total : 0;
            c.k[size_t(i) * c.ksize + x] = int32_t(v < 0 ? std::trunc(v * (1 << kBits) - 0.5) : std::trunc(v * (1 << kBits) + 0.5));
        }
        c.start[i] = xmin;
        c.size[i] = xmax;
    }
    return c;
}

uint8_t clip8(int64_t v) {
    constexpr int kBits = 32 - 8 - 2;
    v = (v + (int64_t(1) << (kBits - 1))) >> kBits;
    return uint8_t(std::clamp<int64_t>(v, 0, 255));
}

// 8-bit [h][w][C] -> [oh][ow][C]: horizontal pass, then vertical (PIL's order), 8-bit intermediate
std::vector<uint8_t> resize_channels(const uint8_t* src, int w, int h, int ow, int oh, int C) {
    std::vector<uint8_t> mid(size_t(h) * ow * C), dst(size_t(oh) * ow * C);
    const auto ch = coefficients(w, ow), cv = coefficients(h, oh);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < ow; ++x)
            for (int c = 0; c < C; ++c) {
                int64_t acc = 0;
                for (int k = 0; k < ch.size[x]; ++k)
                    acc += int64_t(src[(size_t(y) * w + ch.start[x] + k) * C + c]) * ch.k[size_t(x) * ch.ksize + k];
                mid[(size_t(y) * ow + x) * C + c] = clip8(acc);
            }
    for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x)
            for (int c = 0; c < C; ++c) {
                int64_t acc = 0;
                for (int k = 0; k < cv.size[y]; ++k)
                    acc += int64_t(mid[(size_t(cv.start[y] + k) * ow + x) * C + c]) * cv.k[size_t(y) * cv.ksize + k];
                dst[(size_t(y) * ow + x) * C + c] = clip8(acc);
            }
    return dst;
}

std::vector<uint8_t> encode_png_impl(const uint8_t* px, int width, int height, int color_type, int channels) {
    std::vector<uint8_t> out;
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) throw std::runtime_error("png_create_write_struct failed");
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        throw std::runtime_error("png_create_info_struct failed");
    }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        throw std::runtime_error("libpng write error");
    }
    png_set_write_fn(
        png, &out,
        [](png_structp p, png_bytep data, png_size_t len) {
            auto* vec = static_cast<std::vector<uint8_t>*>(png_get_io_ptr(p));
            vec->insert(vec->end(), data, data + len);
        },
        nullptr);
    png_set_compression_level(png, 1);  // fast deflate (~3x faster than default 6, lossless)
    png_set_IHDR(png, info, uint32_t(width), uint32_t(height), 8, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    std::vector<png_bytep> rows(static_cast<size_t>(height));
    for (int y = 0; y < height; ++y) rows[size_t(y)] = const_cast<png_bytep>(px + size_t(y) * width * channels);
    png_write_image(png, rows.data());
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    return out;
}

bool is_webp(const uint8_t* data, size_t size) {
    return data && size >= 12 && std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "WEBP", 4) == 0;
}

std::vector<uint8_t> decode_webp(const uint8_t* data, size_t size, int& width, int& height, int channels) {
    using DecodeFn = uint8_t* (*)(const uint8_t*, size_t, int*, int*);
    using FreeFn = void (*)(void*);
    struct WebpApi {
        DecodeFn decode_rgb = nullptr;
        DecodeFn decode_rgba = nullptr;
        FreeFn free_fn = nullptr;
    };
    static WebpApi api;
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        void* handle = dlopen("libwebp.so.7", RTLD_LAZY);
        if (!handle) handle = dlopen("libwebp.so.6", RTLD_LAZY);
        if (!handle) handle = dlopen("libwebp.so", RTLD_LAZY);
        if (handle) {
            api.decode_rgb = reinterpret_cast<DecodeFn>(dlsym(handle, "WebPDecodeRGB"));
            api.decode_rgba = reinterpret_cast<DecodeFn>(dlsym(handle, "WebPDecodeRGBA"));
            api.free_fn = reinterpret_cast<FreeFn>(dlsym(handle, "WebPFree"));
        }
    });
    const DecodeFn fn = (channels == 4) ? api.decode_rgba : api.decode_rgb;
    if (!fn || !api.free_fn) {
        throw std::invalid_argument("cannot decode the image: WebP image requires libwebp (libwebp.so.7)");
    }
    uint8_t* px = fn(data, size, &width, &height);
    std::unique_ptr<uint8_t, FreeFn> guard(px, api.free_fn);
    if (!px || width <= 0 || height <= 0) {
        throw std::invalid_argument("cannot decode the image: invalid or corrupt WebP");
    }
    return std::vector<uint8_t>(px, px + size_t(width) * size_t(height) * size_t(channels));
}

}  // namespace

std::vector<uint8_t> decode_image(const uint8_t* data, size_t size, int& width, int& height) {
    // Decode as RGBA and composite over solid white so transparent PNG/WebP images (e.g. dark UI text or
    // diagrams with alpha=0) do not turn black-on-black; opaque images (alpha=255) are bit-exact.
    const auto rgba = decode_image_rgba(data, size, width, height);
    return rgba_over_white_to_rgb(rgba.data(), width, height);
}

std::vector<uint8_t> decode_image_rgba(const uint8_t* data, size_t size, int& width, int& height) {
    if (is_webp(data, size)) return decode_webp(data, size, width, height, 4);
    int channels = 0;
    stbi_uc* px = stbi_load_from_memory(data, int(size), &width, &height, &channels, 4);
    if (!px) throw std::invalid_argument(std::string("cannot decode the image: ") + stbi_failure_reason());
    std::vector<uint8_t> rgba(px, px + size_t(width) * height * 4);
    stbi_image_free(px);
    return rgba;
}

std::vector<uint8_t> resize_rgb_bicubic(const uint8_t* src, int w, int h, int ow, int oh) {
    if (w == ow && h == oh) return std::vector<uint8_t>(src, src + size_t(w) * h * 3);
    return resize_channels(src, w, h, ow, oh, 3);
}

std::vector<uint8_t> resize_rgba_bicubic(const uint8_t* src, int w, int h, int ow, int oh) {
    if (w == ow && h == oh) return std::vector<uint8_t>(src, src + size_t(w) * h * 4);
    return resize_channels(src, w, h, ow, oh, 4);
}

std::vector<uint8_t> rgba_over_white_to_rgb(const uint8_t* rgba, int width, int height) {
    const size_t n = size_t(width) * height;
    std::vector<uint8_t> rgb(n * 3);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t a = rgba[i * 4 + 3];
        const uint32_t inv = 255u - a;
        for (int c = 0; c < 3; ++c) {
            rgb[i * 3 + c] = uint8_t((uint32_t(rgba[i * 4 + c]) * a + 255u * inv + 127u) / 255u);
        }
    }
    return rgb;
}

std::vector<uint8_t> encode_png_rgba(const uint8_t* rgba, int width, int height) {
    return encode_png_impl(rgba, width, height, PNG_COLOR_TYPE_RGBA, 4);
}

std::vector<uint8_t> encode_png_rgb(const uint8_t* rgb, int width, int height) {
    return encode_png_impl(rgb, width, height, PNG_COLOR_TYPE_RGB, 3);
}

VisionImage preprocess_image_exact(const uint8_t* rgb, int width, int height, uint32_t patch, uint32_t merge) {
    const uint32_t factor = patch * merge;
    if (width <= 0 || height <= 0 || width % int(factor) != 0 || height % int(factor) != 0) {
        throw std::invalid_argument("preprocess_image_exact: dimensions must be multiples of patch*merge");
    }
    VisionImage out;
    out.grid_h = uint32_t(height) / patch;
    out.grid_w = uint32_t(width) / patch;
    const uint64_t P = patch, K = 3 * 2 * P * P, N = uint64_t(out.grid_h) * out.grid_w;
    out.patches.resize(N * K);
    const uint32_t bw = out.grid_w / merge;
    for (uint64_t p = 0; p < N; ++p) {
        const uint64_t blk = p / (merge * merge), in = p % (merge * merge);
        const uint64_t row = (blk / bw) * merge + in / merge, col = (blk % bw) * merge + in % merge;
        float* dst = &out.patches[p * K];
        for (uint64_t c = 0; c < 3; ++c)
            for (uint64_t t = 0; t < 2; ++t)
                for (uint64_t y = 0; y < P; ++y)
                    for (uint64_t x = 0; x < P; ++x) {
                        const uint8_t v = rgb[((row * P + y) * uint64_t(width) + col * P + x) * 3 + c];
                        dst[((c * 2 + t) * P + y) * P + x] = (float(v) * float(1.0 / 255.0) - 0.5f) / 0.5f;
                    }
    }
    return out;
}

VisionImage preprocess_image(const uint8_t* rgb, int width, int height, uint64_t min_pixels, uint64_t max_pixels,
                             uint32_t patch, uint32_t merge) {
    int rh = 0, rw = 0;
    smart_resize(height, width, int(patch * merge), min_pixels, max_pixels, rh, rw);
    const auto img = (rh == height && rw == width) ? std::vector<uint8_t>(rgb, rgb + size_t(width) * height * 3)
                                                   : resize_channels(rgb, width, height, rw, rh, 3);
    return preprocess_image_exact(img.data(), rw, rh, patch, merge);
}

}  // namespace g2e
