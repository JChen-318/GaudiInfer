// Inkling-2 vision tower vs the Hugging Face golden (ref/dump_vision_golden.py): preprocessing and features.
//   inkling_vision_test --pack V.g2w --golden DIR [names...]   (default names: small mid)
#include "g2e/inkling_vision.hpp"
#include "g2e/numeric.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace g2e;

template <typename T>
static std::vector<T> read_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<T> v(b.size() / sizeof(T));
    std::memcpy(v.data(), b.data(), v.size() * sizeof(T));
    return v;
}

int main(int argc, char** argv) {
    std::string pack_path, golden;
    std::vector<std::string> names;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pack") pack_path = argv[++i];
        else if (a == "--golden") golden = argv[++i];
        else names.push_back(a);
    }
    if (names.empty()) names = {"small", "mid"};
    try {
        Device dev;
        Stream stream(dev);
        WeightPack pack(pack_path);
        pack.upload(dev, stream);
        InklingVision vis(dev, stream, pack);
        for (const auto& nm : names) {
            const auto file = read_raw<uint8_t>(golden + "/" + nm + ".png");
            int w = 0, h = 0;
            const auto rgb = decode_image(file.data(), file.size(), w, h);
            const auto img = preprocess_inkling_image(rgb.data(), w, h, 0, vis.patch(), vis.merge());
            // preprocessing vs the reference pixel_values [tokens, 40, 40, 3]
            const auto pix = read_raw<float>(golden + "/" + nm + ".pix.f32");
            const uint32_t P = vis.patch(), Mg = vis.merge(), C = P * Mg;
            double pmax = 0;
            for (uint64_t p = 0; p < img.patches.size() / (P * P * 3); ++p) {
                const uint64_t tok = p / (Mg * Mg), in = p % (Mg * Mg), mr = in / Mg, mc = in % Mg;
                for (uint32_t y = 0; y < P; ++y)
                    for (uint32_t x = 0; x < P; ++x)
                        for (uint32_t c = 0; c < 3; ++c) {
                            const float ref = pix[((tok * C + mr * P + y) * C + mc * P + x) * 3 + c];
                            pmax = std::max(pmax, double(std::fabs(ref - img.patches[(p * P * P + y * P + x) * 3 + c])));
                        }
            }
            const auto feat = read_raw<float>(golden + "/" + nm + ".feat.f32");
            const uint64_t T = img.tokens(), O = vis.out_hidden();
            if (feat.size() != T * O) throw std::runtime_error(nm + ": token count differs from the golden");
            DeviceBuffer out(dev, T * O * 2);
            vis.encode(img, out, 0);  // compile
            vis.encode(img, out, 0);
            HostBuffer hb(dev, T * O * 2);
            stream.download(out, 0, hb, 0, T * O * 2);
            stream.synchronize();
            const auto* o = hb.as<uint16_t>();
            double num = 0, den = 0, cmin = 1;
            for (uint64_t t = 0; t < T; ++t) {
                double d = 0, na = 0, nb = 0;
                for (uint64_t i = 0; i < O; ++i) {
                    const double a = from_bf16(o[t * O + i]), b = feat[t * O + i];
                    num += (a - b) * (a - b);
                    den += b * b;
                    d += a * b;
                    na += a * a;
                    nb += b * b;
                }
                cmin = std::min(cmin, d / std::sqrt(na * nb + 1e-30));
            }
            std::printf("%s: %ux%u -> %llu tokens; pixels maxdiff %.2e; features rel err %.4f, min row cosine %.5f; %.2f ms\n",
                        nm.c_str(), w, h, (unsigned long long)T, pmax, std::sqrt(num / den), cmin, vis.last_ms());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
