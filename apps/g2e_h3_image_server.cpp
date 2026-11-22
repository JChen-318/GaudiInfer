// Native Gaudi2 MiniMaxAI/MiniMax-H3-Image OpenAI / vLLM-Omni / ComfyUI HTTP server.
// Endpoints (drop-in compatible with ComfyUI VLLMOmniGenerateImage):
//   GET  /health
//   GET  /v1/models
//   GET  /stats
//   POST /v1/images/generations  (application/json -> {"created":..., "data":[{"b64_json":"..."}]})
//   POST /v1/images/edits        (multipart/form-data, 1..16 images -> {"created":..., "data":[{"b64_json":"..."}]})
#include "g2e/direct_mme.hpp"
#include "g2e/h3_image_pipeline.hpp"

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

using namespace g2e;
using json = nlohmann::json;

namespace {

int64_t unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string b64_encode(const uint8_t* data, size_t len) {
    static constexpr char kTab[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        const uint32_t v = (uint32_t(data[i]) << 16) |
                           ((i + 1 < len) ? (uint32_t(data[i + 1]) << 8) : 0u) |
                           ((i + 2 < len) ? uint32_t(data[i + 2]) : 0u);
        out.push_back(kTab[(v >> 18) & 63]);
        out.push_back(kTab[(v >> 12) & 63]);
        out.push_back((i + 1 < len) ? kTab[(v >> 6) & 63] : '=');
        out.push_back((i + 2 < len) ? kTab[v & 63] : '=');
    }
    return out;
}

void parse_size_str(const std::string& s, uint32_t& w, uint32_t& h) {
    auto pos = s.find('x');
    if (pos == std::string::npos) pos = s.find('X');
    if (pos == std::string::npos) throw std::invalid_argument("size must be WIDTHxHEIGHT");
    w = uint32_t(std::stoul(s.substr(0, pos)));
    h = uint32_t(std::stoul(s.substr(pos + 1)));
    if (w < 64 || w > 6400 || h < 64 || h > 6400) {
        throw std::invalid_argument("dimensions out of supported range [64, 6400]");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string host = "0.0.0.0";
        int port = 8099;
        std::string model_name = "MiniMaxAI/MiniMax-H3-Image";

        H3ImagePipelinePaths paths;
        paths.tokenizer_json = "/mnt/sas/models/minimax-h3/FL2VA/tokenizer/tokenizer.json";
        paths.text_encoder_g2w = "/mnt/sas/g2e/h3_text_encoder_fp8.g2w";
        paths.dit_g2w = "/mnt/sas/g2e/h3_image_dit_fp8.g2w";
        paths.vae_g2w = "/mnt/sas/g2e/h3_image_vae_fp8.g2w";

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--host" && i + 1 < argc) host = argv[++i];
            else if (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
            else if (a == "--served-model-name" && i + 1 < argc) model_name = argv[++i];
            else if (a == "--tokenizer" && i + 1 < argc) paths.tokenizer_json = argv[++i];
            else if (a == "--text-enc" && i + 1 < argc) paths.text_encoder_g2w = argv[++i];
            else if (a == "--dit" && i + 1 < argc) paths.dit_g2w = argv[++i];
            else if (a == "--vae" && i + 1 < argc) paths.vae_g2w = argv[++i];
        }

        DirectRuntime::ensure_scal_config();
        Device dev;
        Stream st(dev);
        H3ImagePipeline pipe(dev, st, paths);

        std::mutex gpu_mu;
        std::atomic<uint64_t> completed{0};
        std::vector<json> recent;

        auto run_and_reply = [&](const H3ImageGenerateRequest& req, httplib::Response& res) {
            H3ImageGenerateResult out;
            {
                std::lock_guard lk(gpu_mu);
                out = pipe.generate(req);
                completed.fetch_add(1);
                recent.push_back({
                    {"ts", unix_now()},
                    {"mode", req.reference_images_bytes.empty() ? "T2I" : "Edit"},
                    {"n_ref", req.reference_images_bytes.size()},
                    {"width", out.width},
                    {"height", out.height},
                    {"canvas_width", out.canvas_width},
                    {"canvas_height", out.canvas_height},
                    {"encode_s", out.encode_s},
                    {"denoise_s", out.denoise_s},
                    {"vae_decode_s", out.vae_decode_s},
                    {"total_s", out.total_s},
                });
                if (recent.size() > 40) recent.erase(recent.begin());
            }
            json body = {
                {"created", unix_now()},
                {"data", {{{"b64_json", b64_encode(out.png_bytes.data(), out.png_bytes.size())}}}},
                {"timing", {
                    {"encode_s", out.encode_s},
                    {"denoise_s", out.denoise_s},
                    {"vae_decode_s", out.vae_decode_s},
                    {"png_encode_s", out.png_encode_s},
                    {"total_s", out.total_s},
                }},
            };
            res.set_content(body.dump(), "application/json");
        };

        httplib::Server svr;
        svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
            json o = {{"status", "healthy"}, {"engine", "gaudi2-engine-h3-image"}, {"model", model_name}};
            res.set_content(o.dump(), "application/json");
        });
        svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
            json o = {{"object", "list"}, {"data", {{{"id", model_name}, {"object", "model"}, {"owned_by", "local"}}}}};
            res.set_content(o.dump(), "application/json");
        });
        svr.Get("/stats", [&](const httplib::Request&, httplib::Response& res) {
            std::lock_guard lk(gpu_mu);
            json o = {{"model", model_name}, {"engine", "gaudi2-engine-h3-image"},
                      {"completed", completed.load()}, {"recent_jobs", recent}};
            res.set_content(o.dump(), "application/json");
        });

        svr.Post("/v1/images/generations", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                auto j = json::parse(req.body);
                H3ImageGenerateRequest r;
                r.prompt = j.at("prompt").get<std::string>();
                if (j.contains("size") && j["size"].is_string())
                    parse_size_str(j["size"].get<std::string>(), r.width, r.height);
                if (j.contains("width")) r.width = j["width"].get<uint32_t>();
                if (j.contains("height")) r.height = j["height"].get<uint32_t>();
                if (j.contains("seed")) r.seed = j["seed"].get<uint64_t>();
                if (j.contains("min_generation_area")) r.min_generation_area = j["min_generation_area"].get<uint64_t>();
                run_and_reply(r, res);
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json({{"error", {{"message", e.what()}}}}).dump(), "application/json");
            }
        });

        svr.Post("/v1/images/edits", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                H3ImageGenerateRequest r;
                if (!req.is_multipart_form_data()) {
                    throw std::invalid_argument("/v1/images/edits expects multipart/form-data");
                }
                for (const auto& [name, field] : req.form.fields) {
                    const auto& v = field.content;
                    if (name == "prompt") r.prompt = v;
                    else if (name == "size" && !v.empty()) parse_size_str(v, r.width, r.height);
                    else if (name == "width" && !v.empty()) r.width = uint32_t(std::stoul(v));
                    else if (name == "height" && !v.empty()) r.height = uint32_t(std::stoul(v));
                    else if (name == "seed" && !v.empty()) r.seed = std::stoull(v);
                    else if (name == "min_generation_area" && !v.empty()) r.min_generation_area = std::stoull(v);
                }
                auto add_files = [&](const std::string& key) {
                    for (const auto& f : req.form.get_files(key)) {
                        if (!f.content.empty()) {
                            r.reference_images_bytes.emplace_back(
                                reinterpret_cast<const uint8_t*>(f.content.data()),
                                reinterpret_cast<const uint8_t*>(f.content.data()) + f.content.size());
                        }
                    }
                };
                add_files("image");
                add_files("image[]");
                if (r.reference_images_bytes.empty() || r.reference_images_bytes.size() > 16) {
                    throw std::invalid_argument("Provide 1 to 16 reference images");
                }
                run_and_reply(r, res);
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json({{"error", {{"message", e.what()}}}}).dump(), "application/json");
            }
        });

        std::printf("[g2e_h3_image_server] listening on http://%s:%d (%s)\n", host.c_str(), port, model_name.c_str());
        svr.listen(host, port);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
