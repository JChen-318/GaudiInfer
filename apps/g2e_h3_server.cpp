// Native Gaudi2 MiniMax-H3 OpenAI / vLLM-Omni / ComfyUI-compatible HTTP server.
// Endpoints:
//   GET  /health
//   GET  /stats
//   GET  /v1/models
//   POST /v1/videos            (multipart/form-data or application/json -> async job)
//   POST /v1/videos/sync       (multipart/form-data or application/json -> blocking MP4)
//   GET  /v1/videos/{id}       (job status + timing breakdown)
//   GET  /v1/videos/{id}/content (video/mp4 bytes)
//   DELETE /v1/videos/{id}
#include "g2e/direct_mme.hpp"
#include "g2e/h3_pipeline.hpp"

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>

using namespace g2e;
using json = nlohmann::json;

namespace {

struct VideoJob {
    std::string id;
    std::string status = "queued";  // queued, in_progress, completed, failed
    int progress = 0;
    std::string error;
    H3GenerateRequest req;
    H3GenerateResult res;
    int64_t created_at = 0;
};

int64_t unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void parse_size(const std::string& s, uint32_t& w, uint32_t& h) {
    auto pos = s.find('x');
    if (pos == std::string::npos) pos = s.find('X');
    if (pos != std::string::npos) {
        w = uint32_t(std::stoul(s.substr(0, pos)));
        h = uint32_t(std::stoul(s.substr(pos + 1)));
    }
}

H3GenerateRequest parse_video_request(const httplib::Request& http_req, const std::string& out_mp4) {
    H3GenerateRequest r;
    r.output_mp4 = out_mp4;

    uint32_t req_num_frames = 0;
    bool got_size = false;
    auto parse_frame_indices = [&](const json& arr) {
        if (!arr.is_array()) return;
        r.keyframe_frame_indices.clear();
        for (const auto& el : arr) {
            if (el.is_number_integer()) r.keyframe_frame_indices.push_back(el.get<int>());
        }
    };
    auto apply_kv = [&](const std::string& k, const std::string& v) {
        if (v.empty()) return;
        if (k == "prompt") r.prompt = v;
        else if (k == "size") { parse_size(v, r.width, r.height); got_size = true; }
        else if (k == "width") { r.width = uint32_t(std::stoul(v)); got_size = true; }
        else if (k == "height") { r.height = uint32_t(std::stoul(v)); got_size = true; }
        else if (k == "seconds" || k == "duration" || k == "duration_seconds") r.duration_seconds = std::stof(v);
        else if (k == "num_frames") req_num_frames = uint32_t(std::stoul(v));
        else if (k == "fps") r.fps = uint32_t(std::stoul(v));
        else if (k == "num_inference_steps") {
            // MiniMax-H3 Turbo API passes sigma_points (= denoise_steps + 1), e.g. 5 for 4-step Turbo
            uint32_t n = uint32_t(std::stoul(v));
            r.num_steps = (n > 1) ? (n - 1) : n;
        } else if (k == "steps") r.num_steps = uint32_t(std::stoul(v));
        else if (k == "seed") r.seed = std::stoull(v);
        else if (k == "flow_shift") r.flow_shift = std::stof(v);
        else if (k == "audio_flow_shift") r.audio_flow_shift = std::stof(v);
        else if (k == "frame_indices") {
            try { parse_frame_indices(json::parse(v)); } catch (...) {}
        } else if (k == "lora_path") r.lora_path = v;
        else if (k == "lora") {
            if (!v.empty() && v[0] == '{') {
                try {
                    auto j = json::parse(v);
                    if (j.contains("local_path") && j["local_path"].is_string()) r.lora_path = j["local_path"].get<std::string>();
                    else if (j.contains("path") && j["path"].is_string()) r.lora_path = j["path"].get<std::string>();
                    if (j.contains("scale") && j["scale"].is_number()) r.lora_scale = j["scale"].get<float>();
                } catch (...) {}
            } else {
                r.lora_path = v;
            }
        } else if (k == "lora_scale") r.lora_scale = std::stof(v);
        else if (k == "extra_params" || k == "extra_args" || k == "lora_request") {
            try {
                auto j = json::parse(v);
                if (j.contains("flow_shift")) r.flow_shift = j["flow_shift"].get<float>();
                if (j.contains("audio_flow_shift")) r.audio_flow_shift = j["audio_flow_shift"].get<float>();
                if (j.contains("duration")) r.duration_seconds = j["duration"].get<float>();
                if (j.contains("duration_seconds")) r.duration_seconds = j["duration_seconds"].get<float>();
                if (j.contains("num_frames")) req_num_frames = j["num_frames"].get<uint32_t>();
                if (j.contains("frame_indices")) parse_frame_indices(j["frame_indices"]);
                if (j.contains("local_path")) r.lora_path = j["local_path"].get<std::string>();
                if (j.contains("lora_path")) r.lora_path = j["lora_path"].get<std::string>();
                if (j.contains("scale")) r.lora_scale = j["scale"].get<float>();
                if (j.contains("lora") && j["lora"].is_object()) {
                    const auto& l = j["lora"];
                    if (l.contains("local_path")) r.lora_path = l["local_path"].get<std::string>();
                    if (l.contains("scale")) r.lora_scale = l["scale"].get<float>();
                }
            } catch (...) {}
        }
    };

    const std::string ct = http_req.get_header_value("Content-Type");
    if (http_req.is_multipart_form_data()) {
        for (const auto& [name, field] : http_req.form.fields) {
            apply_kv(name, field.content);
        }
        // ComfyUI-vLLM-Omni & OpenAI FL2VA multipart file uploads:
        // - 2 keyframes (first + last): repeated "input_references" parts
        // - 1 keyframe (first or last): single "input_reference" part
        auto push_files = [&](const std::string& key) {
            for (const auto& f : http_req.form.get_files(key)) {
                if (!f.content.empty()) {
                    r.keyframe_images_bytes.emplace_back(
                        reinterpret_cast<const uint8_t*>(f.content.data()),
                        reinterpret_cast<const uint8_t*>(f.content.data()) + f.content.size());
                }
            }
        };
        if (http_req.form.has_file("input_references")) {
            push_files("input_references");
        } else if (http_req.form.has_file("input_reference")) {
            push_files("input_reference");
        } else if (http_req.form.has_file("image")) {
            push_files("image");
        }
    } else if (ct.find("application/x-www-form-urlencoded") != std::string::npos ||
               (!http_req.params.empty() && (http_req.body.empty() || http_req.body[0] != '{'))) {
        // aiohttp.FormData without file attachments (ComfyUI T2VA) sends x-www-form-urlencoded!
        for (const auto& [name, val] : http_req.params) {
            apply_kv(name, val);
        }
    } else if (!http_req.body.empty()) {
        auto j = json::parse(http_req.body);
        if (j.contains("prompt")) r.prompt = j["prompt"].get<std::string>();
        if (j.contains("size")) { parse_size(j["size"].get<std::string>(), r.width, r.height); got_size = true; }
        if (j.contains("width")) { r.width = j["width"].get<uint32_t>(); got_size = true; }
        if (j.contains("height")) { r.height = j["height"].get<uint32_t>(); got_size = true; }
        if (j.contains("seconds")) r.duration_seconds = j["seconds"].get<float>();
        if (j.contains("duration")) r.duration_seconds = j["duration"].get<float>();
        if (j.contains("num_frames")) req_num_frames = j["num_frames"].get<uint32_t>();
        if (j.contains("fps")) r.fps = j["fps"].get<uint32_t>();
        if (j.contains("num_inference_steps")) {
            uint32_t n = j["num_inference_steps"].get<uint32_t>();
            r.num_steps = (n > 1) ? (n - 1) : n;
        }
        if (j.contains("steps")) r.num_steps = j["steps"].get<uint32_t>();
        if (j.contains("seed")) r.seed = j["seed"].get<uint64_t>();
        if (j.contains("flow_shift")) r.flow_shift = j["flow_shift"].get<float>();
        if (j.contains("audio_flow_shift")) r.audio_flow_shift = j["audio_flow_shift"].get<float>();
        if (j.contains("frame_indices")) parse_frame_indices(j["frame_indices"]);
        if (j.contains("extra_params") && j["extra_params"].is_object()) {
            const auto& ep = j["extra_params"];
            if (ep.contains("frame_indices")) parse_frame_indices(ep["frame_indices"]);
            if (ep.contains("flow_shift")) r.flow_shift = ep["flow_shift"].get<float>();
            if (ep.contains("audio_flow_shift")) r.audio_flow_shift = ep["audio_flow_shift"].get<float>();
        }
        if (j.contains("lora_path")) r.lora_path = j["lora_path"].get<std::string>();
        if (j.contains("lora") && j["lora"].is_object()) {
            const auto& l = j["lora"];
            if (l.contains("local_path")) r.lora_path = l["local_path"].get<std::string>();
            if (l.contains("scale")) r.lora_scale = l["scale"].get<float>();
        }
        if (j.contains("lora_scale")) r.lora_scale = j["lora_scale"].get<float>();
    }
    if (!r.keyframe_images_bytes.empty() && !got_size) {
        r.width = 0;
        r.height = 0;
    }
    if (req_num_frames > 0 && r.fps > 0) {
        r.duration_seconds = float(req_num_frames) / float(r.fps);
    }
    return r;
}

json job_to_json(const VideoJob& j, const std::string& model_name) {
    json o = {
        {"id", j.id},
        {"object", "video"},
        {"model", model_name},
        {"status", j.status},
        {"progress", j.progress},
        {"created_at", j.created_at},
    };
    if (!j.error.empty()) o["error"] = {{"message", j.error}};
    if (j.status == "completed") {
        o["width"] = j.res.width;
        o["height"] = j.res.height;
        o["num_frames"] = j.res.num_frames;
        o["timing"] = {
            {"encode_s", j.res.encode_s},
            {"denoise_s", j.res.denoise_s},
            {"vvae_s", j.res.vvae_s},
            {"avae_s", j.res.avae_s},
            {"mux_s", j.res.mux_s},
            {"total_s", j.res.total_s},
        };
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string host = "0.0.0.0";
        int port = 8097;
        std::string model_name = "MiniMaxAI/MiniMax-H3";
        std::string temp_dir = "/tmp/g2e_h3_jobs";

        H3PipelinePaths paths;
        paths.tokenizer_json = "/mnt/sas/models/minimax-h3/FL2VA/tokenizer/tokenizer.json";
        paths.text_encoder_g2w = "/mnt/sas/g2e/h3_text_encoder_fp8.g2w";
        paths.dit_g2w = "/mnt/sas/g2e/h3_dit_base_fp8.g2w";
        paths.video_vae_g2w = std::filesystem::exists("/mnt/sas/g2e/h3_video_vae_dec_fp8.g2w")
            ? "/mnt/sas/g2e/h3_video_vae_dec_fp8.g2w"
            : "/mnt/sas/g2e/h3_video_vae_dec.g2w";
        paths.audio_vae_g2w = "/mnt/sas/g2e/h3_audio_vae_dec.g2w";
        paths.default_lora = "/mnt/sas/models/minimax-h3/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors";
        paths.ffmpeg_bin = "/home/a/h3-deploy/media-bin/ffmpeg";

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--host" && i + 1 < argc) host = argv[++i];
            else if (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
            else if (a == "--served-model-name" && i + 1 < argc) model_name = argv[++i];
            else if (a == "--temp-dir" && i + 1 < argc) temp_dir = argv[++i];
            else if (a == "--lora" && i + 1 < argc) paths.default_lora = argv[++i];
            else if (a == "--dit" && i + 1 < argc) paths.dit_g2w = argv[++i];
            else if ((a == "--text-enc" || a == "--text") && i + 1 < argc) paths.text_encoder_g2w = argv[++i];
            else if (a == "--tokenizer" && i + 1 < argc) paths.tokenizer_json = argv[++i];
            else if (a == "--vvae" && i + 1 < argc) paths.video_vae_g2w = argv[++i];
            else if (a == "--avae" && i + 1 < argc) paths.audio_vae_g2w = argv[++i];
            else if (a == "--ffmpeg" && i + 1 < argc) paths.ffmpeg_bin = argv[++i];
        }
        std::filesystem::create_directories(temp_dir);

        DirectRuntime::ensure_scal_config();
        Device dev;
        Stream st(dev);
        H3Pipeline pipe(dev, st, paths);

        std::mutex mu;
        std::condition_variable cv;
        std::map<std::string, std::shared_ptr<VideoJob>> jobs;
        std::queue<std::string> pending;
        std::atomic<uint64_t> seq{0};
        std::atomic<bool> stop{false};
        std::atomic<uint64_t> completed_count{0};
        double last_total_s = 0.0;
        std::vector<json> recent_jobs;
        std::string active_job_id;
        int64_t active_started_at = 0;
        H3GenerateRequest active_req;

        // Single-card serial worker thread
        std::thread worker([&]() {
            while (true) {
                std::shared_ptr<VideoJob> job;
                {
                    std::unique_lock lk(mu);
                    cv.wait(lk, [&] { return stop.load() || !pending.empty(); });
                    if (stop.load() && pending.empty()) break;
                    auto id = pending.front();
                    pending.pop();
                    auto it = jobs.find(id);
                    if (it == jobs.end()) continue;
                    job = it->second;
                    job->status = "in_progress";
                    job->progress = 10;
                    active_job_id = job->id;
                    active_started_at = unix_now();
                    active_req = job->req;
                }
                try {
                    auto res = pipe.generate(job->req);
                    std::lock_guard lk(mu);
                    job->res = std::move(res);
                    job->status = "completed";
                    job->progress = 100;
                    last_total_s = job->res.total_s;
                    completed_count.fetch_add(1);
                    recent_jobs.push_back({
                        {"id", job->id},
                        {"ts", unix_now()},
                        {"mode", job->req.keyframe_images_bytes.empty()
                                     ? "T2VA"
                                     : ("FL2VA-" + std::to_string(job->req.keyframe_images_bytes.size()) + "F")},
                        {"width", job->res.width},
                        {"height", job->res.height},
                        {"num_frames", job->res.num_frames},
                        {"steps", job->req.num_steps},
                        {"encode_s", job->res.encode_s},
                        {"denoise_s", job->res.denoise_s},
                        {"vvae_s", job->res.vvae_s},
                        {"avae_s", job->res.avae_s},
                        {"mux_s", job->res.mux_s},
                        {"total_s", job->res.total_s},
                        {"status", "completed"},
                    });
                    if (recent_jobs.size() > 40) recent_jobs.erase(recent_jobs.begin());
                    active_job_id.clear();
                } catch (const std::exception& e) {
                    std::lock_guard lk(mu);
                    job->status = "failed";
                    job->error = e.what();
                    active_job_id.clear();
                }
                cv.notify_all();
            }
        });

        httplib::Server svr;
        svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });
        svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
            json o = {{"object", "list"}, {"data", {{{"id", model_name}, {"object", "model"}}}}};
            res.set_content(o.dump(), "application/json");
        });
        svr.Get("/stats", [&](const httplib::Request&, httplib::Response& res) {
            std::lock_guard lk(mu);
            int running = 0, queued = 0;
            for (const auto& [_, j] : jobs) {
                if (j->status == "in_progress") ++running;
                else if (j->status == "queued") ++queued;
            }
            const char* da = std::getenv("G2E_H3_DIRECT_ATTN");
            const char* scal = std::getenv("SCAL_CONFIG_FILE_PATH");
            const char* fp8 = std::getenv("G2E_H3_FP8_SDPA");
            const char* sp = std::getenv("G2E_H3_SPARSE");
            const bool direct_on = da ? (da[0] != '0') : (scal && scal[0] != '\0');
            const bool sparse_on = direct_on && (!sp || sp[0] != '0');
            std::string attn_mode = direct_on
                ? (sparse_on ? "Direct MME+24-TPC+SRAM (A:Dense / B:BlockSparse)" : "Direct MME + 24-TPC + SRAM (Dense)")
                : ((fp8 && fp8[0] != '0') ? "FP8 Flash SDPA" : "BF16 SDPA");
            json o = {
                {"model", model_name},
                {"engine", "gaudi2-engine-h3"},
                {"attention_backend", attn_mode},
                {"running", running},
                {"queued", queued},
                {"completed", completed_count.load()},
                {"last_total_s", last_total_s},
                {"recent_jobs", recent_jobs},
            };
            if (!active_job_id.empty()) {
                o["active_job"] = {
                    {"id", active_job_id},
                    {"mode", active_req.keyframe_images_bytes.empty()
                                 ? "T2VA"
                                 : ("FL2VA-" + std::to_string(active_req.keyframe_images_bytes.size()) + "F")},
                    {"width", active_req.width},
                    {"height", active_req.height},
                    {"duration_s", active_req.duration_seconds},
                    {"steps", active_req.num_steps},
                    {"elapsed_s", std::max<int64_t>(0, unix_now() - active_started_at)},
                };
            }
            res.set_content(o.dump(), "application/json");
        });

        auto submit_job = [&](const httplib::Request& req) -> std::shared_ptr<VideoJob> {
            const std::string id = "vid_" + std::to_string(unix_now()) + "_" + std::to_string(seq.fetch_add(1) + 1);
            const std::string mp4 = temp_dir + "/" + id + ".mp4";
            auto job = std::make_shared<VideoJob>();
            job->id = id;
            job->created_at = unix_now();
            job->req = parse_video_request(req, mp4);
            {
                std::lock_guard lk(mu);
                jobs[id] = job;
                pending.push(id);
            }
            cv.notify_all();
            return job;
        };

        svr.Post("/v1/videos", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                auto job = submit_job(req);
                res.set_content(job_to_json(*job, model_name).dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json({{"error", e.what()}}).dump(), "application/json");
            }
        });

        svr.Post("/v1/videos/sync", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                auto job = submit_job(req);
                {
                    std::unique_lock lk(mu);
                    cv.wait(lk, [&] { return job->status == "completed" || job->status == "failed"; });
                }
                if (job->status == "failed") {
                    res.status = 500;
                    res.set_content(json({{"error", job->error}}).dump(), "application/json");
                    return;
                }
                std::ifstream f(job->res.output_mp4, std::ios::binary);
                std::ostringstream ss;
                ss << f.rdbuf();
                res.set_content(ss.str(), "video/mp4");
                std::remove(job->res.output_mp4.c_str());
                std::lock_guard lk(mu);
                jobs.erase(job->id);
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json({{"error", e.what()}}).dump(), "application/json");
            }
        });

        svr.Get(R"(/v1/videos/([^/]+))", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string id = req.matches[1];
            std::lock_guard lk(mu);
            auto it = jobs.find(id);
            if (it == jobs.end()) {
                res.status = 404;
                res.set_content("{\"error\":\"not found\"}", "application/json");
                return;
            }
            res.set_content(job_to_json(*it->second, model_name).dump(), "application/json");
        });

        svr.Get(R"(/v1/videos/([^/]+)/content)", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string id = req.matches[1];
            std::string mp4;
            {
                std::lock_guard lk(mu);
                auto it = jobs.find(id);
                if (it == jobs.end() || it->second->status != "completed") {
                    res.status = 404;
                    res.set_content("{\"error\":\"not ready\"}", "application/json");
                    return;
                }
                mp4 = it->second->res.output_mp4;
            }
            std::ifstream f(mp4, std::ios::binary);
            if (!f) {
                res.status = 404;
                res.set_content("{\"error\":\"file missing\"}", "application/json");
                return;
            }
            std::ostringstream ss;
            ss << f.rdbuf();
            res.set_content(ss.str(), "video/mp4");
        });

        svr.Delete(R"(/v1/videos/([^/]+))", [&](const httplib::Request& req, httplib::Response& res) {
            const std::string id = req.matches[1];
            std::lock_guard lk(mu);
            auto it = jobs.find(id);
            if (it != jobs.end()) {
                if (!it->second->res.output_mp4.empty()) std::remove(it->second->res.output_mp4.c_str());
                jobs.erase(it);
            }
            res.set_content(json({{"id", id}, {"deleted", true}}).dump(), "application/json");
        });

        std::printf("[g2e_h3_server] listening on http://%s:%d (%s)\n", host.c_str(), port, model_name.c_str());
        svr.listen(host, port);
        stop.store(true);
        cv.notify_all();
        worker.join();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
