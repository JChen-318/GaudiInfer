#include "g2e/h3_pipeline.hpp"
#include "g2e/vision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>

namespace g2e {

namespace {

using Clock = std::chrono::steady_clock;
double sec_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Write 32kHz stereo PCM16 WAV file from planar float [2, N] in [-1, 1]
void write_wav_stereo_pcm16(const std::string& path, const float* planar, uint32_t n_samples, uint32_t sample_rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("H3Pipeline: cannot write wav " + path);
    const uint16_t num_ch = 2, bits = 16, block_align = num_ch * (bits / 8);
    const uint32_t byte_rate = sample_rate * block_align;
    const uint32_t data_bytes = n_samples * block_align;
    const uint32_t riff_size = 36 + data_bytes;
    auto w16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    f.write("RIFF", 4); w32(riff_size); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(1); w16(num_ch); w32(sample_rate); w32(byte_rate); w16(block_align); w16(bits);
    f.write("data", 4); w32(data_bytes);

    std::vector<int16_t> pcm(size_t(n_samples) * 2);
    for (uint32_t i = 0; i < n_samples; ++i) {
        for (uint32_t ch = 0; ch < 2; ++ch) {
            float v = std::clamp(planar[size_t(ch) * n_samples + i], -1.0f, 1.0f);
            pcm[size_t(i) * 2 + ch] = int16_t(std::lround(v * 32767.0f));
        }
    }
    f.write(reinterpret_cast<const char*>(pcm.data()), std::streamsize(pcm.size() * 2));
}

// Unpatchify DiT video rows [lt * ph * pw, 96] -> [24, lt, ph*2, pw*2] (matches minimax_h3_unpatchify_video_tokens)
std::vector<float> unpatchify_video(const float* rows, uint32_t lt, uint32_t ph, uint32_t pw) {
    constexpr uint32_t C = 24;
    const uint32_t lh = ph * 2, lw = pw * 2;
    std::vector<float> lat(size_t(C) * lt * lh * lw);
    for (uint32_t ti = 0; ti < lt; ++ti)
        for (uint32_t hi = 0; hi < ph; ++hi)
            for (uint32_t wi = 0; wi < pw; ++wi) {
                const float* row = rows + size_t((ti * ph + hi) * pw + wi) * 96;
                for (uint32_t c = 0; c < C; ++c)
                    for (uint32_t dh = 0; dh < 2; ++dh)
                        for (uint32_t dw = 0; dw < 2; ++dw) {
                            const size_t dst = (((size_t(c) * lt + ti) * lh + (hi * 2 + dh)) * lw + (wi * 2 + dw));
                            lat[dst] = row[(c * 2 + dh) * 2 + dw];
                        }
            }
    return lat;
}

// Unpack DiT audio rows [2 * audio_t, 32] -> [2, 32, audio_t] (matches minimax_h3_unpack_audio_tokens)
std::vector<float> unpack_audio(const float* rows, uint32_t audio_t) {
    constexpr uint32_t C = 32;
    std::vector<float> lat(size_t(2) * C * audio_t);
    for (uint32_t ch = 0; ch < 2; ++ch)
        for (uint32_t ti = 0; ti < audio_t; ++ti)
            for (uint32_t c = 0; c < C; ++c)
                lat[(size_t(ch) * C + c) * audio_t + ti] = rows[(size_t(ch) * audio_t + ti) * C + c];
    return lat;
}

}  // namespace

uint32_t H3Pipeline::align_frame_count(uint32_t frames) {
    if (frames == 0) return 1;
    while (frames % 17 != 5) ++frames;
    return frames;
}

uint32_t H3Pipeline::video_latent_t(uint32_t frames) {
    if (frames <= 5) return 2;
    return ((frames - 5) / 17) * 5 + 2;
}

uint32_t H3Pipeline::audio_latent_t(float duration_s) {
    return uint32_t(std::lround(double(duration_s) * 40.0));
}

H3Pipeline::H3Pipeline(const Device& device, const Stream& stream, H3PipelinePaths paths)
    : dev_(device), stream_(stream), paths_(std::move(paths)) {
    const auto t0 = Clock::now();
    tokenizer_ = std::make_unique<Tokenizer>(paths_.tokenizer_json);
    text_enc_ = std::make_unique<H3TextEncoder>(dev_, stream_, paths_.text_encoder_g2w);

    dit_pack_ = std::make_unique<WeightPack>(paths_.dit_g2w);
    auto st_dit = dit_pack_->upload(dev_, stream_);
    std::printf("[H3Pipe] DiT uploaded: %.2f GiB in %.2f s\n", st_dit.gib, st_dit.seconds);
    dit_ = std::make_unique<H3DiT>(dev_, stream_, *dit_pack_);

    vvae_pack_ = std::make_unique<WeightPack>(paths_.video_vae_g2w);
    auto st_vv = vvae_pack_->upload(dev_, stream_);
    std::printf("[H3Pipe] VideoVAE uploaded: %.2f GiB in %.2f s\n", st_vv.gib, st_vv.seconds);
    vvae_ = std::make_unique<H3VideoVae>(dev_, stream_, *vvae_pack_);

    avae_pack_ = std::make_unique<WeightPack>(paths_.audio_vae_g2w);
    auto st_av = avae_pack_->upload(dev_, stream_);
    std::printf("[H3Pipe] AudioVAE uploaded: %.2f GiB in %.2f s\n", st_av.gib, st_av.seconds);
    avae_ = std::make_unique<H3AudioVae>(dev_, stream_, *avae_pack_);

    if (!paths_.default_lora.empty()) {
        set_lora(paths_.default_lora, paths_.default_lora_scale);
        dit_->prepare_lora(active_lora_.get());
    }
    // Load text encoder last so total HBM stays resident for fast short-video requests (~61.7 GiB total)
    text_enc_->load_to_hbm();
    std::printf("[H3Pipe] ready in %.2f s\n", sec_since(t0));
}

H3Pipeline::~H3Pipeline() = default;

void H3Pipeline::set_lora(const std::string& lora_path, float scale) {
    if (lora_path == "none" || lora_path.empty() || scale == 0.0f) {
        active_lora_.reset();
        active_lora_path_.clear();
        active_lora_scale_ = 1.0f;
        return;
    }
    // Prefer /dev/shm/h3/<filename> when a RAM-staged copy of the requested LoRA exists (0 ms cache hit / 10 GiB/s load).
    std::string resolved = lora_path;
    try {
        const auto fname = std::filesystem::path(lora_path).filename();
        const auto shm_cand = std::filesystem::path("/dev/shm/h3") / fname;
        if (!fname.empty() && std::filesystem::exists(shm_cand)) {
            resolved = shm_cand.string();
        }
    } catch (...) {}
    if (active_lora_ && active_lora_path_ == resolved && active_lora_scale_ == scale) return;
    active_lora_.reset();
    active_lora_ = std::make_unique<H3LoraAdapter>(dev_, stream_, resolved, dit_->config(), scale);
    active_lora_path_ = resolved;
    active_lora_scale_ = scale;
}

H3GenerateResult H3Pipeline::generate(const H3GenerateRequest& req) {
    const auto t_all = Clock::now();
    H3GenerateResult res;
    const bool is_fl2va = !req.keyframe_images_bytes.empty();

    struct DecodedImg { std::vector<uint8_t> rgb; int w = 0, h = 0; };
    std::vector<DecodedImg> decoded_kf;
    std::vector<int> frame_indices = req.keyframe_frame_indices;
    if (is_fl2va) {
        if (req.keyframe_images_bytes.size() > 2)
            throw std::invalid_argument("H3Pipeline FL2VA supports at most 2 keyframes (first/last)");
        decoded_kf.reserve(req.keyframe_images_bytes.size());
        for (const auto& bytes : req.keyframe_images_bytes) {
            DecodedImg d;
            d.rgb = decode_image(bytes.data(), bytes.size(), d.w, d.h);
            decoded_kf.push_back(std::move(d));
        }
        if (frame_indices.empty()) {
            frame_indices = (decoded_kf.size() == 1) ? std::vector<int>{0} : std::vector<int>{0, -1};
        } else if (frame_indices.size() != decoded_kf.size()) {
            throw std::invalid_argument("H3Pipeline FL2VA: keyframe_frame_indices count != image count");
        }
    }

    if (req.width > 0 && req.height > 0) {
        res.width = std::max<uint32_t>(32u, (req.width / 32u) * 32u);
        res.height = std::max<uint32_t>(32u, (req.height / 32u) * 32u);
    } else if (is_fl2va) {
        // Auto-resolve canvas from first keyframe aspect ratio (short_edge=768, max_pixels=768*1344, 32-aligned).
        const double iw = double(decoded_kf[0].w), ih = double(decoded_kf[0].h);
        const double aspect = iw / std::max(1.0, ih);
        double tw = 0.0, th = 0.0;
        if (iw >= ih) {
            th = 768.0;
            tw = th * aspect;
        } else {
            tw = 768.0;
            th = tw / aspect;
        }
        constexpr double kMaxPixels = 768.0 * 1344.0;
        if (tw * th > kMaxPixels) {
            const double sc = std::sqrt(kMaxPixels / (tw * th));
            tw *= sc;
            th *= sc;
        }
        res.width = std::max<uint32_t>(32u, uint32_t(std::lround(tw / 32.0)) * 32u);
        res.height = std::max<uint32_t>(32u, uint32_t(std::lround(th / 32.0)) * 32u);
    } else {
        res.width = 1344;
        res.height = 768;
    }

    const uint32_t req_frames = uint32_t(std::lround(double(req.duration_seconds) * double(req.fps)));
    res.num_frames = align_frame_count(req_frames);
    const uint32_t lt = video_latent_t(res.num_frames);
    const uint32_t lh = res.height / 16, lw = res.width / 16;
    const uint32_t audio_t = audio_latent_t(float(res.num_frames) / float(req.fps));

    // Hot-swap LoRA if requested
    if (!req.lora_path.empty()) {
        set_lora(req.lora_path, req.lora_scale);
    } else if (!paths_.default_lora.empty()) {
        set_lora(paths_.default_lora, req.lora_scale);
    }

    // 1. Prompt (+ optional FL2VA Keyframe Vision + VideoVAE Image) Encode
    const auto t_enc = Clock::now();
    std::vector<uint16_t> prompt_embeds;
    std::vector<int32_t> prefix_token_tags;
    std::vector<float> cond_anchor;
    uint32_t text_len = 0;
    constexpr float kCondNoiseAug = 0.999f;

    if (is_fl2va) {
        std::vector<VisionImage> vis_imgs;
        vis_imgs.reserve(decoded_kf.size());
        for (size_t k = 0; k < decoded_kf.size(); ++k) {
            const auto& d = decoded_kf[k];
            std::vector<uint8_t> resized;
            const uint8_t* rgb_ptr = d.rgb.data();
            if (uint32_t(d.w) != res.width || uint32_t(d.h) != res.height) {
                resized = resize_rgb_bicubic(d.rgb.data(), d.w, d.h, int(res.width), int(res.height));
                rgb_ptr = resized.data();
            }
            // Encode keyframe image via VideoVAE Encoder -> clean patchified [ph*pw, 96] F32
            auto clean_rows = vvae_->encode_image(rgb_ptr, res.width, res.height, req.seed + k, /*sample_posterior=*/true);
            // Apply RF condition noise augmentation (0.999 * clean + 0.001 * N(0,1)) per keyframe
            std::mt19937_64 c_rng(req.seed);
            std::normal_distribution<float> c_nd(0.0f, 1.0f);
            for (float& v : clean_rows) v = kCondNoiseAug * v + (1.0f - kCondNoiseAug) * c_nd(c_rng);
            cond_anchor.insert(cond_anchor.end(), clean_rows.begin(), clean_rows.end());
            // Preprocess same resized keyframe for Qwen3-VL Vision Tower + DeepStack
            vis_imgs.push_back(preprocess_image(rgb_ptr, int(res.width), int(res.height), 65536ull, 16777216ull, 16, 2));
        }
        auto mm = text_enc_->encode_multimodal(*tokenizer_, req.prompt, vis_imgs);
        text_len = mm.seq_len;
        prompt_embeds = std::move(mm.embeds);
        prefix_token_tags = std::move(mm.token_tags);
    } else {
        const auto ids = tokenizer_->encode(req.prompt, false);
        text_len = uint32_t(ids.size());
        prompt_embeds = text_enc_->encode_text_ids(ids);
    }
    res.encode_s = sec_since(t_enc);

    auto layout = H3PackedLayout::build(text_len, lt, lh, lw, audio_t, is_fl2va, frame_indices, int(res.num_frames),
                                        prefix_token_tags);
    // Automatic stage-offload for long videos (> 42k tokens, e.g. 10-15s) to free ~24.8 GiB HBM for DiT!
    const bool offloaded = (layout.seq_len > 42000);
    if (offloaded) {
        std::printf("[H3Pipe] long sequence S=%u -> offloading TextEncoder to free ~24.8 GiB HBM\n", layout.seq_len);
        text_enc_->release_hbm();
    }

    // 2. Initial noise + DiT Denoise
    const auto t_den = Clock::now();
    std::mt19937_64 rng(req.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> vid_rows(size_t(layout.img_rows) * 96);
    std::vector<float> aud_rows(size_t(layout.audio_rows) * 32);
    for (auto& v : vid_rows) v = nd(rng);
    for (auto& a : aud_rows) a = nd(rng);

    const auto sv = H3DiT::time_shift_sigmas(req.num_steps + 1, req.flow_shift);
    const auto sa = H3DiT::time_shift_sigmas(req.num_steps + 1, req.audio_flow_shift);
    dit_->denoise(layout, vid_rows, aud_rows, prompt_embeds.data(), sv, sa, active_lora_.get(),
                  cond_anchor.empty() ? nullptr : cond_anchor.data(), kCondNoiseAug);
    res.denoise_s = sec_since(t_den);

    // Target video rows follow any prepended FL2VA condition keyframe rows
    const float* target_vid_rows = vid_rows.data() + size_t(layout.cond_rows) * 96;

    // 3. Unpack + Audio VAE Decode FIRST (~0.08-0.12 s) so WAV is ready before streaming VideoVAE -> ffmpeg!
    const auto t_av = Clock::now();
    const auto aud_lat = unpack_audio(aud_rows.data(), audio_t);
    const auto wav = avae_->decode(aud_lat.data(), audio_t);
    res.avae_s = sec_since(t_av);

    // 4 & 5. Overlapped Streaming VideoVAE Decode || Background FFmpeg libx264/AAC Mux!
    // Each 17-frame temporal chunk is RGB24-converted and piped to ffmpeg stdin while Gaudi2 decodes the next chunk.
    res.output_mp4 = req.output_mp4;
    if (!req.output_mp4.empty()) {
        const std::string raw_wav = req.output_mp4 + ".wav";
        write_wav_stereo_pcm16(raw_wav, wav.data(), audio_t * avae_->config().hop_length(),
                               avae_->config().sample_rate);
        const std::string cmd = paths_.ffmpeg_bin + " -y -loglevel error -f rawvideo -pix_fmt rgb24 -s " +
                                std::to_string(res.width) + "x" + std::to_string(res.height) +
                                " -r " + std::to_string(req.fps) + " -i pipe:0 -i \"" + raw_wav +
                                "\" -c:v libx264 -preset veryfast -threads 16 -pix_fmt yuv420p -crf 18 -c:a aac -b:a 192k -shortest \"" +
                                req.output_mp4 + "\"";
        FILE* pipe = popen(cmd.c_str(), "w");
        if (!pipe) {
            std::remove(raw_wav.c_str());
            throw std::runtime_error("H3Pipeline: popen(ffmpeg) failed");
        }

        std::mutex q_mu;
        std::condition_variable q_cv;
        std::queue<std::vector<uint8_t>> q_chunks;
        bool q_done = false;
        bool write_ok = true;
        std::thread writer([&]() {
            while (true) {
                std::vector<uint8_t> chunk;
                {
                    std::unique_lock lk(q_mu);
                    q_cv.wait(lk, [&]() { return q_done || !q_chunks.empty(); });
                    if (q_chunks.empty()) {
                        if (q_done) break;
                        continue;
                    }
                    chunk = std::move(q_chunks.front());
                    q_chunks.pop();
                }
                q_cv.notify_all();
                if (write_ok && !chunk.empty()) {
                    if (std::fwrite(chunk.data(), 1, chunk.size(), pipe) != chunk.size()) write_ok = false;
                }
            }
        });

        std::exception_ptr decode_err;
        const auto t_vv = Clock::now();
        try {
            const auto vid_lat = unpatchify_video(target_vid_rows, lt, lh / 2, lw / 2);
            vvae_->decode(vid_lat.data(), lt, lh, lw, res.num_frames, nullptr,
                          [&](std::vector<uint8_t>&& chunk) {
                              std::unique_lock lk(q_mu);
                              q_cv.wait(lk, [&]() { return q_chunks.size() < 4; });
                              q_chunks.push(std::move(chunk));
                              lk.unlock();
                              q_cv.notify_one();
                          });
        } catch (...) {
            decode_err = std::current_exception();
        }
        res.vvae_s = sec_since(t_vv);

        const auto t_mux = Clock::now();
        {
            std::lock_guard lk(q_mu);
            q_done = true;
        }
        q_cv.notify_all();
        writer.join();
        const int rc = pclose(pipe);
        std::remove(raw_wav.c_str());
        res.mux_s = sec_since(t_mux);

        if (decode_err) std::rethrow_exception(decode_err);
        if (!write_ok || rc != 0)
            throw std::runtime_error("H3Pipeline: ffmpeg mux failed with code " + std::to_string(rc));
    } else {
        const auto t_vv = Clock::now();
        const auto vid_lat = unpatchify_video(target_vid_rows, lt, lh / 2, lw / 2);
        (void)vvae_->decode(vid_lat.data(), lt, lh, lw, res.num_frames);
        res.vvae_s = sec_since(t_vv);
        res.mux_s = 0.0;
    }
    res.total_s = sec_since(t_all);
    std::printf("[H3Pipe] DONE %s %ux%u %uf -> %s in %.2f s (enc=%.2f, dit=%.2f, vvae=%.2f, avae=%.2f, mux=%.2f)\n",
                is_fl2va ? "FL2VA" : "T2VA", res.width, res.height, res.num_frames, res.output_mp4.c_str(), res.total_s,
                res.encode_s, res.denoise_s, res.vvae_s, res.avae_s, res.mux_s);
    return res;
}

}  // namespace g2e
