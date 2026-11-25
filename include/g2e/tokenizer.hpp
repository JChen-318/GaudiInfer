// Byte-level BPE tokenizer read from a Hugging Face tokenizer.json (Qwen2/Qwen3 family), plus the model's chat
// template rendered with the vendored llama.cpp Jinja engine.
//
// Encoding follows the fast tokenizer that vLLM uses: added tokens are split out first, the rest is pre-tokenized
// with the tokenizer.json Split regex (llama.cpp's exact implementations of the Qwen patterns), mapped to the GPT-2
// byte alphabet and merged by rank. Not implemented: the NFC normalizer (inputs are taken as already NFC).
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace g2e {

class Tokenizer {
public:
    explicit Tokenizer(const std::string& tokenizer_json);
    std::vector<int32_t> encode(const std::string& text, bool parse_special = true) const;
    std::string decode(std::span<const int32_t> ids, bool skip_special = false) const;
    // Raw bytes of one token (an added token: its text).
    const std::string& token_bytes(int32_t id) const { return bytes_.at(size_t(id)); }
    bool is_special(int32_t id) const { return id >= 0 && size_t(id) < special_.size() && special_[size_t(id)]; }
    int32_t token_id(const std::string& text) const;  // -1 when unknown
    size_t vocab_size() const { return bytes_.size(); }

private:
    void bpe(const std::string& word, std::vector<int32_t>& out) const;
    std::unordered_map<std::string, int32_t> vocab_;  // byte-level alphabet string -> id
    std::unordered_map<std::string, int32_t> ranks_;  // "left right" -> merge rank
    std::vector<std::string> bytes_;                  // id -> raw bytes
    std::vector<bool> special_;                       // id -> special added token
    std::vector<std::pair<std::string, int32_t>> added_;  // added tokens, longest first
    std::string regex_;
    bool ignore_merges_ = false;
    mutable std::mutex cache_mutex_;
    mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;
};

// Incremental detokenizer: feed ids, get the text that is complete UTF-8 so far.
class Detokenizer {
public:
    explicit Detokenizer(const Tokenizer& tok, bool skip_special = false) : tok_(tok), skip_(skip_special) {}
    std::string push(int32_t id);
    std::string flush();  // whatever is left (an incomplete UTF-8 tail is emitted as is)
private:
    const Tokenizer& tok_;
    bool skip_;
    std::string pending_;
};

class ChatTemplate {
public:
    explicit ChatTemplate(const std::string& source);
    ~ChatTemplate();
    // `vars` holds every template variable (messages, tools, add_generation_prompt, enable_thinking, ...).
    std::string render(const nlohmann::ordered_json& vars) const;  // key order preserved (as vLLM's dicts)
    const std::string& source() const { return source_; }
private:
    struct Impl;
    std::string source_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace g2e
