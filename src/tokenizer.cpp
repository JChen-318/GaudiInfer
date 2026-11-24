#include "g2e/tokenizer.hpp"

#include "unicode.h"        // third_party/llama
#include "json.h"           // third_party/llama (common_json)
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace g2e {

namespace {
// llama.cpp has exact implementations of these patterns under its own spelling of the case-insensitive group.
std::string llama_regex(const std::string& pattern) {
    const std::string qwen2 =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    const std::string qwen35 =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    const std::string ci = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";
    if (pattern == qwen2 || pattern == qwen35) return ci + pattern.substr(pattern.find(')') + 1);
    // other patterns (o200k / GPT-4o: Inkling): spell every case-insensitive contraction group out
    std::string out = pattern;
    const std::string group = "(?i:'s|'t|'re|'ve|'m|'ll|'d)";
    for (size_t at = out.find(group); at != std::string::npos; at = out.find(group, at + ci.size())) out.replace(at, group.size(), ci);
    return out;
}
// UTF-8 length of the sequence starting with byte c (1 for continuation/invalid bytes)
size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}
}  // namespace

Tokenizer::Tokenizer(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("tokenizer: cannot open " + path);
    const auto j = nlohmann::json::parse(f);
    const auto& model = j.at("model");
    if (model.at("type") != "BPE") throw std::runtime_error("tokenizer: only BPE models are supported");
    int32_t max_id = -1;
    for (const auto& [tok, id] : model.at("vocab").items()) {
        vocab_[tok] = id.get<int32_t>();
        max_id = std::max(max_id, id.get<int32_t>());
    }
    for (const auto& a : j.at("added_tokens")) max_id = std::max(max_id, a.at("id").get<int32_t>());
    bytes_.assign(size_t(max_id) + 1, std::string());
    special_.assign(size_t(max_id) + 1, false);
    for (const auto& [tok, id] : vocab_) {
        std::string raw;
        for (size_t i = 0; i < tok.size();) {
            const size_t n = utf8_len(static_cast<unsigned char>(tok[i]));
            raw.push_back(char(unicode_utf8_to_byte(tok.substr(i, n))));
            i += n;
        }
        bytes_[size_t(id)] = raw;
    }
    ignore_merges_ = model.value("ignore_merges", false);
    int32_t rank = 0;
    for (const auto& m : model.at("merges")) {
        std::string key = m.is_array() ? m[0].get<std::string>() + " " + m[1].get<std::string>() : m.get<std::string>();
        ranks_.emplace(std::move(key), rank++);
    }
    for (const auto& a : j.at("added_tokens")) {
        const auto id = a.at("id").get<int32_t>();
        const auto content = a.at("content").get<std::string>();
        bytes_[size_t(id)] = content;
        special_[size_t(id)] = a.value("special", false);
        added_.push_back({content, id});
    }
    std::sort(added_.begin(), added_.end(), [](const auto& x, const auto& y) { return x.first.size() > y.first.size(); });
    // pre-tokenizer: the Split regex of the Sequence (ByteLevel itself does not split here: use_regex false)
    const auto& pre = j.at("pre_tokenizer");
    auto find_split = [&](const nlohmann::json& p) -> std::string {
        if (p.value("type", "") == "Split") return p.at("pattern").at("Regex").get<std::string>();
        return {};
    };
    if (pre.value("type", "") == "Sequence") {
        for (const auto& p : pre.at("pretokenizers"))
            if (auto r = find_split(p); !r.empty()) regex_ = r;
    } else {
        regex_ = find_split(pre);
    }
    if (regex_.empty()) throw std::runtime_error("tokenizer: no Split pre-tokenizer regex found");
    regex_ = llama_regex(regex_);
}

int32_t Tokenizer::token_id(const std::string& text) const {
    for (const auto& [content, id] : added_)
        if (content == text) return id;
    // a plain token: its byte-level spelling
    std::string spelled;
    for (unsigned char c : text) spelled += unicode_byte_to_utf8(c);
    const auto it = vocab_.find(spelled);
    return it == vocab_.end() ? -1 : it->second;
}

void Tokenizer::bpe(const std::string& word, std::vector<int32_t>& out) const {
    {
        std::lock_guard lock(cache_mutex_);
        if (auto it = cache_.find(word); it != cache_.end()) {
            out.insert(out.end(), it->second.begin(), it->second.end());
            return;
        }
    }
    if (ignore_merges_) {  // a word that is itself a vocabulary entry is one token (tokenizers' ignore_merges)
        if (const auto it = vocab_.find(word); it != vocab_.end()) {
            out.push_back(it->second);
            return;
        }
    }
    std::vector<std::string> sym;
    for (size_t i = 0; i < word.size();) {
        const size_t n = utf8_len(static_cast<unsigned char>(word[i]));
        sym.push_back(word.substr(i, n));
        i += n;
    }
    while (sym.size() > 1) {
        int best = INT_MAX;
        size_t at = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            const auto it = ranks_.find(sym[i] + " " + sym[i + 1]);
            if (it != ranks_.end() && it->second < best) {
                best = it->second;
                at = i;
            }
        }
        if (best == INT_MAX) break;
        // merge every occurrence of this pair, left to right
        const std::string a = sym[at], b = sym[at + 1];
        std::vector<std::string> next;
        for (size_t i = 0; i < sym.size();) {
            if (i + 1 < sym.size() && sym[i] == a && sym[i + 1] == b) {
                next.push_back(a + b);
                i += 2;
            } else {
                next.push_back(sym[i++]);
            }
        }
        sym.swap(next);
    }
    std::vector<int32_t> ids;
    for (const auto& s : sym) {
        const auto it = vocab_.find(s);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else {  // no byte fallback in these vocabularies: every single byte symbol exists
            for (size_t i = 0; i < s.size();) {
                const size_t n = utf8_len(static_cast<unsigned char>(s[i]));
                const auto b = vocab_.find(s.substr(i, n));
                if (b == vocab_.end()) throw std::runtime_error("tokenizer: unknown symbol");
                ids.push_back(b->second);
                i += n;
            }
        }
    }
    out.insert(out.end(), ids.begin(), ids.end());
    std::lock_guard lock(cache_mutex_);
    if (cache_.size() > 200000) cache_.clear();
    cache_.emplace(word, std::move(ids));
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int32_t> out;
    auto plain = [&](const std::string& piece) {
        if (piece.empty()) return;
        for (const auto& word : unicode_regex_split(piece, {regex_}, true)) bpe(word, out);
    };
    if (!parse_special) {
        plain(text);
        return out;
    }
    size_t start = 0, i = 0;
    while (i < text.size()) {
        bool matched = false;
        if (text[i] == '<') {  // every added token of these vocabularies starts with '<'
            for (const auto& [content, id] : added_) {
                if (text.compare(i, content.size(), content) == 0) {
                    plain(text.substr(start, i - start));
                    out.push_back(id);
                    i += content.size();
                    start = i;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) ++i;
    }
    plain(text.substr(start));
    return out;
}

std::string Tokenizer::decode(std::span<const int32_t> ids, bool skip_special) const {
    std::string s;
    for (auto id : ids) {
        if (id < 0 || size_t(id) >= bytes_.size()) continue;
        if (skip_special && special_[size_t(id)]) continue;
        s += bytes_[size_t(id)];
    }
    return s;
}

std::string Detokenizer::push(int32_t id) {
    if (!(skip_ && tok_.is_special(id)) && id >= 0 && size_t(id) < tok_.vocab_size()) pending_ += tok_.token_bytes(id);
    // emit everything up to the last complete UTF-8 sequence
    size_t cut = pending_.size();
    for (size_t back = 1; back <= 4 && back <= pending_.size(); ++back) {
        const unsigned char c = static_cast<unsigned char>(pending_[pending_.size() - back]);
        if ((c & 0xC0) == 0x80) continue;  // continuation byte
        if (utf8_len(c) > back) cut = pending_.size() - back;
        break;
    }
    std::string out = pending_.substr(0, cut);
    pending_.erase(0, cut);
    return out;
}

std::string Detokenizer::flush() {
    std::string out;
    out.swap(pending_);
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
struct ChatTemplate::Impl {
    jinja::program prog;
};

ChatTemplate::ChatTemplate(const std::string& source) : source_(source), impl_(std::make_unique<Impl>()) {
    jinja::lexer lexer;
    const auto tokens = lexer.tokenize(source_);
    impl_->prog = jinja::parse_from_tokens(tokens);
}
ChatTemplate::~ChatTemplate() = default;

std::string ChatTemplate::render(const nlohmann::ordered_json& vars) const {
    jinja::context ctx(source_);
    jinja::global_from_json(ctx, common_json::parse(vars.dump()), false);
    jinja::runtime runtime(ctx);
    const jinja::value result = runtime.execute(impl_->prog);
    return jinja::runtime::gather_string_parts(result)->as_string().str();
}

}  // namespace g2e
