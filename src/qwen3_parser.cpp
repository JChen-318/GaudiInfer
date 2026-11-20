#include "g2e/qwen3_parser.hpp"

#include <random>

namespace g2e {

namespace {
const std::string kThinkStart = "<think>", kThinkEnd = "</think>", kToolStart = "<tool_call>", kToolEnd = "</tool_call>";
const std::string kFunc = "<function=", kFuncEnd = "</function>", kParam = "<parameter=", kParamEnd = "</parameter>";

// Length of the longest suffix of `s` that is a proper prefix of one of `tags` (text that may still become a tag).
size_t held_back(const std::string& s, std::initializer_list<const std::string*> tags) {
    size_t best = 0;
    for (const auto* t : tags)
        for (size_t n = std::min(s.size(), t->size() - 1); n > best; --n)
            if (s.compare(s.size() - n, n, *t, 0, n) == 0) { best = n; break; }
    return best;
}
std::string trim_newlines(std::string v) {  // one leading and one trailing newline: the template's markup
    if (!v.empty() && v.front() == '\n') v.erase(0, 1);
    if (!v.empty() && v.back() == '\n') v.pop_back();
    return v;
}
std::string trim(const std::string& s) {
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
}  // namespace

std::string random_id(const std::string& prefix, size_t hex_chars) {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    static const char* hex = "0123456789abcdef";
    std::string s = prefix;
    for (size_t i = 0; i < hex_chars; ++i) s.push_back(hex[rng() & 15]);
    return s;
}

Qwen3Parser::Qwen3Parser(bool reasoning, bool thinking_enabled, const nlohmann::ordered_json& tools)
    : mode_(reasoning && thinking_enabled ? Mode::Reasoning : Mode::Content),
      tools_on_(tools.is_array() && !tools.empty()), tools_(tools) {}

ToolCallOut Qwen3Parser::parse_call(const std::string& block) const {
    ToolCallOut call;
    call.id = random_id("call_");
    size_t f = block.find(kFunc);
    std::string body = block;
    if (f != std::string::npos) {
        const size_t close = block.find('>', f);
        call.name = trim(block.substr(f + kFunc.size(), close == std::string::npos ? std::string::npos : close - f - kFunc.size()));
        body = close == std::string::npos ? std::string() : block.substr(close + 1);
        if (const size_t e = body.find(kFuncEnd); e != std::string::npos) body.resize(e);
    }
    // declared parameter types of this function
    nlohmann::ordered_json props = nlohmann::ordered_json::object();
    for (const auto& t : tools_)
        if (t.contains("function") && t["function"].value("name", "") == call.name && t["function"].contains("parameters") &&
            t["function"]["parameters"].contains("properties"))
            props = t["function"]["parameters"]["properties"];
    nlohmann::ordered_json args = nlohmann::ordered_json::object();
    size_t p = 0;
    while ((p = body.find(kParam, p)) != std::string::npos) {
        const size_t close = body.find('>', p);
        if (close == std::string::npos) break;
        const std::string key = trim(body.substr(p + kParam.size(), close - p - kParam.size()));
        size_t end = body.find(kParamEnd, close + 1);
        const size_t next = body.find(kParam, close + 1);
        if (end == std::string::npos || (next != std::string::npos && next < end)) end = next;  // unterminated value
        const std::string raw = trim_newlines(body.substr(close + 1, end == std::string::npos ? std::string::npos : end - close - 1));
        nlohmann::ordered_json value = raw;
        const std::string type = props.contains(key) && props[key].is_object() ? props[key].value("type", "string") : "string";
        if (type != "string") {
            const auto parsed = nlohmann::ordered_json::parse(trim(raw), nullptr, false);
            if (!parsed.is_discarded()) value = parsed;
        }
        args[key] = value;
        if (end == std::string::npos) break;
        p = end;
    }
    call.arguments = args.dump();
    return call;
}

ParserDelta Qwen3Parser::feed(const std::string& text, bool final) {
    buf_ += text;
    ParserDelta d;
    while (true) {
        if (mode_ == Mode::Reasoning) {
            if (buf_.compare(0, kThinkStart.size(), kThinkStart) == 0) buf_.erase(0, kThinkStart.size());
            size_t e = buf_.find(kThinkEnd);
            const size_t tc = tools_on_ ? buf_.find(kToolStart) : std::string::npos;  // a tool call ends reasoning too
            if (tc != std::string::npos && (e == std::string::npos || tc < e)) {
                d.reasoning += buf_.substr(0, tc);
                buf_.erase(0, tc);
                mode_ = Mode::Content;
                continue;
            }
            if (e != std::string::npos) {
                d.reasoning += buf_.substr(0, e);
                buf_.erase(0, e + kThinkEnd.size());
                mode_ = Mode::Content;
                continue;
            }
            const size_t hold = final ? 0 : held_back(buf_, {&kThinkEnd, &kToolStart});
            d.reasoning += buf_.substr(0, buf_.size() - hold);
            buf_.erase(0, buf_.size() - hold);
            break;
        }
        if (mode_ == Mode::Content) {
            if (tools_on_) {
                size_t s = buf_.find(kToolStart);
                const size_t fs = buf_.find(kFunc);  // fallback: <function= without <tool_call>
                if (fs != std::string::npos && (s == std::string::npos || fs < s)) s = fs;
                if (s != std::string::npos) {
                    d.content += buf_.substr(0, s);
                    buf_.erase(0, s);
                    mode_ = Mode::Tool;
                    continue;
                }
            }
            const size_t hold = final || !tools_on_ ? 0 : held_back(buf_, {&kToolStart, &kFunc});
            d.content += buf_.substr(0, buf_.size() - hold);
            buf_.erase(0, buf_.size() - hold);
            break;
        }
        // Tool: wait for the end of the call
        const bool wrapped = buf_.compare(0, kToolStart.size(), kToolStart) == 0;
        const std::string& end_tag = wrapped ? kToolEnd : kFuncEnd;
        const size_t e = buf_.find(end_tag);
        if (e == std::string::npos && !final) break;
        const std::string block = e == std::string::npos ? buf_ : buf_.substr(0, e);
        buf_.erase(0, e == std::string::npos ? buf_.size() : e + end_tag.size());
        auto call = parse_call(block);
        if (!call.name.empty()) {
            d.tool_calls.push_back(std::move(call));
            ++calls_;
        }
        mode_ = Mode::Content;
        // whitespace between consecutive calls is not content
        const size_t a = buf_.find_first_not_of(" \t\r\n");
        if (a != std::string::npos && (buf_.compare(a, kToolStart.size(), kToolStart) == 0 || buf_.compare(a, kFunc.size(), kFunc) == 0))
            buf_.erase(0, a);
        if (buf_.empty()) break;
    }
    return d;
}

}  // namespace g2e
