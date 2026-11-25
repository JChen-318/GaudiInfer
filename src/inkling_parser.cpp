#include "g2e/inkling_parser.hpp"

#include <nlohmann/json.hpp>

#include <initializer_list>

namespace g2e {

namespace {
const std::string kModel = "<|message_model|>", kEnd = "<|end_message|>", kEos = "<|content_model_end_sampling|>";
const std::string kThinking = "<|content_thinking|>", kText = "<|content_text|>", kToolText = "<|content_invoke_tool_text|>";
const std::string kToolJson = "<|content_invoke_tool_json|>", kToolError = "<|content_tool_error|>", kContent = "<|content_";

// length of the longest suffix of s that is a proper prefix of a marker (text that may still become one)
size_t held_back(const std::string& s) {
    static const std::string* tags[] = {&kModel, &kEnd, &kEos, &kThinking, &kText, &kToolText, &kToolJson, &kToolError};
    size_t best = 0;
    for (const auto* t : tags)
        for (size_t n = std::min(s.size(), t->size() - 1); n > best; --n)
            if (s.compare(s.size() - n, n, *t, 0, n) == 0) { best = n; break; }
    return best;
}
}  // namespace

ParserDelta InklingParser::feed(const std::string& text, bool final) {
    ParserDelta d;
    buf_ += text;
    while (true) {
        if (state_ == State::Outside || state_ == State::Header) {
            // the next marker decides the block kind; stray text between blocks is content
            const size_t m = buf_.find("<|");
            if (m == std::string::npos) {
                if (state_ == State::Header) name_ += buf_;
                else d.content += buf_;
                buf_.clear();
                break;
            }
            if (m > 0) {
                if (state_ == State::Header) name_ += buf_.substr(0, m);
                else d.content += buf_.substr(0, m);
                buf_.erase(0, m);
            }
            auto starts = [&](const std::string& t) { return buf_.compare(0, t.size(), t) == 0; };
            auto partial = [&]() { return !final && held_back(buf_) == buf_.size(); };
            if (starts(kModel)) {
                buf_.erase(0, kModel.size());
                state_ = State::Header;
                name_.clear();
            } else if (starts(kThinking)) {
                buf_.erase(0, kThinking.size());
                state_ = State::Thinking;
            } else if (starts(kText) || starts(kToolText)) {
                buf_.erase(0, starts(kText) ? kText.size() : kToolText.size());
                state_ = State::Text;
            } else if (starts(kToolJson)) {
                buf_.erase(0, kToolJson.size());
                state_ = State::ToolJson;
            } else if (starts(kToolError)) {
                buf_.erase(0, kToolError.size());
                state_ = State::Drop;
            } else if (starts(kEos) || starts(kEnd)) {
                buf_.erase(0, starts(kEos) ? kEos.size() : kEnd.size());
                state_ = State::Outside;
            } else if (partial()) {
                break;  // wait for the rest of the marker
            } else {
                // not a marker: literal text
                if (state_ == State::Header) name_ += buf_.substr(0, 2);
                else d.content += buf_.substr(0, 2);
                buf_.erase(0, 2);
            }
            continue;
        }
        // inside a body: up to <|end_message|> (or the end-of-sampling marker)
        size_t e = buf_.find(kEnd);
        size_t elen = kEnd.size();
        if (const size_t x = buf_.find(kEos); x != std::string::npos && x < e) { e = x; elen = kEos.size(); }
        if (state_ == State::ToolJson || state_ == State::Drop) {
            if (e == std::string::npos && !final) break;  // a tool block is reported once complete
            const std::string body = buf_.substr(0, e);
            buf_.erase(0, e == std::string::npos ? buf_.size() : e + elen);
            if (state_ == State::ToolJson) {
                ToolCallOut c;
                c.id = random_id("call_");
                const auto j = nlohmann::ordered_json::parse(body, nullptr, false);
                std::string trimmed = name_;
                while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\n')) trimmed.pop_back();
                if (!j.is_discarded() && j.is_object()) {
                    c.name = j.value("name", trimmed);
                    const auto args = j.contains("args") ? j["args"] : (j.contains("arguments") ? j["arguments"] : nlohmann::ordered_json::object());
                    c.arguments = args.is_string() ? args.get<std::string>() : args.dump();
                } else {
                    c.name = trimmed;
                    c.arguments = body;
                }
                d.tool_calls.push_back(std::move(c));
                ++calls_;
            }
            state_ = State::Outside;
            if (e == std::string::npos) break;
            continue;
        }
        std::string& out = state_ == State::Thinking ? d.reasoning : d.content;
        if (e != std::string::npos) {
            out += buf_.substr(0, e);
            buf_.erase(0, e + elen);
            state_ = State::Outside;
            continue;
        }
        const size_t keep = final ? 0 : held_back(buf_);
        out += buf_.substr(0, buf_.size() - keep);
        buf_.erase(0, buf_.size() - keep);
        break;
    }
    return d;
}

}  // namespace g2e
