// Streaming parser for Inkling-2 output (the response_template of its tokenizer_config.json, after ollama's
// model/parsers/inkling2.go). The model writes blocks
//     [<|message_model|>NAME]<|content_KIND|>BODY<|end_message|>
// and ends with <|content_model_end_sampling|>. KIND thinking -> reasoning, text / invoke_tool_text -> content,
// invoke_tool_json -> a tool call (BODY = {"name": ..., "args": {...}}), tool_error -> dropped. Text that may still
// turn into a marker is held back; a tool call is reported once its block is complete.
#pragma once

#include "g2e/qwen3_parser.hpp"  // ParserDelta, ToolCallOut, random_id

#include <string>

namespace g2e {

class InklingParser {
public:
    InklingParser() = default;
    ParserDelta feed(const std::string& text, bool final);
    int tool_calls_seen() const { return calls_; }

private:
    enum class State { Outside, Header, Thinking, Text, ToolJson, Drop };
    State state_ = State::Outside;
    std::string buf_, name_;
    int calls_ = 0;
};

}  // namespace g2e
