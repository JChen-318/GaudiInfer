// Streaming parser for Qwen3 output (after vllm/parser/qwen3.py): optional <think> reasoning split and the XML tool
// call format
//     <tool_call>
//     <function=NAME>
//     <parameter=KEY>
//     VALUE
//     </parameter>
//     </function>
//     </tool_call>
// Text that may still turn into a tag is held back until it is decided. A tool call is reported once it is complete
// (name and the full JSON arguments in one delta). Parameter values are strings, converted to the type declared in
// the request's tool schema (integer, number, boolean, object, array) when it parses.
#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace g2e {

struct ToolCallOut {
    std::string id, name, arguments;  // arguments: JSON object text
};

struct ParserDelta {
    std::string content, reasoning;
    std::vector<ToolCallOut> tool_calls;
    bool empty() const { return content.empty() && reasoning.empty() && tool_calls.empty(); }
};

class Qwen3Parser {
public:
    // reasoning: split <think> ... </think> into `reasoning` (generation starts inside the think block when
    // thinking is enabled); tools: the request's tools (null/empty: no tool parsing).
    Qwen3Parser(bool reasoning, bool thinking_enabled, const nlohmann::ordered_json& tools);
    ParserDelta feed(const std::string& text, bool final);
    int tool_calls_seen() const { return calls_; }

private:
    enum class Mode { Reasoning, Content, Tool };
    ToolCallOut parse_call(const std::string& block) const;
    Mode mode_;
    bool tools_on_;
    nlohmann::ordered_json tools_;
    std::string buf_;
    int calls_ = 0;
};

std::string random_id(const std::string& prefix, size_t hex_chars = 32);

}  // namespace g2e
