#include "bridging.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <dlpack/dlpack.h>
#include <picojson/picojson.h>
#include <xgrammar/object.h>
#include <xgrammar/grammar.h>
#include <xgrammar/tokenizer_info.h>
#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>

namespace {

struct XGrammarCompilerHandle {
    xgrammar::TokenizerInfo tokenizer_info;
    std::optional<xgrammar::GrammarCompiler> compiler = std::nullopt;
    int64_t memory_limit = kNeedleXGrammarCompilerNoMemoryLimit;
    int64_t max_threads = kNeedleXGrammarCompilerHardwareConcurrency;
    bool is_cache_enabled = true;
};

struct XGrammarMatcherHandle {
    xgrammar::GrammarMatcher matcher;
    xgrammar::CompiledGrammar compiled_grammar;
    int32_t bitmask_size;
};

struct XGrammarGrammarHandle {
    xgrammar::Grammar grammar;
};

struct ToolDefinition {
    std::string name;
    std::string arguments_schema;

    static std::optional<std::vector<ToolDefinition>> parse(std::string& tools_json) {
        picojson::value root;
        const auto value = picojson::parse(root, tools_json);
        if (!root.is<picojson::array>()) return std::nullopt;

        const auto& raw_definitions = root.get<picojson::array>();
        std::vector<ToolDefinition> definitions;
        definitions.reserve(raw_definitions.size());

        for (const auto raw_definition : raw_definitions) {
            if (!raw_definition.is<picojson::object>()) continue;

            const auto& object = raw_definition.get<picojson::object>();
            const auto nameit = object.find("name");
            const auto argsit = object.find("arguments");
            const auto has_name = nameit != object.end() && nameit->second.is<std::string>();
            const auto has_args = argsit != object.end() && argsit->second.is<picojson::object>();
            if (!has_name || !has_args) continue;

            definitions.push_back(ToolDefinition {
                nameit->second.get<std::string>(),
                argsit->second.serialize()
            });
        }
        return definitions;
    }
};

constexpr std::string EBNF_LINE_SEPARATOR = " ::= ";

struct EbnfRuleSet {
    std::string root_expression = "";
    std::vector<std::string> remaining_rules;
};

static std::string escape_string_literal(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (char c : text) {
        if (c == '\\' || c == '"') escaped.push_back('\\');
        escaped.push_back(c);
    }
    return escaped;
}

static std::string repeat_expression(const std::string& expression, int lo, int hi) {
    if (hi == -1) return expression + "{" + std::to_string(lo) + ",}";
    return expression + "{" + std::to_string(lo) + "," + std::to_string(hi) + "}";
}

static EbnfRuleSet ebnf_rule_set(const std::string& ebnf) {
    EbnfRuleSet rules;

    std::istringstream stream(ebnf);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;

        const size_t separator_index = line.find(EBNF_LINE_SEPARATOR);
        const auto rule_name = line.substr(0, separator_index);
        const auto rule_expression = line.substr(separator_index + EBNF_LINE_SEPARATOR.size());
        if (rules.root_expression.empty()) {
            rules.root_expression = rule_expression;
        } else {
            rules.remaining_rules.push_back(line);
        }
    }
    return rules;
}

static xgrammar::Grammar make_string_grammar(const std::string& text) {
    return xgrammar::Grammar::FromEBNF("root ::= \"" + escape_string_literal(text) + "\"");
}

static xgrammar::Grammar make_tool_call_grammar(const ToolDefinition& tool) {
    const auto arguments_grammar = xgrammar::Grammar::FromJSONSchema(
        tool.arguments_schema,
        false,
        std::nullopt,
        std::pair<std::string, std::string>{",", ":"},
        true,
        std::nullopt,
        false,
        true
    );
    return xgrammar::Grammar::Concat({
        make_string_grammar("{\"name\":\"" + tool.name + "\",\"arguments\":"),
        arguments_grammar,
        make_string_grammar("}")
    });
}

static xgrammar::GrammarCompiler make_compiler(XGrammarCompilerHandle* handle) {
    if (handle->compiler) return *handle->compiler;
    handle->compiler = xgrammar::GrammarCompiler(
        handle->tokenizer_info,
        handle->max_threads == kNeedleXGrammarCompilerHardwareConcurrency
            ? std::thread::hardware_concurrency()
            : static_cast<int>(handle->max_threads),
        handle->is_cache_enabled,
        handle->memory_limit
    );
    return *handle->compiler;
}

static std::string needle_tool_root_expression(int min_tool_calls, int max_tool_calls) {
    const bool is_unbounded = max_tool_calls == kNeedleXGrammarToolCallsUnbounded;
    const bool is_exact = max_tool_calls == kNeedleXGrammarToolCallsOnlyLowerBound;
    const int min_repeats = std::max(0, min_tool_calls - 1);
    const int max_repeats = is_unbounded ? -1 : std::max(min_repeats, max_tool_calls - 1);

    const std::string separated_call = " (\",\" call_body)";

    std::string call_body_expr;
    if (is_exact && min_tool_calls == 0) {
        call_body_expr = "";
    } else if (min_tool_calls == 0) {
        const int hi = is_unbounded ? -1 : max_repeats;
        call_body_expr = " (call_body" + repeat_expression(separated_call, 0, hi) + ")?";
    } else if (is_exact) {
        call_body_expr = min_repeats == 0
            ? " call_body"
            : " call_body" + repeat_expression(separated_call, min_repeats, min_repeats);
    } else {
        call_body_expr =
            " call_body" + repeat_expression(separated_call, min_repeats, max_repeats);
    }
    return "\"<tool_call> [\" " + call_body_expr + " \"]\"";
}

static xgrammar::Grammar needle_tool_grammar(
    const std::vector<ToolDefinition>& tools,
    int min_tool_calls,
    int max_tool_calls
) {
    if (tools.empty()) return xgrammar::Grammar::FromEBNF("root ::= \"<tool_call> []\"");

    std::vector<xgrammar::Grammar> tool_call_grammars;
    tool_call_grammars.reserve(tools.size());
    for (const auto& tool : tools) {
        tool_call_grammars.push_back(make_tool_call_grammar(tool));
    }

    const auto call_body_rules = ebnf_rule_set(
        xgrammar::Grammar::Union(tool_call_grammars).ToString()
    );

    std::string ebnf = "root ::= " + needle_tool_root_expression(min_tool_calls, max_tool_calls) + "\n";
    ebnf += "call_body ::= " + call_body_rules.root_expression + "\n";
    for (const auto& rule : call_body_rules.remaining_rules) {
        ebnf += rule + "\n";
    }
    return xgrammar::Grammar::FromEBNF(ebnf);
}

}

extern "C" {

needle_xgrammar_compiler_t needle_xgrammar_compiler_init(
    const char** encoded_vocab,
    size_t vocab_size,
    int eos_token_id
) {
    if (!encoded_vocab) return nullptr;

    std::vector<std::string> vocab(encoded_vocab, encoded_vocab + vocab_size);
    std::vector<int32_t> stop_token_ids{static_cast<int32_t>(eos_token_id)};
    xgrammar::TokenizerInfo tokenizer_info(
        vocab,
        xgrammar::VocabType::BYTE_FALLBACK,
        vocab_size,
        stop_token_ids,
        true
    );
    return new XGrammarCompilerHandle{tokenizer_info};
}

void needle_xgrammar_compiler_set_memory_limit(needle_xgrammar_compiler_t compiler, int64_t limit) {
    if (!compiler) return;
    auto handle = static_cast<XGrammarCompilerHandle*>(compiler);
    handle->compiler = std::nullopt;
    handle->memory_limit = limit;
}

void needle_xgrammar_compiler_set_max_threads(needle_xgrammar_compiler_t compiler, int64_t threads) {
    if (!compiler) return;
    auto handle = static_cast<XGrammarCompilerHandle*>(compiler);
    handle->compiler = std::nullopt;
    handle->max_threads = threads;
}

void needle_xgrammar_compiler_set_cache_enabled(needle_xgrammar_compiler_t compiler, int is_enabled) {
    if (!compiler) return;
    auto handle = static_cast<XGrammarCompilerHandle*>(compiler);
    handle->compiler = std::nullopt;
    handle->is_cache_enabled = is_enabled;
}

int64_t needle_xgrammar_compiler_cache_size_bytes(needle_xgrammar_compiler_t compiler) {
    if (!compiler) return 0;
    const auto handle = static_cast<XGrammarCompilerHandle*>(compiler);
    return handle->compiler ? handle->compiler->GetCacheSizeBytes() : 0;
}

int64_t needle_xgrammar_compiler_cache_limit_bytes(needle_xgrammar_compiler_t compiler) {
    if (!compiler) return 0;
    const auto handle = static_cast<XGrammarCompilerHandle*>(compiler);
    return handle->compiler ? handle->compiler->CacheLimitBytes() : 0;
}

void needle_xgrammar_compiler_clear_cache(needle_xgrammar_compiler_t compiler) {
    if (!compiler) return;
    const auto handle = static_cast<XGrammarCompilerHandle*>(compiler);
    if (handle->compiler) handle->compiler->ClearCache();
}

void needle_xgrammar_compiler_destroy(needle_xgrammar_compiler_t compiler) {
    if (compiler) delete static_cast<XGrammarCompilerHandle*>(compiler);
}

needle_xgrammar_matcher_t needle_xgrammar_compile_matcher(
    needle_xgrammar_compiler_t compiler,
    needle_xgrammar_grammar_t grammar
) {
    if (!compiler || !grammar) return nullptr;
    auto compiler_handle = static_cast<XGrammarCompilerHandle*>(compiler);
    auto grammar_handle = static_cast<XGrammarGrammarHandle*>(grammar);
    const auto compiled = make_compiler(compiler_handle).CompileGrammar(grammar_handle->grammar);
    const auto bitmask_size = xgrammar::GetBitmaskSize(
        compiler_handle->tokenizer_info.GetVocabSize()
    );
    return new XGrammarMatcherHandle{xgrammar::GrammarMatcher(compiled), compiled, bitmask_size};
}

needle_xgrammar_matcher_t needle_xgrammar_matcher_fork(needle_xgrammar_matcher_t matcher) {
    if (!matcher) return nullptr;
    const auto handle = static_cast<XGrammarMatcherHandle*>(matcher);
    return new XGrammarMatcherHandle{
        handle->matcher.Fork(),
        handle->compiled_grammar,
        handle->bitmask_size
    };
}

int needle_xgrammar_matcher_bitmask(needle_xgrammar_matcher_t matcher, int* bitmask) {
    if (!matcher) return -1;

    const auto handle = static_cast<XGrammarMatcherHandle*>(matcher);

    int64_t bitmask_shape[1];
    int64_t bitmask_strides[1];
    bitmask_shape[0] = handle->bitmask_size;
    bitmask_strides[0] = 1;

    DLTensor bitmask_tensor;
    bitmask_tensor.data = bitmask;
    bitmask_tensor.device = DLDevice{kDLCPU, 0};
    bitmask_tensor.ndim = 1;
    bitmask_tensor.dtype = xgrammar::GetBitmaskDLType();
    bitmask_tensor.shape = bitmask_shape;
    bitmask_tensor.strides = bitmask_strides;
    bitmask_tensor.byte_offset = 0;
    return handle->matcher.FillNextTokenBitmask(&bitmask_tensor, 0);
}

int needle_xgrammar_matcher_accept_token(needle_xgrammar_matcher_t matcher, int token) {
    if (!matcher) return -1;
    return static_cast<XGrammarMatcherHandle*>(matcher)->matcher.AcceptToken(token);
}

int needle_xgrammar_matcher_is_completed(needle_xgrammar_matcher_t matcher) {
    if (!matcher) return -1;
    return static_cast<XGrammarMatcherHandle*>(matcher)->matcher.IsCompleted();
}

int needle_xgrammar_matcher_is_terminated(needle_xgrammar_matcher_t matcher) {
    if (!matcher) return -1;
    return static_cast<XGrammarMatcherHandle*>(matcher)->matcher.IsTerminated();
}

void needle_xgrammar_matcher_rollback(needle_xgrammar_matcher_t matcher, int num_tokens) {
    if (!matcher) return;
    static_cast<XGrammarMatcherHandle*>(matcher)->matcher.Rollback(num_tokens);
}

void needle_xgrammar_matcher_reset(needle_xgrammar_matcher_t matcher) {
    if (!matcher) return;
    static_cast<XGrammarMatcherHandle*>(matcher)->matcher.Reset();
}

int64_t needle_xgrammar_matcher_memory_size_bytes(needle_xgrammar_matcher_t matcher) {
    if (!matcher) return 0;
    const auto handle = static_cast<const XGrammarMatcherHandle*>(matcher);
    return static_cast<int64_t>(handle->compiled_grammar.MemorySizeBytes());
}

void needle_xgrammar_matcher_destroy(needle_xgrammar_matcher_t matcher) {
    if (matcher) delete static_cast<XGrammarMatcherHandle*>(matcher);
}

needle_xgrammar_grammar_t needle_xgrammar_grammar_init_with_range(
    const char* tools_json,
    int min_tool_calls,
    int max_tool_calls
) {
    const auto is_unbounded = max_tool_calls == kNeedleXGrammarToolCallsUnbounded;
    const auto is_exact = max_tool_calls == kNeedleXGrammarToolCallsOnlyLowerBound;
    const auto is_valid_range = min_tool_calls >= 0
            && (min_tool_calls <= max_tool_calls || is_unbounded || is_exact);
    if (!tools_json || !is_valid_range) return nullptr;

    std::string tools_json_string(tools_json);
    const auto tool_definitions = ToolDefinition::parse(tools_json_string);
    if (!tool_definitions.has_value()) return nullptr;
    return new XGrammarGrammarHandle{
        needle_tool_grammar(*tool_definitions, min_tool_calls, max_tool_calls)
    };
}

void needle_xgrammar_grammar_destroy(needle_xgrammar_grammar_t grammar) {
    if (grammar) delete static_cast<XGrammarGrammarHandle*>(grammar);
}

}
