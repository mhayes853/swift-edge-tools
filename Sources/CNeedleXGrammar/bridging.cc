#include "bridging.h"
#include "ebnf_syntax.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <thread>
#include <dlpack/dlpack.h>
#include <picojson/picojson.h>
#include <xgrammar/object.h>
#include <xgrammar/grammar.h>
#include <xgrammar/tokenizer_info.h>
#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>

struct XGrammarCompilerHandle {
    xgrammar::TokenizerInfo tokenizer_info;
    std::optional<xgrammar::GrammarCompiler> compiler;
    int64_t memory_limit;
    int64_t max_threads;
    bool is_cache_enabled;
};

struct XGrammarMatcherHandle {
    xgrammar::GrammarMatcher matcher;
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

namespace {

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

static void add_call_body_rule(
    EbnfSyntax& merged,
    const std::vector<std::pair<std::string, EbnfSyntax>>& tool_rule_sets,
    std::function<std::string(const std::string&, const std::string&)>&& build_call_rule
) {
    std::vector<std::string> call_rule_names;
    call_rule_names.reserve(tool_rule_sets.size());
    for (const auto& [tool_name, _] : tool_rule_sets) {
        const std::string call_rule_name = tool_name + "_call";
        const std::string args_rule_name = tool_name + "_args";
        merged.rules[call_rule_name] = build_call_rule(tool_name, args_rule_name);
        call_rule_names.push_back(call_rule_name);
    }

    std::string call_body_expr;
    for (size_t i = 0; i < call_rule_names.size(); ++i) {
        if (i != 0) call_body_expr += " | ";
        call_body_expr += call_rule_names[i];
    }
    merged.rules["call_body"] = call_body_expr;
}

static xgrammar::Grammar needle_tool_grammar(const std::vector<ToolDefinition>& tools) {
    if (tools.empty()) return xgrammar::Grammar::FromEBNF("root ::= \"<tool_call>[]\"");

    std::vector<std::pair<std::string, EbnfSyntax>> tool_rule_sets;
    tool_rule_sets.reserve(tools.size());

    for (const auto& tool : tools) {
        const auto ebnf = xgrammar::Grammar::FromJSONSchema(tool.arguments_schema, false, 0)
          .ToString();
        EbnfSyntax tool_syntax = EbnfSyntax::from_string(ebnf);
        tool_syntax.remove_json_whitespaces();
        tool_syntax.rename_rules({{"root", tool.name + "_args"}});
        tool_rule_sets.push_back({tool.name, std::move(tool_syntax)});
    }

    EbnfSyntax merged;
    merged.merge_with(tool_rule_sets);
    add_call_body_rule(merged, tool_rule_sets, [](auto tool_name, auto args_rule_name) {
        const std::string call_prefix =
            EbnfSyntax::escape_string_literal("{\"name\":\"" + tool_name + "\",\"arguments\":");
        return "(\"" + call_prefix + "\" " + args_rule_name + " \"}\")";
    });
    merged.rules["root"] = "\"<tool_call>[\" call_body (\",\" call_body)* \"]\"";
    merged.remove_unreachable_rules();
    return xgrammar::Grammar::FromEBNF(merged.ebnf());
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
    return new XGrammarCompilerHandle{
        tokenizer_info,
        std::nullopt,
        kNeedleXGrammarCompilerNoMemoryLimit,
        kNeedleXGrammarCompilerHardwareConcurrency,
        true
    };
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
    return new XGrammarMatcherHandle{xgrammar::GrammarMatcher(compiled), bitmask_size};
}

needle_xgrammar_matcher_t needle_xgrammar_matcher_fork(needle_xgrammar_matcher_t matcher) {
    if (!matcher) return nullptr;
    return new XGrammarMatcherHandle{static_cast<XGrammarMatcherHandle*>(matcher)->matcher.Fork()};
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

void needle_xgrammar_matcher_destroy(needle_xgrammar_matcher_t matcher) {
    if (matcher) delete static_cast<XGrammarMatcherHandle*>(matcher);
}

needle_xgrammar_grammar_t needle_xgrammar_grammar_init(const char* tools_json) {
    if (!tools_json) return nullptr;

    std::string tools_json_string(tools_json);
    const auto tool_definitions = ToolDefinition::parse(tools_json_string);
    if (!tool_definitions.has_value()) return nullptr;
    return new XGrammarGrammarHandle{needle_tool_grammar(*tool_definitions)};
}

void needle_xgrammar_grammar_destroy(needle_xgrammar_grammar_t grammar) {
    if (grammar) delete static_cast<XGrammarGrammarHandle*>(grammar);
}

}
