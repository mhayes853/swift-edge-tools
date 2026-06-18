#include "bridging.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <regex>
#include <string>
#include <thread>
#include <dlpack/dlpack.h>
#include <picojson/picojson.h>
#include <unordered_set>
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

static void replace_all(std::string& text, const std::string& needle, const std::string& replacement) {
    if (needle.empty()) return;
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        text.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
}

struct EbnfSyntax {
    using RuleExpression = std::string;

    std::unordered_map<std::string, RuleExpression> rules;

    static std::string escape_string_literal(const std::string& text) {
        std::string escaped;
        escaped.reserve(text.size());
        for (char c : text) {
            if (c == '\\' || c == '"') escaped.push_back('\\');
            escaped.push_back(c);
        }
        return escaped;
    }

    static std::string escape_multiline_literal(const std::string& text) {
        std::string escaped = EbnfSyntax::escape_string_literal(text);
        replace_all(escaped, "\n", "\\n");
        return escaped;
    };

    static EbnfSyntax from_string(const std::string& ebnf) {
        EbnfSyntax parsed;

        std::istringstream stream(ebnf);
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty()) continue;

            const size_t separator_index = line.find(EBNF_LINE_SEPARATOR);
            if (separator_index == std::string::npos) {
                throw std::runtime_error("Invalid EBNF rule line: " + line);
            }
            const auto rule_name = line.substr(0, separator_index);
            const auto rule_expression = line.substr(separator_index + EBNF_LINE_SEPARATOR.size());
            parsed.rules[rule_name] = rule_expression;
        }
        return parsed;
    }

    void merge_with(const std::vector<std::pair<std::string, EbnfSyntax>>& others) {
        std::unordered_map<std::string, RuleExpression> merged_expr_by_name;
        UniqueNameGenerator name_generator;

        for (const auto& [identifier, syntax] : others) {
            EbnfSyntax incoming = syntax;
            std::unordered_map<std::string, std::string> rename_map;

            for (const auto& [rule_name, rule_expression] : incoming.rules) {
                auto existing = merged_expr_by_name.find(rule_name);
                if (existing != merged_expr_by_name.end() && existing->second != rule_expression) {
                    rename_map.emplace(rule_name, name_generator.next(identifier, rule_name));
                }
            }

            incoming.rename_rules(rename_map);

            for (const auto& [rule_name, rule_expression] : incoming.rules) {
                auto existing = merged_expr_by_name.find(rule_name);
                if (existing != merged_expr_by_name.end()) continue;
                merged_expr_by_name.emplace(rule_name, rule_expression);
                name_generator.names.insert(rule_name);
                rules[rule_name] = rule_expression;
            }
        }
    }

    void rename_rules(const std::unordered_map<std::string, std::string>& rename_map) {
        if (rename_map.empty()) return;

        std::unordered_map<std::string, RuleExpression> renamed_rules;
        renamed_rules.reserve(rules.size());

        for (const auto& [rule_name, rule_expression] : rules) {
            const auto renamed = rename_map.find(rule_name);
            const std::string& new_name = renamed != rename_map.end() ? renamed->second : rule_name;
            renamed_rules[new_name] = rewrite_rule_references(rule_expression, rename_map);
        }

        rules = std::move(renamed_rules);
    }

    std::string ebnf() {
        std::string out;
        for (const auto& [name, expresion] : rules) {
            out += name + EBNF_LINE_SEPARATOR + expresion + "\n";
        }
        return out;
    }

    void remove_json_whitespaces() {
        for (auto& [_, rule_expression] : rules) {
            replace_all(rule_expression, "\"\\n\"", "\"\"");
            replace_all(rule_expression, "\",\\n\"", "\",\"");
            replace_all(rule_expression, "\", \"", "\",\"");
            replace_all(rule_expression, "\": \"", "\":\"");
        }
    }

    void remove_unreachable_rules(const std::string& start_rule = "root") {
        if (!rules.contains(start_rule)) return;

        std::unordered_set<std::string> reachable;
        collect_reachable_rules(start_rule, reachable);

        std::erase_if(rules, [&](const auto& entry) {
            return !reachable.contains(entry.first);
        });
    }

private:
    static const std::regex& identifier_token_pattern() {
        static const std::regex token_pattern(
            R"((("(?:\\.|[^"\\])*")|(\[(?:\\.|[^\]\\])*\])|([A-Za-z_][A-Za-z0-9_]*)))"
        );
        return token_pattern;
    }

    static std::vector<std::string> referenced_rule_names(const std::string& expr) {
        std::vector<std::string> identifiers;
        std::sregex_iterator it(expr.begin(), expr.end(), identifier_token_pattern());
        const std::sregex_iterator end;
        for (; it != end; ++it) {
            if ((*it)[4].matched) {
                identifiers.push_back((*it)[4].str());
            }
        }
        return identifiers;
    }

    void collect_reachable_rules(const std::string& rule_name, std::unordered_set<std::string>& reachable) const {
        if (reachable.contains(rule_name)) return;

        reachable.insert(rule_name);
        const auto rule_it = rules.find(rule_name);
        if (rule_it == rules.end()) return;

        for (const auto& identifier : referenced_rule_names(rule_it->second)) {
            if (rules.contains(identifier)) {
                collect_reachable_rules(identifier, reachable);
            }
        }
    }

    struct UniqueNameGenerator {
        std::unordered_set<std::string> names;

        std::string next(const std::string& identifier, const std::string& name) {
            const auto combined_name = identifier + "__" + name;
            if (!names.contains(combined_name)) return combined_name;

            size_t suffix = 1;
            while (true) {
                const std::string candiate_name = combined_name + "_" + std::to_string(suffix);
                if (!names.contains(candiate_name)) return candiate_name;
                suffix++;
            }
        }
    };

    std::string rewrite_rule_references(
        const std::string& expr,
        const std::unordered_map<std::string, std::string>& rename_map
    ) {
        std::string out;
        out.reserve(expr.size());

        std::sregex_iterator it(expr.begin(), expr.end(), identifier_token_pattern());
        const std::sregex_iterator end;
        size_t last_pos = 0;
        for (; it != end; ++it) {
            const auto& match = *it;
            out += expr.substr(last_pos, static_cast<size_t>(match.position()) - last_pos);

            if (match[4].matched) {
                const std::string identifier = match[4].str();
                auto rename = rename_map.find(identifier);
                out += rename != rename_map.end() ? rename->second : identifier;
            } else {
                out += match.str();
            }

            last_pos = static_cast<size_t>(match.position() + match.length());
        }

        out += expr.substr(last_pos);
        return out;
    }
};

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

static xgrammar::Grammar needle_tool_grammar(
    const std::vector<ToolDefinition>& tools,
    int min_tool_calls,
    int max_tool_calls
) {
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


    std::string root_expr = "\"<tool_call>[\"";
    const bool is_unbounded = max_tool_calls == kNeedleXGrammarToolCallsUnbounded;
    const int min_repeats = std::max(0, min_tool_calls - 1);
    const int max_repeats = is_unbounded ? -1 : std::max(min_repeats, max_tool_calls - 1);

    std::string call_body_expr;
    if (min_tool_calls == 0) {
        std::string inner = "call_body";
        if (is_unbounded) {
            inner += " (\",\" call_body)*";
        } else {
            inner += " (\",\" call_body){0," + std::to_string(max_repeats) + "}";
        }
        call_body_expr = " (" + inner + ")?";
    } else {
        call_body_expr = " call_body";
        if (is_unbounded) {
            call_body_expr += " (\",\" call_body){" + std::to_string(min_repeats) + ",}";
        } else {
            call_body_expr += " (\",\" call_body){" + std::to_string(min_repeats) + ","
                + std::to_string(max_repeats) + "}";
        }
    }
    root_expr += call_body_expr + " \"]\"";
    merged.rules["root"] = root_expr;
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
    const auto handle = static_cast<XGrammarMatcherHandle*>(matcher);
    return new XGrammarMatcherHandle{handle->matcher.Fork(), handle->bitmask_size};
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

needle_xgrammar_grammar_t needle_xgrammar_grammar_init(
    const char* tools_json,
    int min_tool_calls,
    int max_tool_calls
) {
    const auto is_unbounded = max_tool_calls == kNeedleXGrammarToolCallsUnbounded;
    const auto is_valid_range = min_tool_calls >= 0
            && (min_tool_calls <= max_tool_calls || is_unbounded);
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
