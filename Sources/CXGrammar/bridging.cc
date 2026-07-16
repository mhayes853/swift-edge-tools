#include "include/bridging.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "xgrammar/3rdparty/dlpack/include/dlpack/dlpack.h"
#include "xgrammar/3rdparty/picojson/picojson.h"
#include "xgrammar/include/xgrammar/compiler.h"
#include "xgrammar/include/xgrammar/grammar.h"
#include "xgrammar/include/xgrammar/matcher.h"
#include "xgrammar/include/xgrammar/tokenizer_info.h"

namespace {

thread_local std::string last_error;

void set_error(const std::exception& error) {
    last_error = error.what();
}

template <typename Body>
auto with_error_handling(Body&& body) -> std::invoke_result_t<Body> {
    try {
        return body();
    } catch (const std::exception& error) {
        set_error(error);
        return std::invoke_result_t<Body>{};
    }
}

struct XGrammarCompilerHandle {
    xgrammar::TokenizerInfo tokenizer_info;
    std::optional<xgrammar::GrammarCompiler> compiler = std::nullopt;
    int64_t memory_limit = kXGrammarUnlimited;
    int64_t max_threads = kXGrammarUnlimited;
    bool is_cache_enabled = true;
};

struct XGrammarMatcherHandle {
    xgrammar::GrammarMatcher matcher;
    xgrammar::CompiledGrammar compiled_grammar;
    int32_t bitmask_word_count;
};

struct XGrammarGrammarHandle {
    xgrammar::Grammar grammar;
};

int64_t normalized_memory_limit(int64_t limit) {
    return limit < 0 ? kXGrammarUnlimited : limit;
}

int64_t normalized_max_threads(int64_t threads) {
    return threads > 0 && threads <= std::numeric_limits<int>::max()
        ? threads
        : kXGrammarUnlimited;
}

xgrammar::VocabType vocab_type(xgrammar_vocab_type_t type) {
    switch (type) {
    case xgrammar_vocab_type_raw:
        return xgrammar::VocabType::RAW;
    case xgrammar_vocab_type_byte_fallback:
        return xgrammar::VocabType::BYTE_FALLBACK;
    case xgrammar_vocab_type_byte_level:
        return xgrammar::VocabType::BYTE_LEVEL;
    }
    throw std::invalid_argument("Unknown XGrammar vocabulary type.");
}

picojson::value grammar_format(const xgrammar::Grammar& grammar) {
    picojson::object format;
    format["type"] = picojson::value("grammar");
    format["grammar"] = picojson::value(grammar.ToString());
    return picojson::value(format);
}

xgrammar::Grammar grammar_from_structural_tag_json(const std::string& structural_tag_json) {
    const auto result = xgrammar::Grammar::FromStructuralTag(structural_tag_json);
    if (const auto* grammar = std::get_if<xgrammar::Grammar>(&result)) {
        return *grammar;
    }

    const auto& structural_tag_error = std::get<xgrammar::StructuralTagError>(result);
    std::visit(
        [](const auto& error) { throw std::runtime_error(error.what()); },
        structural_tag_error
    );
    throw std::runtime_error("Unknown structural tag error.");
}

xgrammar::Grammar grammar_from_structural_tag(const picojson::object& format) {
    picojson::object structural_tag;
    structural_tag["type"] = picojson::value("structural_tag");
    structural_tag["format"] = picojson::value(format);
    return grammar_from_structural_tag_json(picojson::value(structural_tag).serialize());
}

xgrammar::Grammar optional_grammar(const xgrammar::Grammar& grammar) {
    picojson::object tag;
    tag["type"] = picojson::value("optional");
    tag["content"] = grammar_format(grammar);
    return grammar_from_structural_tag(tag);
}

xgrammar::Grammar repeated_grammar(
    const xgrammar::Grammar& grammar,
    int32_t minimum,
    int32_t maximum
) {
    picojson::object tag;
    tag["type"] = picojson::value("repeat");
    tag["min"] = picojson::value(static_cast<int64_t>(minimum));
    tag["max"] = picojson::value(static_cast<int64_t>(maximum));
    tag["content"] = grammar_format(grammar);
    return grammar_from_structural_tag(tag);
}

xgrammar::GrammarCompiler& compiler_for(XGrammarCompilerHandle* handle) {
    if (!handle->compiler.has_value()) {
        const int max_threads = handle->max_threads == kXGrammarUnlimited
            ? static_cast<int>(std::thread::hardware_concurrency())
            : static_cast<int>(handle->max_threads);
        handle->compiler.emplace(
            handle->tokenizer_info,
            max_threads,
            handle->is_cache_enabled,
            handle->memory_limit
        );
    }
    return *handle->compiler;
}

std::vector<xgrammar::Grammar> grammar_vector(
    const xgrammar_grammar_t* grammars,
    size_t grammar_count
) {
    if (!grammars || grammar_count == 0) {
        throw std::invalid_argument("Expected at least one grammar.");
    }
    std::vector<xgrammar::Grammar> result;
    result.reserve(grammar_count);
    for (size_t index = 0; index < grammar_count; ++index) {
        if (!grammars[index]) {
            throw std::invalid_argument("Expected a non-null grammar.");
        }
        const auto handle = static_cast<XGrammarGrammarHandle*>(grammars[index]);
        result.push_back(handle->grammar);
    }
    return result;
}

}  // namespace

extern "C" {

const char* xgrammar_last_error_message(void) {
    return last_error.c_str();
}

xgrammar_compiler_t xgrammar_compiler_init(
    const char* const* encoded_vocab,
    size_t encoded_vocab_count,
    xgrammar_vocab_type_t type,
    int32_t vocab_size,
    const int32_t* stop_token_ids,
    size_t stop_token_id_count,
    int add_prefix_space
) {
    return with_error_handling([&] {
        if (!encoded_vocab) {
            throw std::invalid_argument("Expected a vocabulary.");
        }
        if (vocab_size < kXGrammarUnlimited) {
            throw std::invalid_argument("Vocabulary size must be non-negative or unlimited.");
        }
        if (stop_token_id_count > 0 && !stop_token_ids) {
            throw std::invalid_argument("Expected stop token IDs.");
        }
        const std::vector<std::string> vocabulary(encoded_vocab, encoded_vocab + encoded_vocab_count);
        const std::optional<int> optional_vocab_size = vocab_size == kXGrammarUnlimited
            ? std::nullopt
            : std::optional<int>(vocab_size);
        const std::optional<std::vector<int32_t>> optional_stop_token_ids = stop_token_id_count == 0
            ? std::nullopt
            : std::optional<std::vector<int32_t>>(
                std::vector<int32_t>(stop_token_ids, stop_token_ids + stop_token_id_count)
            );
        const xgrammar::TokenizerInfo tokenizer_info(
            vocabulary,
            vocab_type(type),
            optional_vocab_size,
            optional_stop_token_ids,
            add_prefix_space != 0
        );
        return new XGrammarCompilerHandle{tokenizer_info};
    });
}

void xgrammar_compiler_set_memory_limit(xgrammar_compiler_t compiler, int64_t limit) {
    if (!compiler) return;
    auto handle = static_cast<XGrammarCompilerHandle*>(compiler);
    handle->compiler = std::nullopt;
    handle->memory_limit = normalized_memory_limit(limit);
}

void xgrammar_compiler_set_max_threads(xgrammar_compiler_t compiler, int64_t threads) {
    if (!compiler) return;
    auto handle = static_cast<XGrammarCompilerHandle*>(compiler);
    handle->compiler = std::nullopt;
    handle->max_threads = normalized_max_threads(threads);
}

void xgrammar_compiler_set_cache_enabled(xgrammar_compiler_t compiler, int is_enabled) {
    if (!compiler) return;
    auto handle = static_cast<XGrammarCompilerHandle*>(compiler);
    handle->compiler = std::nullopt;
    handle->is_cache_enabled = is_enabled != 0;
}

int64_t xgrammar_compiler_cache_size_bytes(xgrammar_compiler_t compiler) {
    if (!compiler) return 0;
    const auto handle = static_cast<XGrammarCompilerHandle*>(compiler);
    return handle->compiler ? handle->compiler->GetCacheSizeBytes() : 0;
}

int64_t xgrammar_compiler_cache_limit_bytes(xgrammar_compiler_t compiler) {
    if (!compiler) return 0;
    const auto handle = static_cast<XGrammarCompilerHandle*>(compiler);
    return handle->compiler ? handle->compiler->CacheLimitBytes() : 0;
}

void xgrammar_compiler_clear_cache(xgrammar_compiler_t compiler) {
    if (!compiler) return;
    const auto handle = static_cast<XGrammarCompilerHandle*>(compiler);
    if (handle->compiler) handle->compiler->ClearCache();
}

void xgrammar_compiler_destroy(xgrammar_compiler_t compiler) {
    delete static_cast<XGrammarCompilerHandle*>(compiler);
}

xgrammar_grammar_t xgrammar_grammar_init_ebnf(
    const char* ebnf,
    const char* root_rule_name
) {
    return with_error_handling([&] {
        if (!ebnf) throw std::invalid_argument("Expected EBNF.");
        return new XGrammarGrammarHandle{
            xgrammar::Grammar::FromEBNF(ebnf, root_rule_name ? root_rule_name : "root")
        };
    });
}

xgrammar_grammar_t xgrammar_grammar_init_json_schema(
    const char* json_schema,
    int any_whitespace,
    int32_t indent,
    const char* comma_separator,
    const char* colon_separator,
    int strict_mode,
    int32_t max_whitespace_count,
    int any_order
) {
    return with_error_handling([&] {
        if (!json_schema) throw std::invalid_argument("Expected a JSON Schema.");
        const std::optional<int> optional_indent = indent == kXGrammarUnlimited
            ? std::nullopt
            : std::optional<int>(indent);
        const std::optional<std::pair<std::string, std::string>> separators =
            comma_separator && colon_separator
            ? std::optional<std::pair<std::string, std::string>>({comma_separator, colon_separator})
            : std::nullopt;
        const std::optional<int> optional_max_whitespace_count =
            max_whitespace_count == kXGrammarUnlimited
            ? std::nullopt
            : std::optional<int>(max_whitespace_count);
        return new XGrammarGrammarHandle{xgrammar::Grammar::FromJSONSchema(
            json_schema,
            any_whitespace != 0,
            optional_indent,
            separators,
            strict_mode != 0,
            optional_max_whitespace_count,
            false,
            any_order != 0
        )};
    });
}

xgrammar_grammar_t xgrammar_grammar_init_regex(const char* regex) {
    return with_error_handling([&] {
        if (!regex) throw std::invalid_argument("Expected a regular expression.");
        return new XGrammarGrammarHandle{xgrammar::Grammar::FromRegex(regex)};
    });
}

xgrammar_grammar_t xgrammar_grammar_init_structural_tag(const char* structural_tag_json) {
    return with_error_handling([&] {
        if (!structural_tag_json) throw std::invalid_argument("Expected a structural tag.");
        return new XGrammarGrammarHandle{grammar_from_structural_tag_json(structural_tag_json)};
    });
}

size_t xgrammar_grammar_ebnf(
    xgrammar_grammar_t grammar,
    char* buffer,
    size_t buffer_capacity
) {
    return with_error_handling([&] {
        if (!grammar) throw std::invalid_argument("Expected a grammar.");
        const std::string ebnf = static_cast<XGrammarGrammarHandle*>(grammar)->grammar.ToString();
        const size_t required_capacity = ebnf.size() + 1;
        if (buffer && buffer_capacity >= required_capacity) {
            std::memcpy(buffer, ebnf.c_str(), required_capacity);
        }
        return required_capacity;
    });
}

xgrammar_grammar_t xgrammar_grammar_concatenate(
    const xgrammar_grammar_t* grammars,
    size_t grammar_count
) {
    return with_error_handling([&] {
        return new XGrammarGrammarHandle{xgrammar::Grammar::Concat(grammar_vector(grammars, grammar_count))};
    });
}

xgrammar_grammar_t xgrammar_grammar_union(
    const xgrammar_grammar_t* grammars,
    size_t grammar_count
) {
    return with_error_handling([&] {
        return new XGrammarGrammarHandle{xgrammar::Grammar::Union(grammar_vector(grammars, grammar_count))};
    });
}

xgrammar_grammar_t xgrammar_grammar_optional(xgrammar_grammar_t grammar) {
    return with_error_handling([&] {
        if (!grammar) throw std::invalid_argument("Expected a grammar.");
        const auto handle = static_cast<XGrammarGrammarHandle*>(grammar);
        return new XGrammarGrammarHandle{optional_grammar(handle->grammar)};
    });
}

xgrammar_grammar_t xgrammar_grammar_repeat(
    xgrammar_grammar_t grammar,
    int32_t minimum,
    int32_t maximum
) {
    return with_error_handling([&] {
        if (!grammar) throw std::invalid_argument("Expected a grammar.");
        if (minimum < 0 || (maximum != kXGrammarUnlimited && maximum < minimum)) {
            throw std::invalid_argument("Invalid grammar repetition range.");
        }
        const auto handle = static_cast<XGrammarGrammarHandle*>(grammar);
        return new XGrammarGrammarHandle{repeated_grammar(handle->grammar, minimum, maximum)};
    });
}

void xgrammar_grammar_destroy(xgrammar_grammar_t grammar) {
    delete static_cast<XGrammarGrammarHandle*>(grammar);
}


xgrammar_matcher_t xgrammar_compile_matcher(
    xgrammar_compiler_t compiler,
    xgrammar_grammar_t grammar
) {
    return with_error_handling([&] {
        if (!compiler || !grammar) throw std::invalid_argument("Expected a compiler and grammar.");
        auto compiler_handle = static_cast<XGrammarCompilerHandle*>(compiler);
        const auto grammar_handle = static_cast<XGrammarGrammarHandle*>(grammar);
        const auto compiled = compiler_for(compiler_handle).CompileGrammar(grammar_handle->grammar);
        const auto bitmask_word_count = xgrammar::GetBitmaskSize(compiler_handle->tokenizer_info.GetVocabSize());
        return new XGrammarMatcherHandle{xgrammar::GrammarMatcher(compiled), compiled, bitmask_word_count};
    });
}

xgrammar_matcher_t xgrammar_matcher_fork(xgrammar_matcher_t matcher) {
    if (!matcher) return nullptr;
    const auto handle = static_cast<XGrammarMatcherHandle*>(matcher);
    return new XGrammarMatcherHandle{handle->matcher.Fork(), handle->compiled_grammar, handle->bitmask_word_count};
}

size_t xgrammar_matcher_bit_count(xgrammar_matcher_t matcher) {
    if (!matcher) return 0;
    const auto handle = static_cast<const XGrammarMatcherHandle*>(matcher);
    return static_cast<size_t>(handle->bitmask_word_count) * 32;
}

int xgrammar_matcher_bitmask(xgrammar_matcher_t matcher, int* bitmask) {
    if (!matcher || !bitmask) return -1;
    const auto handle = static_cast<XGrammarMatcherHandle*>(matcher);
    int64_t shape[] = {handle->bitmask_word_count};
    int64_t strides[] = {1};
    DLTensor bitmask_tensor{
        bitmask,
        DLDevice{kDLCPU, 0},
        1,
        xgrammar::GetBitmaskDLType(),
        shape,
        strides,
        0
    };
    return handle->matcher.FillNextTokenBitmask(&bitmask_tensor, 0);
}

int xgrammar_matcher_accept_token(xgrammar_matcher_t matcher, int token) {
    return matcher ? static_cast<XGrammarMatcherHandle*>(matcher)->matcher.AcceptToken(token) : -1;
}

int xgrammar_matcher_is_completed(xgrammar_matcher_t matcher) {
    return matcher ? static_cast<XGrammarMatcherHandle*>(matcher)->matcher.IsCompleted() : -1;
}

int xgrammar_matcher_is_terminated(xgrammar_matcher_t matcher) {
    return matcher ? static_cast<XGrammarMatcherHandle*>(matcher)->matcher.IsTerminated() : -1;
}

void xgrammar_matcher_rollback(xgrammar_matcher_t matcher, int num_tokens) {
    if (matcher) static_cast<XGrammarMatcherHandle*>(matcher)->matcher.Rollback(num_tokens);
}

void xgrammar_matcher_reset(xgrammar_matcher_t matcher) {
    if (matcher) static_cast<XGrammarMatcherHandle*>(matcher)->matcher.Reset();
}

int64_t xgrammar_matcher_memory_size_bytes(xgrammar_matcher_t matcher) {
    if (!matcher) return 0;
    const auto handle = static_cast<const XGrammarMatcherHandle*>(matcher);
    return static_cast<int64_t>(handle->compiled_grammar.MemorySizeBytes());
}

void xgrammar_matcher_destroy(xgrammar_matcher_t matcher) {
    delete static_cast<XGrammarMatcherHandle*>(matcher);
}

}
