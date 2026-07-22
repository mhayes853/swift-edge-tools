#include "include/bridging.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
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

struct XGrammarTokenizerInfoHandle {
    xgrammar::TokenizerInfo tokenizer_info;
};

struct XGrammarCompilerHandle {
    xgrammar::TokenizerInfo tokenizer_info;
    xgrammar::GrammarCompiler compiler;
};

struct XGrammarCompiledGrammarHandle {
    xgrammar::CompiledGrammar compiled_grammar;
};

struct XGrammarMatcherHandle {
    xgrammar::GrammarMatcher matcher;
    xgrammar::CompiledGrammar compiled_grammar;
    int32_t bitmask_word_count;
};

struct XGrammarGrammarHandle {
    xgrammar::Grammar grammar;
};

template <typename Value, typename Error>
Value value_or_throw(std::variant<Value, Error> result) {
    if (const auto* value = std::get_if<Value>(&result)) {
        return *value;
    }
    const auto& error = std::get<Error>(result);
    std::visit([](const auto& value) { throw std::runtime_error(value.what()); }, error);
    std::abort();
}

size_t write_string(const std::string& string, char* buffer, size_t buffer_capacity) {
    const size_t required_capacity = string.size() + 1;
    if (buffer && buffer_capacity >= required_capacity) {
        std::memcpy(buffer, string.c_str(), required_capacity);
    }
    return required_capacity;
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

xgrammar_tokenizer_info_t xgrammar_tokenizer_info_init(
    const char* const* encoded_vocab,
    size_t encoded_vocab_count,
    xgrammar_vocab_type_t type,
    int32_t vocab_size,
    const int32_t* stop_token_ids,
    size_t stop_token_id_count,
    int add_prefix_space
) {
    return with_error_handling([&] {
        if (!encoded_vocab || vocab_size < kXGrammarUnlimited || (stop_token_id_count > 0 && !stop_token_ids)) {
            throw std::invalid_argument("Invalid tokenizer information.");
        }
        const std::vector<std::string> vocabulary(encoded_vocab, encoded_vocab + encoded_vocab_count);
        const auto optional_vocab_size = vocab_size == kXGrammarUnlimited
            ? std::optional<int>{} : std::optional<int>{vocab_size};
        const auto optional_stop_token_ids = stop_token_id_count == 0
            ? std::optional<std::vector<int32_t>>{}
            : std::optional<std::vector<int32_t>>{std::vector<int32_t>(stop_token_ids, stop_token_ids + stop_token_id_count)};
        return new XGrammarTokenizerInfoHandle{xgrammar::TokenizerInfo(
            vocabulary, vocab_type(type), optional_vocab_size, optional_stop_token_ids, add_prefix_space != 0
        )};
    });
}

xgrammar_tokenizer_info_t xgrammar_tokenizer_info_from_vocab_and_metadata(
    const char* const* encoded_vocab,
    size_t encoded_vocab_count,
    const char* metadata
) {
    return with_error_handling([&] {
        if (!encoded_vocab || !metadata) throw std::invalid_argument("Expected vocabulary and metadata.");
        return new XGrammarTokenizerInfoHandle{xgrammar::TokenizerInfo::FromVocabAndMetadata(
            std::vector<std::string>(encoded_vocab, encoded_vocab + encoded_vocab_count), metadata
        )};
    });
}

size_t xgrammar_tokenizer_info_detect_metadata_from_hf(
    const char* backend_json,
    char* buffer,
    size_t buffer_capacity
) {
    return with_error_handling([&] {
        if (!backend_json) throw std::invalid_argument("Expected Hugging Face backend JSON.");
        return write_string(xgrammar::TokenizerInfo::DetectMetadataFromHF(backend_json), buffer, buffer_capacity);
    });
}

size_t xgrammar_tokenizer_info_serialize_json(
    xgrammar_tokenizer_info_t tokenizer_info,
    char* buffer,
    size_t buffer_capacity
) {
    return with_error_handling([&] {
        if (!tokenizer_info) throw std::invalid_argument("Expected tokenizer information.");
        return write_string(static_cast<XGrammarTokenizerInfoHandle*>(tokenizer_info)->tokenizer_info.SerializeJSON(), buffer, buffer_capacity);
    });
}

xgrammar_tokenizer_info_t xgrammar_tokenizer_info_deserialize_json(const char* json) {
    return with_error_handling([&] {
        if (!json) throw std::invalid_argument("Expected tokenizer information JSON.");
        return new XGrammarTokenizerInfoHandle{value_or_throw(xgrammar::TokenizerInfo::DeserializeJSON(json))};
    });
}

void xgrammar_tokenizer_info_destroy(xgrammar_tokenizer_info_t tokenizer_info) {
    delete static_cast<XGrammarTokenizerInfoHandle*>(tokenizer_info);
}

xgrammar_compiler_t xgrammar_compiler_init(
    xgrammar_tokenizer_info_t tokenizer_info,
    int32_t max_threads,
    int cache_enabled,
    int64_t max_memory_bytes
) {
    return with_error_handling([&] {
        if (!tokenizer_info || max_threads <= 0) throw std::invalid_argument("Invalid compiler configuration.");
        const auto info = static_cast<XGrammarTokenizerInfoHandle*>(tokenizer_info)->tokenizer_info;
        return new XGrammarCompilerHandle{info, xgrammar::GrammarCompiler(info, max_threads, cache_enabled != 0, max_memory_bytes)};
    });
}

int64_t xgrammar_compiler_cache_size_bytes(xgrammar_compiler_t compiler) {
    return compiler ? static_cast<XGrammarCompilerHandle*>(compiler)->compiler.GetCacheSizeBytes() : 0;
}

int64_t xgrammar_compiler_cache_limit_bytes(xgrammar_compiler_t compiler) {
    return compiler ? static_cast<XGrammarCompilerHandle*>(compiler)->compiler.CacheLimitBytes() : 0;
}

void xgrammar_compiler_clear_cache(xgrammar_compiler_t compiler) {
    if (compiler) static_cast<XGrammarCompilerHandle*>(compiler)->compiler.ClearCache();
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

xgrammar_grammar_t xgrammar_grammar_init_lark(const char* lark) {
    return with_error_handling([&] {
        if (!lark) throw std::invalid_argument("Expected a Lark grammar.");
        return new XGrammarGrammarHandle{xgrammar::Grammar::FromLark(lark)};
    });
}

xgrammar_grammar_t xgrammar_grammar_init_structural_tag(const char* structural_tag_json) {
    return with_error_handling([&] {
        if (!structural_tag_json) throw std::invalid_argument("Expected a structural tag.");
        return new XGrammarGrammarHandle{grammar_from_structural_tag_json(structural_tag_json)};
    });
}

xgrammar_grammar_t xgrammar_grammar_builtin_json(void) {
    return with_error_handling([&] { return new XGrammarGrammarHandle{xgrammar::Grammar::BuiltinJSONGrammar()}; });
}

size_t xgrammar_grammar_ebnf(xgrammar_grammar_t grammar, char* buffer, size_t buffer_capacity) {
    return with_error_handling([&] {
        if (!grammar) throw std::invalid_argument("Expected a grammar.");
        return write_string(static_cast<XGrammarGrammarHandle*>(grammar)->grammar.ToString(), buffer, buffer_capacity);
    });
}

size_t xgrammar_grammar_serialize_json(xgrammar_grammar_t grammar, char* buffer, size_t buffer_capacity) {
    return with_error_handling([&] {
        if (!grammar) throw std::invalid_argument("Expected a grammar.");
        return write_string(static_cast<XGrammarGrammarHandle*>(grammar)->grammar.SerializeJSON(), buffer, buffer_capacity);
    });
}

xgrammar_grammar_t xgrammar_grammar_deserialize_json(const char* json) {
    return with_error_handling([&] {
        if (!json) throw std::invalid_argument("Expected grammar JSON.");
        return new XGrammarGrammarHandle{value_or_throw(xgrammar::Grammar::DeserializeJSON(json))};
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


xgrammar_compiled_grammar_t xgrammar_compiler_compile_grammar(
    xgrammar_compiler_t compiler, xgrammar_grammar_t grammar
) {
    return with_error_handling([&] {
        if (!compiler || !grammar) throw std::invalid_argument("Expected a compiler and grammar.");
        const auto compiler_handle = static_cast<XGrammarCompilerHandle*>(compiler);
        const auto grammar_handle = static_cast<XGrammarGrammarHandle*>(grammar);
        return new XGrammarCompiledGrammarHandle{compiler_handle->compiler.CompileGrammar(grammar_handle->grammar)};
    });
}

xgrammar_grammar_t xgrammar_compiled_grammar_grammar(xgrammar_compiled_grammar_t compiled_grammar) {
    return with_error_handling([&] {
        if (!compiled_grammar) throw std::invalid_argument("Expected a compiled grammar.");
        return new XGrammarGrammarHandle{static_cast<XGrammarCompiledGrammarHandle*>(compiled_grammar)->compiled_grammar.GetGrammar()};
    });
}

xgrammar_tokenizer_info_t xgrammar_compiled_grammar_tokenizer_info(xgrammar_compiled_grammar_t compiled_grammar) {
    return with_error_handling([&] {
        if (!compiled_grammar) throw std::invalid_argument("Expected a compiled grammar.");
        return new XGrammarTokenizerInfoHandle{static_cast<XGrammarCompiledGrammarHandle*>(compiled_grammar)->compiled_grammar.GetTokenizerInfo()};
    });
}

int64_t xgrammar_compiled_grammar_memory_size_bytes(xgrammar_compiled_grammar_t compiled_grammar) {
    return compiled_grammar
        ? static_cast<int64_t>(static_cast<XGrammarCompiledGrammarHandle*>(compiled_grammar)->compiled_grammar.MemorySizeBytes())
        : 0;
}

size_t xgrammar_compiled_grammar_serialize_json(
    xgrammar_compiled_grammar_t compiled_grammar,
    char* buffer,
    size_t buffer_capacity
) {
    return with_error_handling([&] {
        if (!compiled_grammar) throw std::invalid_argument("Expected a compiled grammar.");
        return write_string(static_cast<XGrammarCompiledGrammarHandle*>(compiled_grammar)->compiled_grammar.SerializeJSON(), buffer, buffer_capacity);
    });
}

xgrammar_compiled_grammar_t xgrammar_compiled_grammar_deserialize_json(
    const char* json,
    xgrammar_tokenizer_info_t tokenizer_info
) {
    return with_error_handling([&] {
        if (!json || !tokenizer_info) throw std::invalid_argument("Expected compiled grammar JSON and tokenizer information.");
        const auto info = static_cast<XGrammarTokenizerInfoHandle*>(tokenizer_info)->tokenizer_info;
        return new XGrammarCompiledGrammarHandle{value_or_throw(xgrammar::CompiledGrammar::DeserializeJSON(json, info))};
    });
}

void xgrammar_compiled_grammar_destroy(xgrammar_compiled_grammar_t compiled_grammar) {
    delete static_cast<XGrammarCompiledGrammarHandle*>(compiled_grammar);
}

xgrammar_matcher_t xgrammar_matcher_init(
    xgrammar_compiled_grammar_t compiled_grammar,
    const int32_t* override_stop_token_ids,
    size_t override_stop_token_id_count,
    int terminate_without_stop_token,
    int32_t max_rollback_tokens
) {
    return with_error_handling([&] {
        if (!compiled_grammar || (override_stop_token_id_count > 0 && !override_stop_token_ids)) {
            throw std::invalid_argument("Invalid matcher configuration.");
        }
        const auto compiled = static_cast<XGrammarCompiledGrammarHandle*>(compiled_grammar)->compiled_grammar;
        const auto stop_tokens = override_stop_token_id_count == 0
            ? std::optional<std::vector<int>>{}
            : std::optional<std::vector<int>>{std::vector<int>(override_stop_token_ids, override_stop_token_ids + override_stop_token_id_count)};
        const auto bitmask_word_count = xgrammar::GetBitmaskSize(compiled.GetTokenizerInfo().GetVocabSize());
        return new XGrammarMatcherHandle{
            xgrammar::GrammarMatcher(compiled, stop_tokens, terminate_without_stop_token != 0, max_rollback_tokens),
            compiled,
            bitmask_word_count
        };
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

int xgrammar_matcher_accept_string(xgrammar_matcher_t matcher, const char* string) {
    return matcher && string ? static_cast<XGrammarMatcherHandle*>(matcher)->matcher.AcceptString(string) : -1;
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

void xgrammar_matcher_destroy(xgrammar_matcher_t matcher) {
    delete static_cast<XGrammarMatcherHandle*>(matcher);
}

}
