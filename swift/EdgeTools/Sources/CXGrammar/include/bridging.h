#ifndef XGRAMMAR_BRIDGING_H
#define XGRAMMAR_BRIDGING_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define kXGrammarUnlimited -1

typedef enum {
  xgrammar_vocab_type_raw = 0,
  xgrammar_vocab_type_byte_fallback = 1,
  xgrammar_vocab_type_byte_level = 2,
} xgrammar_vocab_type_t;

typedef void* xgrammar_tokenizer_info_t;
typedef void* xgrammar_compiler_t;
typedef void* xgrammar_compiled_grammar_t;
typedef void* xgrammar_matcher_t;
typedef void* xgrammar_grammar_t;

typedef enum {
  xgrammar_named_grammar_lark = 0,
  xgrammar_named_grammar_handle = 1,
} xgrammar_named_grammar_kind_t;

typedef struct {
  const char* name;
  xgrammar_named_grammar_kind_t kind;
  const char* lark_source;
  xgrammar_grammar_t grammar;
} xgrammar_named_grammar_t;

const char* xgrammar_last_error_message(void);

xgrammar_tokenizer_info_t xgrammar_tokenizer_info_init(
    const char* const* encoded_vocab,
    size_t encoded_vocab_count,
    xgrammar_vocab_type_t vocab_type,
    int32_t vocab_size,
    const int32_t* stop_token_ids,
    size_t stop_token_id_count,
    int add_prefix_space
);
xgrammar_tokenizer_info_t xgrammar_tokenizer_info_from_vocab_and_metadata(
    const char* const* encoded_vocab,
    size_t encoded_vocab_count,
    const char* metadata
);
size_t xgrammar_tokenizer_info_detect_metadata_from_hf(
    const char* backend_json,
    char* buffer,
    size_t buffer_capacity
);
size_t xgrammar_tokenizer_info_serialize_json(
    xgrammar_tokenizer_info_t tokenizer_info,
    char* buffer,
    size_t buffer_capacity
);
xgrammar_tokenizer_info_t xgrammar_tokenizer_info_deserialize_json(const char* json);
xgrammar_tokenizer_info_t xgrammar_tokenizer_info_copy(xgrammar_tokenizer_info_t tokenizer_info);
void xgrammar_tokenizer_info_destroy(xgrammar_tokenizer_info_t tokenizer_info);

xgrammar_compiler_t xgrammar_compiler_init(
    xgrammar_tokenizer_info_t tokenizer_info,
    int32_t max_threads,
    int cache_enabled,
    int64_t max_memory_bytes
);
xgrammar_compiler_t xgrammar_compiler_copy(xgrammar_compiler_t compiler);
int64_t xgrammar_compiler_cache_size_bytes(xgrammar_compiler_t compiler);
int64_t xgrammar_compiler_cache_limit_bytes(xgrammar_compiler_t compiler);
void xgrammar_compiler_clear_cache(xgrammar_compiler_t compiler);
void xgrammar_compiler_destroy(xgrammar_compiler_t compiler);

xgrammar_grammar_t xgrammar_grammar_init_ebnf(const char* ebnf, const char* root_rule_name);
xgrammar_grammar_t xgrammar_grammar_init_json_schema(
    const char* json_schema,
    int any_whitespace,
    int32_t indent,
    const char* comma_separator,
    const char* colon_separator,
    int strict_mode,
    int32_t max_whitespace_count,
    int any_order
);
xgrammar_grammar_t xgrammar_grammar_init_regex(const char* regex);
xgrammar_grammar_t xgrammar_grammar_init_lark(
    const char* lark,
    xgrammar_tokenizer_info_t tokenizer_info,
    const xgrammar_named_grammar_t* named_grammars,
    size_t named_grammar_count
);
xgrammar_grammar_t xgrammar_grammar_init_structural_tag(
    const char* structural_tag_json,
    xgrammar_tokenizer_info_t tokenizer_info
);
xgrammar_grammar_t xgrammar_grammar_builtin_json(void);
xgrammar_grammar_t xgrammar_grammar_copy(xgrammar_grammar_t grammar);
size_t xgrammar_grammar_ebnf(xgrammar_grammar_t grammar, char* buffer, size_t buffer_capacity);
size_t xgrammar_grammar_serialize_json(xgrammar_grammar_t grammar, char* buffer, size_t buffer_capacity);
xgrammar_grammar_t xgrammar_grammar_deserialize_json(const char* json);
xgrammar_grammar_t xgrammar_grammar_concatenate(const xgrammar_grammar_t* grammars, size_t grammar_count);
xgrammar_grammar_t xgrammar_grammar_union(const xgrammar_grammar_t* grammars, size_t grammar_count);
xgrammar_grammar_t xgrammar_grammar_optional(xgrammar_grammar_t grammar);
xgrammar_grammar_t xgrammar_grammar_repeat(xgrammar_grammar_t grammar, int32_t minimum, int32_t maximum);
void xgrammar_grammar_destroy(xgrammar_grammar_t grammar);

xgrammar_compiled_grammar_t xgrammar_compiler_compile_grammar(
    xgrammar_compiler_t compiler, xgrammar_grammar_t grammar
);
xgrammar_grammar_t xgrammar_compiled_grammar_grammar(xgrammar_compiled_grammar_t compiled_grammar);
xgrammar_tokenizer_info_t xgrammar_compiled_grammar_tokenizer_info(
    xgrammar_compiled_grammar_t compiled_grammar
);
int64_t xgrammar_compiled_grammar_memory_size_bytes(xgrammar_compiled_grammar_t compiled_grammar);
size_t xgrammar_compiled_grammar_serialize_json(
    xgrammar_compiled_grammar_t compiled_grammar,
    char* buffer,
    size_t buffer_capacity
);
xgrammar_compiled_grammar_t xgrammar_compiled_grammar_deserialize_json(
    const char* json,
    xgrammar_tokenizer_info_t tokenizer_info
);
xgrammar_compiled_grammar_t xgrammar_compiled_grammar_copy(
    xgrammar_compiled_grammar_t compiled_grammar
);
void xgrammar_compiled_grammar_destroy(xgrammar_compiled_grammar_t compiled_grammar);

xgrammar_matcher_t xgrammar_matcher_init(
    xgrammar_compiled_grammar_t compiled_grammar,
    const int32_t* override_stop_token_ids,
    size_t override_stop_token_id_count,
    int terminate_without_stop_token,
    int32_t max_rollback_tokens
);
xgrammar_matcher_t xgrammar_matcher_fork(xgrammar_matcher_t matcher);
size_t xgrammar_matcher_bit_count(xgrammar_matcher_t matcher);
int xgrammar_matcher_bitmask(xgrammar_matcher_t matcher, int* bitmask);
int xgrammar_matcher_accept_token(xgrammar_matcher_t matcher, int token);
int xgrammar_matcher_accept_string(xgrammar_matcher_t matcher, const char* string);
int xgrammar_matcher_is_completed(xgrammar_matcher_t matcher);
int xgrammar_matcher_is_terminated(xgrammar_matcher_t matcher);
void xgrammar_matcher_rollback(xgrammar_matcher_t matcher, int num_tokens);
void xgrammar_matcher_reset(xgrammar_matcher_t matcher);
void xgrammar_matcher_destroy(xgrammar_matcher_t matcher);

#ifdef __cplusplus
}
#endif

#endif
