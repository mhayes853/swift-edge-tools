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

typedef void* xgrammar_compiler_t;
typedef void* xgrammar_matcher_t;
typedef void* xgrammar_grammar_t;

const char* xgrammar_last_error_message(void);

xgrammar_compiler_t xgrammar_compiler_init(
    const char* const* encoded_vocab,
    size_t encoded_vocab_count,
    xgrammar_vocab_type_t vocab_type,
    int32_t vocab_size,
    const int32_t* stop_token_ids,
    size_t stop_token_id_count,
    int add_prefix_space
);
void xgrammar_compiler_set_memory_limit(xgrammar_compiler_t compiler, int64_t limit);
void xgrammar_compiler_set_max_threads(xgrammar_compiler_t compiler, int64_t threads);
void xgrammar_compiler_set_cache_enabled(xgrammar_compiler_t compiler, int is_enabled);
int64_t xgrammar_compiler_cache_size_bytes(xgrammar_compiler_t compiler);
int64_t xgrammar_compiler_cache_limit_bytes(xgrammar_compiler_t compiler);
void xgrammar_compiler_clear_cache(xgrammar_compiler_t compiler);
void xgrammar_compiler_destroy(xgrammar_compiler_t compiler);

xgrammar_grammar_t xgrammar_grammar_init_ebnf(
    const char* ebnf,
    const char* root_rule_name
);
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
xgrammar_grammar_t xgrammar_grammar_init_structural_tag(const char* structural_tag_json);
size_t xgrammar_grammar_ebnf(
    xgrammar_grammar_t grammar,
    char* buffer,
    size_t buffer_capacity
);
xgrammar_grammar_t xgrammar_grammar_concatenate(
    const xgrammar_grammar_t* grammars,
    size_t grammar_count
);
xgrammar_grammar_t xgrammar_grammar_union(
    const xgrammar_grammar_t* grammars,
    size_t grammar_count
);
xgrammar_grammar_t xgrammar_grammar_optional(xgrammar_grammar_t grammar);
xgrammar_grammar_t xgrammar_grammar_repeat(
    xgrammar_grammar_t grammar,
    int32_t minimum,
    int32_t maximum
);
void xgrammar_grammar_destroy(xgrammar_grammar_t grammar);

xgrammar_matcher_t xgrammar_compile_matcher(
    xgrammar_compiler_t compiler,
    xgrammar_grammar_t grammar
);
xgrammar_matcher_t xgrammar_matcher_fork(xgrammar_matcher_t matcher);
size_t xgrammar_matcher_bit_count(xgrammar_matcher_t matcher);
int xgrammar_matcher_bitmask(xgrammar_matcher_t matcher, int* bitmask);
int xgrammar_matcher_accept_token(xgrammar_matcher_t matcher, int token);
int xgrammar_matcher_is_completed(xgrammar_matcher_t matcher);
int xgrammar_matcher_is_terminated(xgrammar_matcher_t matcher);
void xgrammar_matcher_rollback(xgrammar_matcher_t matcher, int num_tokens);
void xgrammar_matcher_reset(xgrammar_matcher_t matcher);
int64_t xgrammar_matcher_memory_size_bytes(xgrammar_matcher_t matcher);
void xgrammar_matcher_destroy(xgrammar_matcher_t matcher);

#ifdef __cplusplus
}
#endif

#endif
