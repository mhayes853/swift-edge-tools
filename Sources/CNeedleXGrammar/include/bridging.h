#ifndef __NEEDLE_XGRAMMAR_BRIDGING_H__
#define __NEEDLE_XGRAMMAR_BRIDGING_H__

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define kNeedleXGrammarCompilerHardwareConcurrency -1
#define kNeedleXGrammarCompilerNoMemoryLimit -1

typedef void* needle_xgrammar_compiler_t;
typedef void* needle_xgrammar_matcher_t;
typedef void* needle_xgrammar_grammar_t;

needle_xgrammar_compiler_t needle_xgrammar_compiler_init(const char** encoded_vocab, size_t vocab_size);
void needle_xgrammar_compiler_set_memory_limit(needle_xgrammar_compiler_t compiler, int64_t limit);
void needle_xgrammar_compiler_set_max_threads(needle_xgrammar_compiler_t compiler, int64_t threads);
void needle_xgrammar_compiler_set_cache_enabled(needle_xgrammar_compiler_t compiler, int is_enabled);
void needle_xgrammar_compiler_destroy(needle_xgrammar_compiler_t compiler);

needle_xgrammar_matcher_t needle_xgrammar_compile_matcher(
    needle_xgrammar_compiler_t compiler,
    needle_xgrammar_grammar_t grammar
);
needle_xgrammar_matcher_t needle_xgrammar_matcher_fork(needle_xgrammar_matcher_t matcher);
int needle_xgrammar_matcher_next_bitmask(needle_xgrammar_matcher_t matcher, int* bitmask);
int needle_xgrammar_matcher_accept_token(needle_xgrammar_matcher_t matcher, int token);
int needle_xgrammar_matcher_is_completed(needle_xgrammar_matcher_t matcher);
int needle_xgrammar_matcher_is_terminated(needle_xgrammar_matcher_t matcher);
void needle_xgrammar_matcher_rollback(needle_xgrammar_matcher_t matcher, int num_tokens);
void needle_xgrammar_matcher_reset(needle_xgrammar_matcher_t matcher);
void needle_xgrammar_matcher_destroy(needle_xgrammar_matcher_t matcher);

needle_xgrammar_grammar_t needle_xgrammar_grammar_init(const char* tools_json);
void needle_xgrammar_grammar_destroy(needle_xgrammar_grammar_t grammar);

#ifdef __cplusplus
}
#endif

#endif
