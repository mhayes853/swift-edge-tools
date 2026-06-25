#ifndef __NEEDLE_SENTENCEPIECE_BRIDGING_H__
#define __NEEDLE_SENTENCEPIECE_BRIDGING_H__

#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* needle_sp_tokenizer_t;

needle_sp_tokenizer_t needle_sp_tokenizer_init_from_file(const char* model_file);

const char* needle_sp_last_error_message();

int* needle_sp_tokenizer_encode(needle_sp_tokenizer_t tokenizer, const char* text, size_t* size);
const char* needle_sp_tokenizer_decode(
    needle_sp_tokenizer_t tokenizer,
    const int* token_ids,
    size_t token_ids_size,
    size_t* size
);

int needle_sp_tokenizer_unk_token_id(needle_sp_tokenizer_t tokenizer);
int needle_sp_tokenizer_bos_token_id(needle_sp_tokenizer_t tokenizer);
int needle_sp_tokenizer_eos_token_id(needle_sp_tokenizer_t tokenizer);
int needle_sp_tokenizer_pad_token_id(needle_sp_tokenizer_t tokenizer);

size_t needle_sp_tokenizer_vocab_size(needle_sp_tokenizer_t tokenizer);

int needle_sp_tokenizer_token_to_id(needle_sp_tokenizer_t tokenizer, const char* token);
int needle_sp_tokenizer_id_to_token(
    needle_sp_tokenizer_t tokenizer,
    int token_id,
    char* out_token,
    size_t out_token_size
);

void needle_sp_tokenizer_destroy(needle_sp_tokenizer_t tokenizer);

#ifdef __cplusplus
}
#endif
#endif
