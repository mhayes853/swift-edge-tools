#pragma once

#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* needle_sp_t;

needle_sp_t needle_sp_init_from_file(const char* model_file);

const char* needle_last_error_message();

int* needle_sp_encode(needle_sp_t tokenizer, const char* text, size_t* size);
const char* needle_sp_decode(
    needle_sp_t tokenizer,
    const int* token_ids,
    size_t token_ids_size,
    size_t* size
);

int needle_sp_unk_token_id(needle_sp_t tokenizer);
int needle_sp_bos_token_id(needle_sp_t tokenizer);
int needle_sp_eos_token_id(needle_sp_t tokenizer);
int needle_sp_pad_token_id(needle_sp_t tokenizer);

int needle_sp_tokens_to_ids(
    needle_sp_t tokenizer,
    const char** tokens,
    int* token_ids,
    size_t size
);
int needle_sp_ids_to_tokens(
    needle_sp_t tokenizer,
    const int* token_ids,
    char** tokens,
    size_t size
);

void needle_sp_destroy(needle_sp_t tokenizer);

#ifdef __cplusplus
}
#endif
