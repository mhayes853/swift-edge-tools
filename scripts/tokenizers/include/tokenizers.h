#ifndef HF_TOKENIZERS_H
#define HF_TOKENIZERS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct hf_tokenizer_t hf_tokenizer_t;

enum {
  HF_TOKENIZER_SUCCESS = 0,
  HF_TOKENIZER_FAILURE = 1,
  HF_TOKENIZER_INVALID_ARGUMENT = 2,
  HF_TOKENIZER_BUFFER_TOO_SMALL = 3,
};

const char *hf_tokenizer_last_error_message(void);

int32_t hf_tokenizer_create(
  const uint8_t *tokenizer_json,
  size_t tokenizer_json_count,
  hf_tokenizer_t **tokenizer
);
void hf_tokenizer_destroy(hf_tokenizer_t *tokenizer);

int32_t hf_tokenizer_encode(
  const hf_tokenizer_t *tokenizer,
  const uint8_t *text,
  size_t text_count,
  bool add_special_tokens,
  int32_t *token_ids,
  size_t token_ids_capacity,
  size_t *token_ids_count
);

int32_t hf_tokenizer_decode(
  const hf_tokenizer_t *tokenizer,
  const int32_t *token_ids,
  size_t token_ids_count,
  bool skip_special_tokens,
  uint8_t *text,
  size_t text_capacity,
  size_t *text_count
);

int32_t hf_tokenizer_token_to_id(
  const hf_tokenizer_t *tokenizer,
  const uint8_t *token,
  size_t token_count,
  int32_t *token_id,
  bool *found
);

int32_t hf_tokenizer_id_to_token(
  const hf_tokenizer_t *tokenizer,
  int32_t token_id,
  uint8_t *token,
  size_t token_capacity,
  size_t *token_count,
  bool *found
);

int32_t hf_tokenizer_vocabulary(
  const hf_tokenizer_t *tokenizer,
  uint8_t *tokens,
  size_t tokens_capacity,
  size_t *tokens_count,
  size_t *lengths,
  size_t lengths_capacity,
  size_t *lengths_count
);

#endif
