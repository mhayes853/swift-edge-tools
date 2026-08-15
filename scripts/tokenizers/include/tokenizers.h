#ifndef EDGE_TOOLS_TOKENIZERS_H
#define EDGE_TOOLS_TOKENIZERS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct edge_tokenizer_t edge_tokenizer_t;

enum {
  EDGE_TOKENIZER_SUCCESS = 0,
  EDGE_TOKENIZER_FAILURE = 1,
  EDGE_TOKENIZER_INVALID_ARGUMENT = 2,
  EDGE_TOKENIZER_BUFFER_TOO_SMALL = 3,
};

const char *edge_tokenizer_last_error_message(void);

int32_t edge_tokenizer_create(
  const uint8_t *tokenizer_json,
  size_t tokenizer_json_count,
  edge_tokenizer_t **tokenizer
);
void edge_tokenizer_destroy(edge_tokenizer_t *tokenizer);

int32_t edge_tokenizer_encode(
  const edge_tokenizer_t *tokenizer,
  const uint8_t *text,
  size_t text_count,
  bool add_special_tokens,
  int32_t *token_ids,
  size_t token_ids_capacity,
  size_t *token_ids_count
);

int32_t edge_tokenizer_decode(
  const edge_tokenizer_t *tokenizer,
  const int32_t *token_ids,
  size_t token_ids_count,
  bool skip_special_tokens,
  uint8_t *text,
  size_t text_capacity,
  size_t *text_count
);

int32_t edge_tokenizer_token_to_id(
  const edge_tokenizer_t *tokenizer,
  const uint8_t *token,
  size_t token_count,
  int32_t *token_id,
  bool *found
);

int32_t edge_tokenizer_id_to_token(
  const edge_tokenizer_t *tokenizer,
  int32_t token_id,
  uint8_t *token,
  size_t token_capacity,
  size_t *token_count,
  bool *found
);

int32_t edge_tokenizer_vocabulary(
  const edge_tokenizer_t *tokenizer,
  uint8_t *tokens,
  size_t tokens_capacity,
  size_t *tokens_count,
  size_t *lengths,
  size_t lengths_capacity,
  size_t *lengths_count
);

#endif
