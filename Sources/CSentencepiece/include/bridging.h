#ifndef SENTENCEPIECE_BRIDGING_H
#define SENTENCEPIECE_BRIDGING_H

#include <stdbool.h>
#include <stdlib.h>

typedef void *sp_tokenizer_t;

sp_tokenizer_t sp_tokenizer_init_from_file(const char *model_file);
sp_tokenizer_t sp_tokenizer_init_from_data(const char *data, size_t size);

const char *sp_tokenizer_last_error_message(void);

int *sp_tokenizer_encode(sp_tokenizer_t tokenizer, const char *text, size_t *size);
const char *sp_tokenizer_decode(
  sp_tokenizer_t tokenizer,
  const int *token_ids,
  size_t token_ids_size,
  size_t *size
);

int sp_tokenizer_unk_token_id(sp_tokenizer_t tokenizer);
int sp_tokenizer_bos_token_id(sp_tokenizer_t tokenizer);
int sp_tokenizer_eos_token_id(sp_tokenizer_t tokenizer);
int sp_tokenizer_pad_token_id(sp_tokenizer_t tokenizer);

size_t sp_tokenizer_vocab_size(sp_tokenizer_t tokenizer);

int sp_tokenizer_token_to_id(sp_tokenizer_t tokenizer, const char *token);
int sp_tokenizer_id_to_token(
  sp_tokenizer_t tokenizer,
  int token_id,
  char *out_token,
  size_t out_token_size
);

void sp_tokenizer_destroy(sp_tokenizer_t tokenizer);

#endif
