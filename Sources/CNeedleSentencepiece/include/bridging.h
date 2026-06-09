#ifndef SENTENCEPIECE_C_ADAPTER_H
#define SENTENCEPIECE_C_ADAPTER_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Create a new SentencePieceProcessor instance
void *spm_new_sentencepiece_processor();

// Free SentencePieceProcessor instance
void spm_free_sentencepiece_processor(void *ptr_inst);

// Load  model from a file
bool spm_load_model(void *ptr_inst, const char *model_file);

// Normalize input text
// Returns NULL on failure
// Caller must free the returned string
char *spm_normalize(void *ptr_inst, const char *text);

// Encode text into a sequence of token IDs
// Returns NULL on failure
// Caller must free the returned array
int *spm_encode(void *ptr_inst, const char *text, int *size);

// Set encode extra options
void spm_set_encode_extra_options(void *ptr_inst, const char *extra_option);

// Decode a sequence of token IDs into text
// Returns NULL on failure
// Caller must free the returned string
char *spm_decode(void *ptr_inst, const int *ids, int size);

// Set decode extra options
void spm_set_decode_extra_options(void *ptr_inst, const char *extra_option);

// Return id of a piece
// Returns NULL on failure
// Caller must free the returned array
char *spm_id_to_piece(void *ptr_inst, int id);

// Return id of a piece
// Returns NULL on failure
int spm_piece_to_id(void *ptr_inst, const char *piece);

// Returns unknown (<unk>) id.
int spm_unk_id(void *ptr_inst);

// Returns BOS (<s>) id.
int spm_bos_id(void *ptr_inst);

// Returns EOS (</s>) id.
int spm_eos_id(void *ptr_inst);

// Returns PAD (<pad>) id.
int spm_pad_id(void *ptr_inst);

#ifdef __cplusplus
} /* end of the 'extern "C"' block */
#endif

#endif // SENTENCEPIECE_C_ADAPTER_H
