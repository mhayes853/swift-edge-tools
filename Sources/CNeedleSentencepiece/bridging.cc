#include "bridging.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "sentencepiece_processor.h"

namespace {

// Allocates a heap buffer of `count` ints initialized from the source vector.
// The caller is responsible for `free()`-ing the returned pointer (matching
// the existing C ABI contract: "caller must free the returned array").
int *ToHeapIntArray(const std::vector<int> &values) {
  const size_t byteCount = values.size() * sizeof(int);
  int *buffer = static_cast<int *>(std::malloc(byteCount));
  if (buffer == nullptr) {
    return nullptr;
  }
  std::memcpy(buffer, values.data(), byteCount);
  return buffer;
}

// Allocates a heap copy of a std::string. The caller is responsible for
// `free()`-ing the returned pointer (matching the existing C ABI contract).
char *ToHeapCString(const std::string &value) {
  char *buffer = static_cast<char *>(std::malloc(value.size() + 1));
  if (buffer == nullptr) {
    return nullptr;
  }
  std::memcpy(buffer, value.data(), value.size());
  buffer[value.size()] = '\0';
  return buffer;
}

} // namespace

extern "C" {

// NB: The processor is allocated with `malloc` (not `new`) and constructed
// in place via placement new. This keeps the ABI compatible with the existing
// Swift call site, which calls `UnsafeMutableRawPointer.deallocate()` (i.e.
// `free()`) in its deinit rather than `spm_free_sentencepiece_processor`.
void *spm_new_sentencepiece_processor() {
  void *storage = std::malloc(sizeof(sentencepiece::SentencePieceProcessor));
  if (storage == nullptr) {
    return nullptr;
  }
  return new (storage) sentencepiece::SentencePieceProcessor();
}

void spm_free_sentencepiece_processor(void *ptr_inst) {
  if (ptr_inst == nullptr) {
    return;
  }
  auto *processor = static_cast<sentencepiece::SentencePieceProcessor *>(ptr_inst);
  processor->~SentencePieceProcessor();
  std::free(ptr_inst);
}

bool spm_load_model(void *ptr_inst, const char *model_file) {
  if (ptr_inst == nullptr || model_file == nullptr) {
    return false;
  }
  auto *processor = static_cast<sentencepiece::SentencePieceProcessor *>(ptr_inst);
  return processor->Load(model_file).ok();
}

char *spm_normalize(void *ptr_inst, const char *text) {
  if (ptr_inst == nullptr || text == nullptr) {
    return nullptr;
  }
  auto *processor = static_cast<sentencepiece::SentencePieceProcessor *>(ptr_inst);
  std::string normalized;
  if (!processor->Normalize(text, &normalized).ok()) {
    return nullptr;
  }
  return ToHeapCString(normalized);
}

int *spm_encode(void *ptr_inst, const char *text, int *size) {
  if (ptr_inst == nullptr || text == nullptr || size == nullptr) {
    return nullptr;
  }
  auto *processor = static_cast<sentencepiece::SentencePieceProcessor *>(ptr_inst);
  std::vector<int> ids;
  if (!processor->Encode(text, &ids).ok()) {
    return nullptr;
  }
  *size = static_cast<int>(ids.size());
  return ToHeapIntArray(ids);
}

void spm_set_encode_extra_options(void *ptr_inst, const char *extra_option) {
  if (ptr_inst == nullptr || extra_option == nullptr) {
    return;
  }
  auto *processor = static_cast<sentencepiece::SentencePieceProcessor *>(ptr_inst);
  processor->SetEncodeExtraOptions(extra_option).IgnoreError();
}

char *spm_decode(void *ptr_inst, const int *ids, int size) {
  if (ptr_inst == nullptr || ids == nullptr || size < 0) {
    return nullptr;
  }
  auto *processor = static_cast<sentencepiece::SentencePieceProcessor *>(ptr_inst);
  std::vector<int> idVector(ids, ids + size);
  std::string detokenized;
  if (!processor->Decode(idVector, &detokenized).ok()) {
    return nullptr;
  }
  return ToHeapCString(detokenized);
}

void spm_set_decode_extra_options(void *ptr_inst, const char *extra_option) {
  if (ptr_inst == nullptr || extra_option == nullptr) {
    return;
  }
  auto *processor = static_cast<sentencepiece::SentencePieceProcessor *>(ptr_inst);
  processor->SetDecodeExtraOptions(extra_option).IgnoreError();
}

char *spm_id_to_piece(void *ptr_inst, int id) {
  if (ptr_inst == nullptr) {
    return nullptr;
  }
  auto *processor = static_cast<sentencepiece::SentencePieceProcessor *>(ptr_inst);
  if (!processor->status().ok()) {
    return nullptr;
  }
  return ToHeapCString(processor->IdToPiece(id));
}

int spm_piece_to_id(void *ptr_inst, const char *piece) {
  if (ptr_inst == nullptr || piece == nullptr) {
    return -1;
  }
  auto *processor = static_cast<sentencepiece::SentencePieceProcessor *>(ptr_inst);
  if (!processor->status().ok()) {
    return -1;
  }
  return processor->PieceToId(piece);
}

int spm_unk_id(void *ptr_inst) {
  if (ptr_inst == nullptr) {
    return -1;
  }
  auto *processor = static_cast<sentencepiece::SentencePieceProcessor *>(ptr_inst);
  return processor->unk_id();
}

int spm_bos_id(void *ptr_inst) {
  if (ptr_inst == nullptr) {
    return -1;
  }
  auto *processor = static_cast<sentencepiece::SentencePieceProcessor *>(ptr_inst);
  return processor->bos_id();
}

int spm_eos_id(void *ptr_inst) {
  if (ptr_inst == nullptr) {
    return -1;
  }
  auto *processor = static_cast<sentencepiece::SentencePieceProcessor *>(ptr_inst);
  return processor->eos_id();
}

int spm_pad_id(void *ptr_inst) {
  if (ptr_inst == nullptr) {
    return -1;
  }
  auto *processor = static_cast<sentencepiece::SentencePieceProcessor *>(ptr_inst);
  return processor->pad_id();
}

} // extern "C"
