extern "C" {
#include "include/bridging.h"
}

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "sentencepiece/src/sentencepiece_processor.h"

#include "sentencepiece/third_party/absl/strings/string_view.h"

#include <sys/stat.h>

struct SPTokenizerHandle {
  sentencepiece::SentencePieceProcessor processor;
};

extern "C" {

static thread_local std::string last_error_message;

static void set_error_message(const sentencepiece::util::Status &status) {
  last_error_message = status.error_message();
}

const char *sp_tokenizer_last_error_message() {
  return last_error_message.c_str();
}

sp_tokenizer_t sp_tokenizer_init_from_file(const char *model_file) {
  if (!model_file) {
    last_error_message = "model file path should not be null";
    return nullptr;
  }

  struct stat st;
  if (stat(model_file, &st) != 0 || !S_ISREG(st.st_mode)) {
    last_error_message = std::string(model_file) + ": file not found";
    return nullptr;
  }

  const auto handle = new SPTokenizerHandle{};
  const auto result = handle->processor.Load(model_file);
  if (result.ok()) return handle;
  set_error_message(result);
  delete handle;
  return nullptr;
}

sp_tokenizer_t sp_tokenizer_init_from_data(const char *data, size_t size) {
  if (!data || size == 0) {
    last_error_message = "model data should not be empty";
    return nullptr;
  }

  const auto handle = new SPTokenizerHandle{};
  const auto result = handle->processor.LoadFromSerializedProto(absl::string_view(data, size));
  if (result.ok()) return handle;
  set_error_message(result);
  delete handle;
  return nullptr;
}

int *sp_tokenizer_encode(sp_tokenizer_t tokenizer, const char *text, size_t *size) {
  const auto *handle = static_cast<SPTokenizerHandle *>(tokenizer);
  if (!handle || !text || !size) return nullptr;

  std::vector<int> ids;
  const auto status = handle->processor.Encode(text, &ids);
  if (!status.ok()) {
    set_error_message(status);
    return nullptr;
  }

  *size = ids.size();
  if (ids.empty()) return nullptr;

  auto *buffer = static_cast<int *>(std::malloc(sizeof(int) * ids.size()));
  if (!buffer) return nullptr;
  std::memcpy(buffer, ids.data(), sizeof(int) * ids.size());
  return buffer;
}

const char *sp_tokenizer_decode(
  sp_tokenizer_t tokenizer,
  const int *token_ids,
  size_t token_ids_size,
  size_t *size
) {
  const auto *handle = static_cast<SPTokenizerHandle *>(tokenizer);
  if (!handle || !token_ids) return nullptr;

  std::vector<int> ids(token_ids, token_ids + token_ids_size);
  std::string decoded;
  const auto status = handle->processor.Decode(ids, &decoded);
  if (!status.ok()) {
    set_error_message(status);
    if (size) *size = 0;
    return nullptr;
  }

  if (size) *size = decoded.size();
  if (decoded.empty()) return nullptr;

  auto *buffer = static_cast<char *>(std::malloc(decoded.size() + 1));
  if (!buffer) return nullptr;
  std::memcpy(buffer, decoded.c_str(), decoded.size() + 1);
  return buffer;
}

int sp_tokenizer_unk_token_id(sp_tokenizer_t tokenizer) {
  const auto *handle = static_cast<SPTokenizerHandle *>(tokenizer);
  if (!handle) return -1;
  return handle->processor.unk_id();
}

int sp_tokenizer_bos_token_id(sp_tokenizer_t tokenizer) {
  const auto *handle = static_cast<SPTokenizerHandle *>(tokenizer);
  if (!handle) return -1;
  return handle->processor.bos_id();
}

int sp_tokenizer_eos_token_id(sp_tokenizer_t tokenizer) {
  const auto *handle = static_cast<SPTokenizerHandle *>(tokenizer);
  return !handle ? -1 : handle->processor.eos_id();
}

int sp_tokenizer_pad_token_id(sp_tokenizer_t tokenizer) {
  const auto *handle = static_cast<SPTokenizerHandle *>(tokenizer);
  return !handle ? -1 : handle->processor.pad_id();
}

size_t sp_tokenizer_vocab_size(sp_tokenizer_t tokenizer) {
  const auto *handle = static_cast<SPTokenizerHandle *>(tokenizer);
  return !handle ? 0 : handle->processor.GetPieceSize();
}

int sp_tokenizer_token_to_id(sp_tokenizer_t tokenizer, const char *token) {
  const auto *handle = static_cast<SPTokenizerHandle *>(tokenizer);
  if (!handle || !token) return -1;
  return handle->processor.PieceToId(token);
}

int sp_tokenizer_id_to_token(
  sp_tokenizer_t tokenizer,
  int token_id,
  char *out_token,
  size_t out_token_size
) {
  const auto *handle = static_cast<SPTokenizerHandle *>(tokenizer);
  if (!handle || !out_token || out_token_size == 0) return -1;

  const auto piece_size = static_cast<size_t>(handle->processor.GetPieceSize());
  if (token_id < 0 || static_cast<size_t>(token_id) >= piece_size) return -1;

  const auto &piece = handle->processor.IdToPiece(token_id);
  std::snprintf(out_token, out_token_size, "%s", piece.c_str());
  return 0;
}

void sp_tokenizer_destroy(sp_tokenizer_t tokenizer) {
  if (tokenizer) delete static_cast<SPTokenizerHandle *>(tokenizer);
}

}
