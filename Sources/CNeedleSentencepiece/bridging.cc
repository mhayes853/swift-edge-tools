#include "bridging.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "sentencepiece_processor.h"

#include <sys/stat.h>

struct NeedleSPHandle {
    sentencepiece::SentencePieceProcessor processor;
};

extern "C" {

static thread_local std::string last_error_message;

static void set_error_message(const sentencepiece::util::Status& status) {
    last_error_message = status.error_message();
}

const char* needle_sp_last_error_message() {
    return last_error_message.c_str();
}

needle_sp_tokenizer_t needle_sp_tokenizer_init_from_file(const char* model_file) {
    if (!model_file) {
        last_error_message = "model file path should not be null";
        return nullptr;
    }

    struct stat st;
    if (stat(model_file, &st) != 0 || !S_ISREG(st.st_mode)) {
        last_error_message = std::string(model_file) + ": file not found";
        return nullptr;
    }

    const auto handle = new NeedleSPHandle{};
    const auto result = handle->processor.Load(model_file);
    if (result.ok()) return handle;
    set_error_message(result);
    delete handle;
    return nullptr;
}

int* needle_sp_tokenizer_encode(needle_sp_tokenizer_t tokenizer, const char* text, size_t* size) {
    const auto* handle = static_cast<NeedleSPHandle*>(tokenizer);
    if (!handle || !text || !size) return nullptr;

    std::vector<int> ids;
    const auto status = handle->processor.Encode(text, &ids);
    if (!status.ok()) {
        set_error_message(status);
        return nullptr;
    }

    *size = ids.size();
    if (ids.empty()) return nullptr;

    auto* buffer = static_cast<int*>(std::malloc(sizeof(int) * ids.size()));
    if (!buffer) return nullptr;
    std::memcpy(buffer, ids.data(), sizeof(int) * ids.size());
    return buffer;
}

const char* needle_sp_tokenizer_decode(
    needle_sp_tokenizer_t tokenizer,
    const int* token_ids,
    size_t token_ids_size,
    size_t* size
) {
    const auto* handle = static_cast<NeedleSPHandle*>(tokenizer);
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
    if (decoded.empty()) {
        return nullptr;
    }

    auto* buffer = static_cast<char*>(std::malloc(decoded.size() + 1));
    if (!buffer) return nullptr;
    std::memcpy(buffer, decoded.c_str(), decoded.size() + 1);
    return buffer;
}

int needle_sp_tokenizer_unk_token_id(needle_sp_tokenizer_t tokenizer) {
    const auto* handle = static_cast<NeedleSPHandle*>(tokenizer);
    if (!handle) return -1;
    return handle->processor.unk_id();
}

int needle_sp_tokenizer_bos_token_id(needle_sp_tokenizer_t tokenizer) {
    const auto* handle = static_cast<NeedleSPHandle*>(tokenizer);
    if (!handle) return -1;
    return handle->processor.bos_id();
}

int needle_sp_tokenizer_eos_token_id(needle_sp_tokenizer_t tokenizer) {
    const auto* handle = static_cast<NeedleSPHandle*>(tokenizer);
    return !handle ? -1 : handle->processor.eos_id();
}

int needle_sp_tokenizer_pad_token_id(needle_sp_tokenizer_t tokenizer) {
    const auto* handle = static_cast<NeedleSPHandle*>(tokenizer);
    return !handle ? -1 : handle->processor.pad_id();
}

size_t needle_sp_tokenizer_vocab_size(needle_sp_tokenizer_t tokenizer) {
    const auto* handle = static_cast<NeedleSPHandle*>(tokenizer);
    return !handle ? 0 : handle->processor.GetPieceSize();
}

int needle_sp_tokenizer_tokens_to_ids(
    needle_sp_tokenizer_t tokenizer,
    const char** tokens,
    int* token_ids,
    size_t size
) {
    const auto* handle = static_cast<NeedleSPHandle*>(tokenizer);
    if (!handle || !tokens || !token_ids) return -1;

    for (size_t i = 0; i < size; i++) {
        token_ids[i] = handle->processor.PieceToId(tokens[i]);
    }
    return 0;
}

int needle_sp_tokenizer_ids_to_tokens(
    needle_sp_tokenizer_t tokenizer,
    const int* token_ids,
    char** tokens,
    size_t size
) {
    const auto* handle = static_cast<NeedleSPHandle*>(tokenizer);
    if (!handle || !tokens || !token_ids) return -1;

    const auto piece_size = static_cast<size_t>(handle->processor.GetPieceSize());
    for (size_t i = 0; i < size; i++) {
        const auto id = token_ids[i];
        if (id < 0 || static_cast<size_t>(id) >= piece_size) {
            tokens[i][0] = '\0';
            continue;
        }
        const auto& token = handle->processor.IdToPiece(id);
        std::strcpy(tokens[i], token.c_str());
    }
    return 0;
}

int needle_sp_tokenizer_encoded_vocab(needle_sp_tokenizer_t tokenizer, char** encoded_vocab) {
    const auto* handle = static_cast<NeedleSPHandle*>(tokenizer);
    if (!handle || !encoded_vocab) return -1;

    for (size_t i = 0; i < needle_sp_tokenizer_vocab_size(tokenizer); i++) {
        const auto& token = handle->processor.IdToPiece(i);
        std::strcpy(encoded_vocab[i], token.c_str());
    }
    return 0;
}

void needle_sp_tokenizer_destroy(needle_sp_tokenizer_t tokenizer) {
    if (tokenizer) delete static_cast<NeedleSPHandle*>(tokenizer);
}

} // extern "C"
