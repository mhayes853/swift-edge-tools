#if Llama && canImport(CLlama)
  import CLlama
  import EdgeToolsCore

  // MARK: - Vendored LlamaApi

  extension LlamaApi {
    /// The vendored cactus-patched llama.cpp build.
    public static let vendored = LlamaApi(
      backend: Backend(
        initialize: { llama_backend_init() },
        shutdown: { llama_backend_free() }
      ),
      model: Model(
        load: { path, parameters in
          var modelParameters = llama_model_default_params()
          modelParameters.n_gpu_layers =
            parameters.gpuLayerCount == .max ? -1 : Int32(clamping: parameters.gpuLayerCount)
          modelParameters.use_mmap = parameters.useMemoryMapping
          guard let model = llama_model_load_from_file(path, modelParameters) else {
            throw LlamaRuntimeError(
              code: .modelLoadFailed,
              message: "The model at \(path) could not be loaded."
            )
          }
          return LlamaModelRef(rawValue: model)
        },
        free: { llama_model_free($0.rawValue) },
        metadataValue: { model, key in
          withMeasuredCString(
            measure: { llama_model_meta_val_str(model.rawValue, key, nil, 0) },
            fill: { llama_model_meta_val_str(model.rawValue, key, $0, $1) }
          )
        },
        chatTemplate: { model, name in
          llama_model_chat_template(model.rawValue, name).map { String(cString: $0) }
        },
        vocabularySize: { model in
          Int(llama_vocab_n_tokens(llama_model_get_vocab(model.rawValue)))
        },
        vocabKind: { model in
          switch llama_vocab_type(llama_model_get_vocab(model.rawValue)) {
          case LLAMA_VOCAB_TYPE_NONE: .none
          case LLAMA_VOCAB_TYPE_SPM: .sentencePiece
          case LLAMA_VOCAB_TYPE_BPE: .bytePairEncoding
          case LLAMA_VOCAB_TYPE_WPM: .wordPiece
          case LLAMA_VOCAB_TYPE_UGM: .unigram
          default: .other
          }
        },
        tokenText: { model, tokenId in
          llama_vocab_get_text(llama_model_get_vocab(model.rawValue), llama_token(tokenId))
            .map { String(cString: $0) }
        },
        eosToken: { model in
          let tokenId = llama_vocab_eos(llama_model_get_vocab(model.rawValue))
          return tokenId == LLAMA_TOKEN_NULL ? nil : EdgeToolsToken.ID(tokenId)
        },
        addsBOSToken: { model in
          llama_vocab_get_add_bos(llama_model_get_vocab(model.rawValue))
        },
        isEndOfGeneration: { model, tokenId in
          llama_vocab_is_eog(llama_model_get_vocab(model.rawValue), llama_token(tokenId))
        },
        tokenize: { model, text, addSpecialTokens, parseSpecialTokens in
          let vocab = llama_model_get_vocab(model.rawValue)
          let utf8 = Array(text.utf8)
          let count = utf8.withUnsafeBufferPointer { buffer in
            -llama_tokenize(
              vocab,
              buffer.baseAddress,
              Int32(buffer.count),
              nil,
              0,
              addSpecialTokens,
              parseSpecialTokens
            )
          }
          guard count >= 0 else {
            throw LlamaRuntimeError(
              code: .tokenizationFailed,
              message: "The text could not be tokenized."
            )
          }
          var tokens = [llama_token](repeating: 0, count: Int(count))
          let written = utf8.withUnsafeBufferPointer { buffer in
            tokens.withUnsafeMutableBufferPointer { tokens in
              llama_tokenize(
                vocab,
                buffer.baseAddress,
                Int32(buffer.count),
                tokens.baseAddress,
                Int32(tokens.count),
                addSpecialTokens,
                parseSpecialTokens
              )
            }
          }
          guard written >= 0 else {
            throw LlamaRuntimeError(
              code: .tokenizationFailed,
              message: "The text could not be tokenized."
            )
          }
          return tokens.prefix(Int(written)).map { EdgeToolsToken.ID($0) }
        },
        detokenize: { model, tokenIds, renderSpecialTokens in
          let vocab = llama_model_get_vocab(model.rawValue)
          let tokens = tokenIds.map { llama_token($0) }
          let text = tokens.withUnsafeBufferPointer { tokens in
            withMeasuredCString(
              measure: {
                llama_detokenize(
                  vocab,
                  tokens.baseAddress,
                  Int32(tokens.count),
                  nil,
                  0,
                  !renderSpecialTokens,
                  renderSpecialTokens
                )
              },
              fill: {
                llama_detokenize(
                  vocab,
                  tokens.baseAddress,
                  Int32(tokens.count),
                  $0,
                  Int32($1),
                  !renderSpecialTokens,
                  renderSpecialTokens
                )
              }
            )
          }
          guard let text else {
            throw LlamaRuntimeError(
              code: .tokenizationFailed,
              message: "The tokens could not be detokenized."
            )
          }
          return text
        },
        hasProbe: { llama_model_has_probe($0.rawValue) }
      ),
      context: Context(
        create: { model, parameters in
          var contextParameters = llama_context_default_params()
          contextParameters.n_ctx = UInt32(parameters.contextLength)
          contextParameters.n_seq_max = UInt32(parameters.maximumSequenceCount)
          contextParameters.kv_unified = parameters.unifiedKVCache
          if parameters.threadCount > 0 {
            contextParameters.n_threads = Int32(parameters.threadCount)
            contextParameters.n_threads_batch = Int32(parameters.threadCount)
          }
          contextParameters.flash_attn_type =
            switch parameters.flashAttention {
            case .auto: LLAMA_FLASH_ATTN_TYPE_AUTO
            case .disabled: LLAMA_FLASH_ATTN_TYPE_DISABLED
            case .enabled: LLAMA_FLASH_ATTN_TYPE_ENABLED
            }
          contextParameters.type_k = ggmlType(of: parameters.keyCacheType)
          contextParameters.type_v = ggmlType(of: parameters.valueCacheType)
          guard let context = llama_init_from_model(model.rawValue, contextParameters) else {
            throw LlamaRuntimeError(
              code: .contextCreationFailed,
              message: "A llama context could not be created."
            )
          }
          return LlamaContextRef(rawValue: context)
        },
        free: { llama_free($0.rawValue) },
        decode: { context, batch in
          var nativeBatch = llama_batch_init(Int32(batch.tokens.count), 0, 1)
          defer { llama_batch_free(nativeBatch) }
          nativeBatch.n_tokens = Int32(batch.tokens.count)
          for (index, tokenId) in batch.tokens.enumerated() {
            nativeBatch.token[index] = llama_token(tokenId)
            nativeBatch.pos[index] = llama_pos(batch.startPosition + index)
            nativeBatch.n_seq_id[index] = 1
            nativeBatch.seq_id[index]![0] = llama_seq_id(batch.sequenceId)
            nativeBatch.logits[index] = 0
          }
          if batch.wantsLogits && !batch.tokens.isEmpty {
            nativeBatch.logits[batch.tokens.count - 1] = 1
          }
          let status = llama_decode(context.rawValue, nativeBatch)
          guard status == 0 else {
            throw LlamaRuntimeError(
              code: .decodeFailed,
              message: "llama_decode failed with status \(status)."
            )
          }
        },
        lastLogits: { context in
          llama_get_logits_ith(context.rawValue, -1)
        },
        memoryRemove: { context, sequenceId, from, to in
          llama_memory_seq_rm(
            llama_get_memory(context.rawValue),
            llama_seq_id(sequenceId),
            llama_pos(from),
            llama_pos(to)
          )
        },
        memoryCopy: { context, source, destination, from, to in
          llama_memory_seq_cp(
            llama_get_memory(context.rawValue),
            llama_seq_id(source),
            llama_seq_id(destination),
            llama_pos(from),
            llama_pos(to)
          )
        },
        probeConfidence: { context, sequenceId in
          llama_probe_confidence(context.rawValue, llama_seq_id(sequenceId))
        },
        probeReset: { context, sequenceId in
          llama_probe_reset(context.rawValue, llama_seq_id(sequenceId))
        }
      )
    )
  }

  // MARK: - Helpers

  private func withMeasuredCString(
    measure: () -> Int32,
    fill: (UnsafeMutablePointer<CChar>?, Int) -> Int32
  ) -> String? {
    let count = measure()
    guard count >= 0 else { return nil }
    var storage = [CChar](repeating: 0, count: Int(count) + 1)
    let written = storage.withUnsafeMutableBufferPointer { fill($0.baseAddress, $0.count) }
    guard written >= 0 else { return nil }
    return storage.withUnsafeBufferPointer { buffer in
      String(decoding: buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
  }

  private func ggmlType(of cacheType: LlamaKVCacheType) -> ggml_type {
    switch cacheType {
    case .f32: GGML_TYPE_F32
    case .q8_0: GGML_TYPE_Q8_0
    case .q4_0: GGML_TYPE_Q4_0
    default: GGML_TYPE_F16
    }
  }
#endif
