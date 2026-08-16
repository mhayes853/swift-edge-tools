#if Llama && canImport(CLlama)
  import CLlama

  // MARK: - Vendored LlamaAPI

  extension LlamaAPI {
    /// The vendored cactus-patched llama.cpp build.
    public static let vendored = LlamaAPI(
      backendInit: llama_backend_init,
      backendFree: llama_backend_free,
      modelLoad: { path, parameters in
        var modelParameters = llama_model_default_params()
        modelParameters.n_gpu_layers =
          parameters.gpuLayerCount == .max ? -1 : Int32(clamping: parameters.gpuLayerCount)
        modelParameters.use_mmap = parameters.useMemoryMapping
        return llama_model_load_from_file(path, modelParameters)
      },
      modelFree: llama_model_free,
      modelGetVocab: llama_model_get_vocab,
      modelMetaValStr: llama_model_meta_val_str,
      modelChatTemplate: llama_model_chat_template,
      modelHasProbe: llama_model_has_probe,
      vocabNTokens: llama_vocab_n_tokens,
      vocabType: { Int32(llama_vocab_type($0).rawValue) },
      vocabGetText: llama_vocab_get_text,
      vocabBOS: llama_vocab_bos,
      vocabEOS: llama_vocab_eos,
      vocabGetAddBOS: llama_vocab_get_add_bos,
      vocabIsEOG: llama_vocab_is_eog,
      tokenize: llama_tokenize,
      detokenize: llama_detokenize,
      contextInit: { model, parameters in
        var contextParameters = llama_context_default_params()
        contextParameters.n_ctx = UInt32(parameters.contextLength)
        contextParameters.n_seq_max = UInt32(parameters.maximumSequenceCount)
        contextParameters.kv_unified = parameters.unifiedKVCache
        if parameters.threadCount > 0 {
          contextParameters.n_threads = Int32(parameters.threadCount)
          contextParameters.n_threads_batch = Int32(parameters.threadCount)
        }
        contextParameters.flash_attn_type =
          llama_flash_attn_type(rawValue: parameters.flashAttention.rawValue)
        contextParameters.type_k = ggml_type(rawValue: UInt32(parameters.keyCacheType.rawValue))
        contextParameters.type_v = ggml_type(
          rawValue: UInt32(parameters.valueCacheType.rawValue)
        )
        return llama_init_from_model(model, contextParameters)
      },
      contextFree: llama_free,
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
        return llama_decode(context, nativeBatch)
      },
      getLogitsIth: llama_get_logits_ith,
      getMemory: llama_get_memory,
      memorySeqRm: llama_memory_seq_rm,
      memorySeqCp: llama_memory_seq_cp,
      probeConfidence: llama_probe_confidence,
      probeReset: llama_probe_reset
    )
  }
#endif
