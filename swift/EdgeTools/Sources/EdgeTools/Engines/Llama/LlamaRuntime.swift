#if Llama && canImport(CLlama)
  import CLlama
  import EdgeToolsCore
  import EdgeToolsTokenizers

  // MARK: - LlamaKVCacheType

  /// The raw values of the `ggml_type` cases usable for the KV cache.
  public struct LlamaKVCacheType: RawRepresentable, Hashable, Sendable {
    public let rawValue: Int32

    public init(rawValue: Int32) {
      self.rawValue = rawValue
    }

    public static let f32 = Self(rawValue: 0)
    public static let f16 = Self(rawValue: 1)
    public static let q4_0 = Self(rawValue: 2)
    public static let q8_0 = Self(rawValue: 8)
  }

  // MARK: - LlamaFlashAttention

  /// The raw values of llama.cpp's `llama_flash_attn_type` enum.
  public struct LlamaFlashAttention: RawRepresentable, Hashable, Sendable {
    public let rawValue: Int32

    public init(rawValue: Int32) {
      self.rawValue = rawValue
    }

    public static let auto = Self(rawValue: -1)
    public static let disabled = Self(rawValue: 0)
    public static let enabled = Self(rawValue: 1)
  }

  // MARK: - LlamaContextParameters

  public struct LlamaContextParameters: Hashable, Sendable {
    /// The token capacity of the context (`n_ctx`). Zero uses the model's training length.
    public var contextLength: Int
    /// The maximum number of forked sequences the context supports (`n_seq_max`).
    public var maximumSequenceCount: Int
    /// Shares KV cells between sequences (`kv_unified`), making forks copy-on-write.
    public var unifiedKVCache: Bool
    /// The number of threads used for decoding. Zero picks a hardware-based default.
    public var threadCount: Int
    public var flashAttention: LlamaFlashAttention
    public var keyCacheType: LlamaKVCacheType
    public var valueCacheType: LlamaKVCacheType

    public init(
      contextLength: Int = 4096,
      maximumSequenceCount: Int = 8,
      unifiedKVCache: Bool = true,
      threadCount: Int = 0,
      flashAttention: LlamaFlashAttention = .auto,
      keyCacheType: LlamaKVCacheType = .f16,
      valueCacheType: LlamaKVCacheType = .f16
    ) {
      self.contextLength = contextLength
      self.maximumSequenceCount = maximumSequenceCount
      self.unifiedKVCache = unifiedKVCache
      self.threadCount = threadCount
      self.flashAttention = flashAttention
      self.keyCacheType = keyCacheType
      self.valueCacheType = valueCacheType
    }
  }

  // MARK: - LlamaContextHandle

  /// A `llama_context` and the decode/KV operations that mutate it.
  ///
  /// Non-copyable so the context has exactly one owner and is freed on destruction. The
  /// handle carries no synchronization of its own; ``LlamaSequenceFamily`` owns one and
  /// serializes every call through its lock. Keeping it `Sendable` stops region isolation
  /// from tying the pointer to the lock state.
  struct LlamaContextHandle: ~Copyable, @unchecked Sendable {
    let raw: OpaquePointer

    deinit {
      llama_free(self.raw)
    }

    borrowing func decode(
      tokenIds: [EdgeToolsToken.ID],
      startPosition: Int,
      sequenceId: Int,
      wantsLogits: Bool
    ) throws {
      var batch = llama_batch_init(Int32(tokenIds.count), 0, 1)
      defer { llama_batch_free(batch) }
      batch.n_tokens = Int32(tokenIds.count)
      for (index, tokenId) in tokenIds.enumerated() {
        batch.token[index] = llama_token(tokenId)
        batch.pos[index] = llama_pos(startPosition + index)
        batch.n_seq_id[index] = 1
        batch.seq_id[index]![0] = llama_seq_id(sequenceId)
        batch.logits[index] = 0
      }
      if wantsLogits && !tokenIds.isEmpty {
        batch.logits[tokenIds.count - 1] = 1
      }
      let status = llama_decode(self.raw, batch)
      guard status == 0 else {
        throw LlamaRuntimeError(
          code: .decodeFailed,
          message: "llama_decode failed with status \(status)."
        )
      }
    }

    borrowing func lastLogits() -> UnsafeMutablePointer<Float>? {
      llama_get_logits_ith(self.raw, -1)
    }

    borrowing func memoryRemove(sequenceId: Int, from: Int, to: Int) -> Bool {
      llama_memory_seq_rm(
        llama_get_memory(self.raw), Int32(sequenceId), Int32(from), Int32(to)
      )
    }

    borrowing func memoryCopy(source: Int, destination: Int, from: Int, to: Int) {
      llama_memory_seq_cp(
        llama_get_memory(self.raw),
        Int32(source),
        Int32(destination),
        Int32(from),
        Int32(to)
      )
    }
  }

  // MARK: - LlamaModel + Contexts

  extension LlamaModel {
    func createContext(parameters: LlamaContextParameters) throws -> LlamaContextHandle {
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
      contextParameters.type_v = ggml_type(rawValue: UInt32(parameters.valueCacheType.rawValue))
      let context = self.withUnsafeModelPointer { llama_init_from_model($0, contextParameters) }
      guard let context else {
        throw LlamaRuntimeError(
          code: .contextCreationFailed,
          message: "A llama context could not be created."
        )
      }
      return LlamaContextHandle(raw: context)
    }
  }
#endif
