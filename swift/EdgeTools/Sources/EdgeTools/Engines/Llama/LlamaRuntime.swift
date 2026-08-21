#if Llama && canImport(CLlama)
  import CLlama
  import EdgeToolsCore
  import EdgeToolsLlama
  import EdgeToolsTokenizers

  // MARK: - LlamaKVCacheType

  public struct LlamaKVCacheType: RawRepresentable, Hashable, Sendable {
    public let rawValue: UInt32

    public init(rawValue: UInt32) {
      self.rawValue = rawValue
    }

    public static let f32 = Self(rawValue: 0)
    public static let f16 = Self(rawValue: 1)
    // pi-lens-ignore: identifier_name
    public static let q4_0 = Self(rawValue: 2)
    // pi-lens-ignore: identifier_name
    public static let q8_0 = Self(rawValue: 8)
  }

  // MARK: - LlamaFlashAttention

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
    public var contextLength: UInt32
    public var maximumSequenceCount: UInt32
    public var unifiedKVCache: Bool
    public var threadCount: Int32
    public var flashAttention: LlamaFlashAttention
    public var keyCacheType: LlamaKVCacheType
    public var valueCacheType: LlamaKVCacheType

    public init(
      contextLength: UInt32 = 4096,
      maximumSequenceCount: UInt32 = 8,
      unifiedKVCache: Bool = true,
      threadCount: Int32 = 0,
      flashAttention: LlamaFlashAttention = .auto,
      keyCacheType: LlamaKVCacheType = .f16,
      valueCacheType: LlamaKVCacheType = .f16
    ) {
      precondition(maximumSequenceCount > 0)
      self.contextLength = contextLength
      self.maximumSequenceCount = maximumSequenceCount
      self.unifiedKVCache = unifiedKVCache
      self.threadCount = threadCount
      self.flashAttention = flashAttention
      self.keyCacheType = keyCacheType
      self.valueCacheType = valueCacheType
    }
  }

  // MARK: - LlamaRuntimeContext

  struct LlamaRuntimeContext: ~Copyable {
    let handle: OpaquePointer
    private let batch: UnsafeMutablePointer<llama_batch>

    var batchCapacity: Int {
      Int(llama_n_batch(self.handle))
    }

    var microBatchCapacity: Int {
      Int(llama_n_ubatch(self.handle))
    }

    init(handle: OpaquePointer) {
      self.handle = handle
      self.batch = UnsafeMutablePointer.allocate(capacity: 1)
      self.batch.initialize(
        to: llama_batch_init(Int32(clamping: llama_n_batch(handle)), 0, 1)
      )
    }

    deinit {
      llama_batch_free(self.batch.pointee)
      self.batch.deinitialize(count: 1)
      self.batch.deallocate()
      llama_free(self.handle)
    }

    borrowing func decode(
      tokenIds: some Collection<EdgeToolsToken.ID>,
      startPosition: Int,
      sequenceId: Int,
      wantsLogits: Bool
    ) throws {
      guard tokenIds.count <= self.batchCapacity else {
        throw LlamaRuntimeError(
          code: .decodeFailed,
          message: "The llama decode batch exceeds the configured batch capacity."
        )
      }
      var batch = self.batch.pointee
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
      let status = llama_decode(self.handle, batch)
      guard status == 0 else {
        throw llamaDecodeError(status: status)
      }
    }

    borrowing func lastLogits() -> UnsafeMutablePointer<Float>? {
      llama_get_logits_ith(self.handle, -1)
    }

    borrowing func memoryRemove(sequenceId: Int, from: Int, to: Int) -> Bool {
      llama_memory_seq_rm(
        llama_get_memory(self.handle), Int32(sequenceId), Int32(from), Int32(to)
      )
    }

    borrowing func memoryCopy(source: Int, destination: Int, from: Int, to: Int) -> Bool {
      let memory = llama_get_memory(self.handle)
      let sourceMinimumPosition = llama_memory_seq_pos_min(memory, Int32(source))
      let sourceMaximumPosition = llama_memory_seq_pos_max(memory, Int32(source))
      llama_memory_seq_cp(
        memory,
        Int32(source),
        Int32(destination),
        Int32(from),
        Int32(to)
      )
      return llama_memory_seq_pos_min(memory, Int32(destination)) == sourceMinimumPosition
        && llama_memory_seq_pos_max(memory, Int32(destination)) == sourceMaximumPosition
    }

    borrowing func probeConfidence(sequenceId: Int) -> Float? {
      let confidence = llama_probe_confidence(self.handle, Int32(sequenceId))
      return confidence < 0 ? nil : confidence
    }

    borrowing func probeReset(sequenceId: Int) {
      llama_probe_reset(self.handle, Int32(sequenceId))
    }
  }

  // MARK: - LlamaModel + Contexts

  extension LlamaModel {
    borrowing func createContext(parameters: LlamaContextParameters) throws -> LlamaRuntimeContext {
      var contextParameters = llama_context_default_params()
      contextParameters.n_ctx = parameters.contextLength
      contextParameters.n_seq_max = parameters.maximumSequenceCount
      contextParameters.kv_unified = parameters.unifiedKVCache
      if parameters.threadCount > 0 {
        contextParameters.n_threads = parameters.threadCount
        contextParameters.n_threads_batch = parameters.threadCount
      }
      contextParameters.flash_attn_type =
        llama_flash_attn_type(rawValue: parameters.flashAttention.rawValue)
      contextParameters.type_k = ggml_type(rawValue: parameters.keyCacheType.rawValue)
      contextParameters.type_v = ggml_type(rawValue: parameters.valueCacheType.rawValue)
      guard let handle = llama_init_from_model(self.handle, contextParameters) else {
        throw LlamaRuntimeError(
          code: .contextCreationFailed,
          message: "A llama context could not be created."
        )
      }
      return LlamaRuntimeContext(handle: handle)
    }
  }

  // MARK: - Helpers

  private func llamaDecodeError(status: Int32) -> LlamaRuntimeError {
    guard status == 1 else {
      return LlamaRuntimeError(
        code: .decodeFailed,
        message: "llama_decode failed with status \(status)."
      )
    }
    return LlamaRuntimeError(
      code: .contextLengthExceeded,
      message: "The batch does not fit in the context's remaining KV cache."
    )
  }
#endif
