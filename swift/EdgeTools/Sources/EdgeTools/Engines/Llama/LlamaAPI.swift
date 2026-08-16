#if LlamaCore
  import EdgeToolsCore

  // MARK: - LlamaRuntimeError

  public struct LlamaRuntimeError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let modelLoadFailed = Self(rawValue: "model-load-failed")
      public static let contextCreationFailed = Self(rawValue: "context-creation-failed")
      public static let tokenizationFailed = Self(rawValue: "tokenization-failed")
      public static let decodeFailed = Self(rawValue: "decode-failed")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }
  }

  // MARK: - LlamaVocabKind

  /// The raw values of llama.cpp's `llama_vocab_type` enum.
  public struct LlamaVocabKind: RawRepresentable, Hashable, Sendable {
    public let rawValue: Int32

    public init(rawValue: Int32) {
      self.rawValue = rawValue
    }

    public static let none = Self(rawValue: 0)
    public static let sentencePiece = Self(rawValue: 1)
    public static let bytePairEncoding = Self(rawValue: 2)
    public static let wordPiece = Self(rawValue: 3)
    public static let unigram = Self(rawValue: 4)
  }

  // MARK: - LlamaModelParameters

  public struct LlamaModelParameters: Hashable, Sendable {
    /// The number of layers to offload to the GPU. Defaults to all layers.
    public var gpuLayerCount: Int
    public var useMemoryMapping: Bool

    public init(
      gpuLayerCount: Int = .max,
      useMemoryMapping: Bool = true
    ) {
      self.gpuLayerCount = gpuLayerCount
      self.useMemoryMapping = useMemoryMapping
    }
  }

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

  // MARK: - LlamaDecodeBatch

  /// One decode submission: tokens for a single sequence at explicit positions.
  public struct LlamaDecodeBatch: Hashable, Sendable {
    public var tokens: [EdgeToolsToken.ID]
    /// The position of the first token; later tokens follow contiguously.
    public var startPosition: Int
    public var sequenceId: Int
    /// Requests logits for the final token of the batch.
    public var wantsLogits: Bool

    public init(
      tokens: [EdgeToolsToken.ID],
      startPosition: Int,
      sequenceId: Int,
      wantsLogits: Bool = true
    ) {
      self.tokens = tokens
      self.startPosition = startPosition
      self.sequenceId = sequenceId
      self.wantsLogits = wantsLogits
    }
  }

  // MARK: - LlamaAPI

  /// The llama.cpp C symbols the engine calls.
  ///
  /// llama.cpp binds symbols at link time, so unlike ONNX Runtime there is no function
  /// pointer table to hand around; this struct plays that role. Every field whose C
  /// signature is expressible across modules stores the symbol itself, so a custom build
  /// is wired by handing functions over directly:
  ///
  /// ```swift
  /// import myllama
  ///
  /// let api = LlamaAPI(
  ///   backendInit: llama_backend_init,
  ///   backendFree: llama_backend_free,
  ///   // ...
  /// )
  /// ```
  ///
  /// Only the operations whose C signatures name non-opaque structs (`modelLoad`,
  /// `contextInit`, and `decode` use `llama_model_params`, `llama_context_params`, and
  /// `llama_batch`) and the enum-returning `vocabType` take small adapter closures, since
  /// those types cannot be shared between C modules. The `Llama` trait provides
  /// ``LlamaAPI/vendored`` wired to the vendored cactus-patched build.
  public struct LlamaAPI: Sendable {
    /// `llama_backend_init`
    nonisolated(unsafe) public var backendInit: () -> Void
    /// `llama_backend_free`
    nonisolated(unsafe) public var backendFree: () -> Void
    /// Loads a model; the body typically calls `llama_model_load_from_file` with
    /// `llama_model_default_params` adjusted from the parameters.
    nonisolated(unsafe) public var modelLoad: (String, LlamaModelParameters) -> OpaquePointer?
    /// `llama_model_free`
    nonisolated(unsafe) public var modelFree: (OpaquePointer?) -> Void
    /// `llama_model_get_vocab`
    nonisolated(unsafe) public var modelGetVocab: (OpaquePointer?) -> OpaquePointer?
    /// `llama_model_meta_val_str`
    nonisolated(unsafe) public var modelMetaValStr:
      (OpaquePointer?, UnsafePointer<CChar>?, UnsafeMutablePointer<CChar>?, Int) ->
        Int32
    /// `llama_model_chat_template`
    nonisolated(unsafe) public var modelChatTemplate:
      (OpaquePointer?, UnsafePointer<CChar>?) -> UnsafePointer<CChar>?
    /// `llama_model_has_probe` when the build carries the cactus handoff probe.
    nonisolated(unsafe) public var modelHasProbe: ((OpaquePointer?) -> Bool)?
    /// `llama_vocab_n_tokens`
    nonisolated(unsafe) public var vocabNTokens: (OpaquePointer?) -> Int32
    /// The raw value of `llama_vocab_type`; C enums cannot be shared between modules, so
    /// the body is `{ Int32(llama_vocab_type($0).rawValue) }`.
    nonisolated(unsafe) public var vocabType: (OpaquePointer?) -> Int32
    /// `llama_vocab_get_text`
    nonisolated(unsafe) public var vocabGetText: (OpaquePointer?, Int32) -> UnsafePointer<CChar>?
    /// `llama_vocab_bos`
    nonisolated(unsafe) public var vocabBOS: (OpaquePointer?) -> Int32
    /// `llama_vocab_eos`
    nonisolated(unsafe) public var vocabEOS: (OpaquePointer?) -> Int32
    /// `llama_vocab_get_add_bos`
    nonisolated(unsafe) public var vocabGetAddBOS: (OpaquePointer?) -> Bool
    /// `llama_vocab_is_eog`
    nonisolated(unsafe) public var vocabIsEOG: (OpaquePointer?, Int32) -> Bool
    /// `llama_tokenize`
    nonisolated(unsafe) public var tokenize:
      (
        OpaquePointer?, UnsafePointer<CChar>?, Int32, UnsafeMutablePointer<Int32>?, Int32,
        Bool, Bool
      ) -> Int32
    /// `llama_detokenize`
    nonisolated(unsafe) public var detokenize:
      (
        OpaquePointer?, UnsafePointer<Int32>?, Int32, UnsafeMutablePointer<CChar>?, Int32,
        Bool, Bool
      ) -> Int32
    /// Creates a context; the body typically calls `llama_init_from_model` with
    /// `llama_context_default_params` adjusted from the parameters.
    nonisolated(unsafe) public var contextInit: (OpaquePointer?, LlamaContextParameters) -> OpaquePointer?
    /// `llama_free`
    nonisolated(unsafe) public var contextFree: (OpaquePointer?) -> Void
    /// Decodes one batch; the body typically fills a `llama_batch_init` batch and calls
    /// `llama_decode`, returning its status.
    nonisolated(unsafe) public var decode: (OpaquePointer?, LlamaDecodeBatch) -> Int32
    /// `llama_get_logits_ith`
    nonisolated(unsafe) public var getLogitsIth: (OpaquePointer?, Int32) -> UnsafeMutablePointer<Float>?
    /// `llama_get_memory`
    nonisolated(unsafe) public var getMemory: (OpaquePointer?) -> OpaquePointer?
    /// `llama_memory_seq_rm`
    nonisolated(unsafe) public var memorySeqRm: (OpaquePointer?, Int32, Int32, Int32) -> Bool
    /// `llama_memory_seq_cp`
    nonisolated(unsafe) public var memorySeqCp: (OpaquePointer?, Int32, Int32, Int32, Int32) -> Void
    /// `llama_probe_confidence` when the build carries the cactus handoff probe.
    nonisolated(unsafe) public var probeConfidence: ((OpaquePointer?, Int32) -> Float)?
    /// `llama_probe_reset` when the build carries the cactus handoff probe.
    nonisolated(unsafe) public var probeReset: ((OpaquePointer?, Int32) -> Void)?

    public init(
      backendInit: @escaping () -> Void,
      backendFree: @escaping () -> Void,
      modelLoad: @escaping (String, LlamaModelParameters) -> OpaquePointer?,
      modelFree: @escaping (OpaquePointer?) -> Void,
      modelGetVocab: @escaping (OpaquePointer?) -> OpaquePointer?,
      modelMetaValStr: @escaping (
        OpaquePointer?, UnsafePointer<CChar>?, UnsafeMutablePointer<CChar>?, Int
      ) -> Int32,
      modelChatTemplate: @escaping (OpaquePointer?, UnsafePointer<CChar>?) ->
        UnsafePointer<CChar>?,
      modelHasProbe: ((OpaquePointer?) -> Bool)? = nil,
      vocabNTokens: @escaping (OpaquePointer?) -> Int32,
      vocabType: @escaping (OpaquePointer?) -> Int32,
      vocabGetText: @escaping (OpaquePointer?, Int32) -> UnsafePointer<CChar>?,
      vocabBOS: @escaping (OpaquePointer?) -> Int32,
      vocabEOS: @escaping (OpaquePointer?) -> Int32,
      vocabGetAddBOS: @escaping (OpaquePointer?) -> Bool,
      vocabIsEOG: @escaping (OpaquePointer?, Int32) -> Bool,
      tokenize: @escaping (
        OpaquePointer?, UnsafePointer<CChar>?, Int32, UnsafeMutablePointer<Int32>?, Int32,
        Bool, Bool
      ) -> Int32,
      detokenize: @escaping (
        OpaquePointer?, UnsafePointer<Int32>?, Int32, UnsafeMutablePointer<CChar>?, Int32,
        Bool, Bool
      ) -> Int32,
      contextInit: @escaping (OpaquePointer?, LlamaContextParameters) ->
        OpaquePointer?,
      contextFree: @escaping (OpaquePointer?) -> Void,
      decode: @escaping (OpaquePointer?, LlamaDecodeBatch) -> Int32,
      getLogitsIth: @escaping (OpaquePointer?, Int32) -> UnsafeMutablePointer<Float>?,
      getMemory: @escaping (OpaquePointer?) -> OpaquePointer?,
      memorySeqRm: @escaping (OpaquePointer?, Int32, Int32, Int32) -> Bool,
      memorySeqCp: @escaping (OpaquePointer?, Int32, Int32, Int32, Int32) -> Void,
      probeConfidence: ((OpaquePointer?, Int32) -> Float)? = nil,
      probeReset: ((OpaquePointer?, Int32) -> Void)? = nil
    ) {
      self.backendInit = backendInit
      self.backendFree = backendFree
      self.modelLoad = modelLoad
      self.modelFree = modelFree
      self.modelGetVocab = modelGetVocab
      self.modelMetaValStr = modelMetaValStr
      self.modelChatTemplate = modelChatTemplate
      self.modelHasProbe = modelHasProbe
      self.vocabNTokens = vocabNTokens
      self.vocabType = vocabType
      self.vocabGetText = vocabGetText
      self.vocabBOS = vocabBOS
      self.vocabEOS = vocabEOS
      self.vocabGetAddBOS = vocabGetAddBOS
      self.vocabIsEOG = vocabIsEOG
      self.tokenize = tokenize
      self.detokenize = detokenize
      self.contextInit = contextInit
      self.contextFree = contextFree
      self.decode = decode
      self.getLogitsIth = getLogitsIth
      self.getMemory = getMemory
      self.memorySeqRm = memorySeqRm
      self.memorySeqCp = memorySeqCp
      self.probeConfidence = probeConfidence
      self.probeReset = probeReset
    }
  }

  // MARK: - Model Operations

  extension LlamaAPI {
    func loadModel(
      path: String,
      parameters: LlamaModelParameters
    ) throws -> OpaquePointer {
      guard let model = self.modelLoad(path, parameters) else {
        throw LlamaRuntimeError(
          code: .modelLoadFailed,
          message: "The model at \(path) could not be loaded."
        )
      }
      return model
    }

    func metadataValue(model: OpaquePointer, key: String) -> String? {
      key.withCString { key in
        measuredCString(
          measure: { self.modelMetaValStr(model, key, nil, 0) },
          fill: { self.modelMetaValStr(model, key, $0, $1) }
        )
      }
    }

    func chatTemplate(model: OpaquePointer, name: String?) -> String? {
      let template =
        if let name {
          name.withCString { self.modelChatTemplate(model, $0) }
        } else {
          self.modelChatTemplate(model, nil)
        }
      return template.map { String(cString: $0) }
    }

    func vocabularySize(model: OpaquePointer) -> Int {
      Int(self.vocabNTokens(self.modelGetVocab(model)))
    }

    func vocabKind(model: OpaquePointer) -> LlamaVocabKind {
      LlamaVocabKind(rawValue: self.vocabType(self.modelGetVocab(model)))
    }

    func tokenText(model: OpaquePointer, tokenId: EdgeToolsToken.ID) -> String? {
      self.vocabGetText(self.modelGetVocab(model), Int32(tokenId)).map { String(cString: $0) }
    }

    func eosToken(model: OpaquePointer) -> EdgeToolsToken.ID? {
      let tokenId = self.vocabEOS(self.modelGetVocab(model))
      return tokenId == llamaNullToken ? nil : EdgeToolsToken.ID(tokenId)
    }

    func bosToken(model: OpaquePointer) -> EdgeToolsToken.ID? {
      let tokenId = self.vocabBOS(self.modelGetVocab(model))
      return tokenId == llamaNullToken ? nil : EdgeToolsToken.ID(tokenId)
    }

    func addsBOSToken(model: OpaquePointer) -> Bool {
      self.vocabGetAddBOS(self.modelGetVocab(model))
    }

    func isEndOfGeneration(model: OpaquePointer, tokenId: EdgeToolsToken.ID) -> Bool {
      self.vocabIsEOG(self.modelGetVocab(model), Int32(tokenId))
    }

    func hasProbe(model: OpaquePointer) -> Bool {
      self.modelHasProbe?(model) ?? false
    }

    func tokenIds(
      model: OpaquePointer,
      text: String,
      addSpecialTokens: Bool,
      parseSpecialTokens: Bool
    ) throws -> [EdgeToolsToken.ID] {
      let vocab = self.modelGetVocab(model)
      let utf8Count = Int32(text.utf8.count)
      return try text.withCString { text in
        let count = -self.tokenize(
          vocab, text, utf8Count, nil, 0, addSpecialTokens, parseSpecialTokens
        )
        guard count >= 0 else {
          throw LlamaRuntimeError(
            code: .tokenizationFailed,
            message: "The text could not be tokenized."
          )
        }
        var tokens = [Int32](repeating: 0, count: Int(count))
        let written = tokens.withUnsafeMutableBufferPointer { tokens in
          self.tokenize(
            vocab,
            text,
            utf8Count,
            tokens.baseAddress,
            Int32(tokens.count),
            addSpecialTokens,
            parseSpecialTokens
          )
        }
        guard written >= 0 else {
          throw LlamaRuntimeError(
            code: .tokenizationFailed,
            message: "The text could not be tokenized."
          )
        }
        return tokens.prefix(Int(written)).map { EdgeToolsToken.ID($0) }
      }
    }

    func text(
      model: OpaquePointer,
      tokenIds: [EdgeToolsToken.ID],
      renderSpecialTokens: Bool
    ) throws -> String {
      let vocab = self.modelGetVocab(model)
      let tokens = tokenIds.map { Int32($0) }
      let text = tokens.withUnsafeBufferPointer { tokens in
        measuredCString(
          measure: {
            self.detokenize(
              vocab, tokens.baseAddress, Int32(tokens.count), nil, 0,
              !renderSpecialTokens, renderSpecialTokens
            )
          },
          fill: {
            self.detokenize(
              vocab, tokens.baseAddress, Int32(tokens.count), $0, Int32($1),
              !renderSpecialTokens, renderSpecialTokens
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
    }
  }

  // MARK: - Context Operations

  extension LlamaAPI {
    func createContext(
      model: OpaquePointer,
      parameters: LlamaContextParameters
    ) throws -> OpaquePointer {
      guard let context = self.contextInit(model, parameters) else {
        throw LlamaRuntimeError(
          code: .contextCreationFailed,
          message: "A llama context could not be created."
        )
      }
      return context
    }

    func decode(context: OpaquePointer, batch: LlamaDecodeBatch) throws {
      let status = self.decode(context, batch)
      guard status == 0 else {
        throw LlamaRuntimeError(
          code: .decodeFailed,
          message: "llama_decode failed with status \(status)."
        )
      }
    }

    func lastLogits(context: OpaquePointer) -> UnsafeMutablePointer<Float>? {
      self.getLogitsIth(context, -1)
    }

    func memoryRemove(context: OpaquePointer, sequenceId: Int, from: Int, to: Int) -> Bool {
      self.memorySeqRm(self.getMemory(context), Int32(sequenceId), Int32(from), Int32(to))
    }

    func memoryCopy(
      context: OpaquePointer,
      source: Int,
      destination: Int,
      from: Int,
      to: Int
    ) {
      self.memorySeqCp(
        self.getMemory(context),
        Int32(source),
        Int32(destination),
        Int32(from),
        Int32(to)
      )
    }
  }

  // MARK: - Helpers

  private let llamaNullToken = Int32(-1)

  private func measuredCString(
    measure: () -> Int32,
    fill: (UnsafeMutablePointer<CChar>?, Int) -> Int32
  ) -> String? {
    // Measuring calls report the required size either directly (`llama_model_meta_val_str`)
    // or negated (`llama_detokenize`); a fill that still fails reports the true error.
    let measured = measure()
    let count = measured >= 0 ? measured : -measured
    var storage = [CChar](repeating: 0, count: Int(count) + 1)
    let written = storage.withUnsafeMutableBufferPointer { fill($0.baseAddress, $0.count) }
    guard written >= 0 else { return nil }
    return storage.withUnsafeBufferPointer { buffer in
      String(decoding: buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
  }
#endif
