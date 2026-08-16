#if LlamaCore
  import EdgeToolsCore

  // MARK: - Handles

  /// A loaded `llama_model`.
  ///
  /// Handles wrap opaque C pointers owned by the engine; the runtime that created a handle
  /// is the only one that may operate on it.
  public struct LlamaModelRef: Hashable, @unchecked Sendable {
    public let rawValue: OpaquePointer

    public init(rawValue: OpaquePointer) {
      self.rawValue = rawValue
    }
  }

  /// A `llama_context` created from a model.
  public struct LlamaContextRef: Hashable, @unchecked Sendable {
    public let rawValue: OpaquePointer

    public init(rawValue: OpaquePointer) {
      self.rawValue = rawValue
    }
  }

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

  public enum LlamaVocabKind: Hashable, Sendable {
    case none
    case sentencePiece
    case bytePairEncoding
    case wordPiece
    case unigram
    case other
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

  public struct LlamaKVCacheType: RawRepresentable, Hashable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
      self.rawValue = rawValue
    }

    public static let f32 = Self(rawValue: "f32")
    public static let f16 = Self(rawValue: "f16")
    public static let q8_0 = Self(rawValue: "q8_0")
    public static let q4_0 = Self(rawValue: "q4_0")
  }

  // MARK: - LlamaFlashAttention

  public enum LlamaFlashAttention: Hashable, Sendable {
    case auto
    case disabled
    case enabled
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

  // MARK: - LlamaApi

  /// The llama.cpp surface the engine calls through.
  ///
  /// llama.cpp exposes plain C symbols that bind at link time, so unlike ONNX Runtime there
  /// is no function-pointer table to hand around. This struct plays that role: the `Llama`
  /// trait provides an instance wired to the vendored cactus-patched build, and `LlamaCore`
  /// users construct one against their own llama.cpp.
  public struct LlamaApi: Sendable {
    public struct Backend: Sendable {
      public var initialize: @Sendable () -> Void
      public var shutdown: @Sendable () -> Void

      public init(
        initialize: @escaping @Sendable () -> Void,
        shutdown: @escaping @Sendable () -> Void
      ) {
        self.initialize = initialize
        self.shutdown = shutdown
      }
    }

    public struct Model: Sendable {
      public var load: @Sendable (_ path: String, _ parameters: LlamaModelParameters) throws ->
        LlamaModelRef
      public var free: @Sendable (LlamaModelRef) -> Void
      public var metadataValue: @Sendable (LlamaModelRef, _ key: String) -> String?
      public var chatTemplate: @Sendable (LlamaModelRef, _ name: String?) -> String?
      public var vocabularySize: @Sendable (LlamaModelRef) -> Int
      public var vocabKind: @Sendable (LlamaModelRef) -> LlamaVocabKind
      public var tokenText: @Sendable (LlamaModelRef, EdgeToolsToken.ID) -> String?
      public var eosToken: @Sendable (LlamaModelRef) -> EdgeToolsToken.ID?
      public var bosToken: @Sendable (LlamaModelRef) -> EdgeToolsToken.ID?
      public var addsBOSToken: @Sendable (LlamaModelRef) -> Bool
      public var isEndOfGeneration: @Sendable (LlamaModelRef, EdgeToolsToken.ID) -> Bool
      public var tokenize: @Sendable (
        LlamaModelRef, _ text: String, _ addSpecialTokens: Bool, _ parseSpecialTokens: Bool
      ) throws -> [EdgeToolsToken.ID]
      public var detokenize: @Sendable (
        LlamaModelRef, _ tokens: [EdgeToolsToken.ID], _ renderSpecialTokens: Bool
      ) throws -> String
      /// Present when the build carries the cactus handoff probe tensors support.
      public var hasProbe: (@Sendable (LlamaModelRef) -> Bool)?

      public init(
        load: @escaping @Sendable (String, LlamaModelParameters) throws -> LlamaModelRef,
        free: @escaping @Sendable (LlamaModelRef) -> Void,
        metadataValue: @escaping @Sendable (LlamaModelRef, String) -> String?,
        chatTemplate: @escaping @Sendable (LlamaModelRef, String?) -> String?,
        vocabularySize: @escaping @Sendable (LlamaModelRef) -> Int,
        vocabKind: @escaping @Sendable (LlamaModelRef) -> LlamaVocabKind,
        tokenText: @escaping @Sendable (LlamaModelRef, EdgeToolsToken.ID) -> String?,
        eosToken: @escaping @Sendable (LlamaModelRef) -> EdgeToolsToken.ID?,
        bosToken: @escaping @Sendable (LlamaModelRef) -> EdgeToolsToken.ID?,
        addsBOSToken: @escaping @Sendable (LlamaModelRef) -> Bool,
        isEndOfGeneration: @escaping @Sendable (LlamaModelRef, EdgeToolsToken.ID) -> Bool,
        tokenize: @escaping @Sendable (LlamaModelRef, String, Bool, Bool) throws ->
          [EdgeToolsToken.ID],
        detokenize: @escaping @Sendable (LlamaModelRef, [EdgeToolsToken.ID], Bool) throws ->
          String,
        hasProbe: (@Sendable (LlamaModelRef) -> Bool)? = nil
      ) {
        self.load = load
        self.free = free
        self.metadataValue = metadataValue
        self.chatTemplate = chatTemplate
        self.vocabularySize = vocabularySize
        self.vocabKind = vocabKind
        self.tokenText = tokenText
        self.eosToken = eosToken
        self.bosToken = bosToken
        self.addsBOSToken = addsBOSToken
        self.isEndOfGeneration = isEndOfGeneration
        self.tokenize = tokenize
        self.detokenize = detokenize
        self.hasProbe = hasProbe
      }
    }

    public struct Context: Sendable {
      public var create: @Sendable (LlamaModelRef, LlamaContextParameters) throws ->
        LlamaContextRef
      public var free: @Sendable (LlamaContextRef) -> Void
      public var decode: @Sendable (LlamaContextRef, LlamaDecodeBatch) throws -> Void
      /// The logits of the final token of the last decoded batch, `vocabularySize` wide.
      public var lastLogits: @Sendable (LlamaContextRef) -> UnsafeMutablePointer<Float>?
      /// Removes positions `[from, to)` of a sequence; a negative bound spans the edge.
      public var memoryRemove: @Sendable (
        LlamaContextRef, _ sequenceId: Int, _ from: Int, _ to: Int
      ) -> Bool
      /// Copies positions `[from, to)` between sequences; a negative bound spans the edge.
      public var memoryCopy: @Sendable (
        LlamaContextRef, _ source: Int, _ destination: Int, _ from: Int, _ to: Int
      ) -> Void
      /// Present when the build carries the cactus handoff probe runtime.
      public var probeConfidence: (@Sendable (LlamaContextRef, _ sequenceId: Int) -> Float)?
      /// Present when the build carries the cactus handoff probe runtime.
      public var probeReset: (@Sendable (LlamaContextRef, _ sequenceId: Int) -> Void)?

      public init(
        create: @escaping @Sendable (LlamaModelRef, LlamaContextParameters) throws ->
          LlamaContextRef,
        free: @escaping @Sendable (LlamaContextRef) -> Void,
        decode: @escaping @Sendable (LlamaContextRef, LlamaDecodeBatch) throws -> Void,
        lastLogits: @escaping @Sendable (LlamaContextRef) -> UnsafeMutablePointer<Float>?,
        memoryRemove: @escaping @Sendable (LlamaContextRef, Int, Int, Int) -> Bool,
        memoryCopy: @escaping @Sendable (LlamaContextRef, Int, Int, Int, Int) -> Void,
        probeConfidence: (@Sendable (LlamaContextRef, Int) -> Float)? = nil,
        probeReset: (@Sendable (LlamaContextRef, Int) -> Void)? = nil
      ) {
        self.create = create
        self.free = free
        self.decode = decode
        self.lastLogits = lastLogits
        self.memoryRemove = memoryRemove
        self.memoryCopy = memoryCopy
        self.probeConfidence = probeConfidence
        self.probeReset = probeReset
      }
    }

    public var backend: Backend
    public var model: Model
    public var context: Context

    public init(backend: Backend, model: Model, context: Context) {
      self.backend = backend
      self.model = model
      self.context = context
    }
  }
#endif
