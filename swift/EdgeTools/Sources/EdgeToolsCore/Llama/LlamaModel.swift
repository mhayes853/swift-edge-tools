#if Llama && canImport(CLlama)
  import CLlama

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
      public static let vocabularyUnavailable = Self(rawValue: "vocabulary-unavailable")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }
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

  // MARK: - System Info

  /// A description of the backend features the linked llama.cpp build was compiled with.
  public func llamaSystemInfo() -> String {
    String(cString: llama_print_system_info())
  }

  // MARK: - LlamaModel

  /// A loaded `llama_model`.
  ///
  /// A class rather than a non-copyable handle because one loaded model is shared by the
  /// tokenizer, the engine, and every fork family at once; the pointer is immutable for the
  /// lifetime of the handle and llama.cpp's model-level reads are thread-safe, so reference
  /// sharing is safe and ARC frees the model exactly once.
  ///
  /// The pointers are reachable only for the duration of a `withUnsafe…Pointer` call, so
  /// they cannot outlive the model that owns them.
  public final class LlamaModel: @unchecked Sendable {
    private let raw: OpaquePointer
    private let vocab: OpaquePointer

    public init(path: String, parameters: LlamaModelParameters = LlamaModelParameters()) throws {
      _ = llamaBackendInitialized
      var modelParameters = llama_model_default_params()
      modelParameters.n_gpu_layers =
        parameters.gpuLayerCount == .max ? -1 : Int32(clamping: parameters.gpuLayerCount)
      modelParameters.use_mmap = parameters.useMemoryMapping
      guard let raw = llama_model_load_from_file(path, modelParameters) else {
        throw LlamaRuntimeError(
          code: .modelLoadFailed,
          message: "The model at \(path) could not be loaded."
        )
      }
      guard let vocab = llama_model_get_vocab(raw) else {
        llama_model_free(raw)
        throw LlamaRuntimeError(
          code: .vocabularyUnavailable,
          message: "The model at \(path) does not carry a vocabulary."
        )
      }
      self.raw = raw
      self.vocab = vocab
    }

    deinit {
      llama_model_free(self.raw)
    }

    /// Calls `body` with the `llama_model *`, which is valid only for that call.
    public func withUnsafeModelPointer<R, E: Error>(
      _ body: (OpaquePointer) throws(E) -> R
    ) throws(E) -> R {
      try body(self.raw)
    }

    /// Calls `body` with the model's `llama_vocab *`, which is valid only for that call.
    ///
    /// The vocabulary is owned by the model — llama.cpp exposes no way to free it
    /// separately, and it dies with the model.
    public func withUnsafeVocabPointer<R, E: Error>(
      _ body: (OpaquePointer) throws(E) -> R
    ) throws(E) -> R {
      try body(self.vocab)
    }
  }

  // MARK: - Helpers

  /// Registers llama.cpp's backends once per process on the first model load.
  ///
  /// `llama_backend_free` is deliberately never called: it tears down state shared by every
  /// live model, so there is no point at which one engine can safely invoke it.
  private let llamaBackendInitialized: Bool = {
    llama_backend_init()
    return true
  }()
#endif
