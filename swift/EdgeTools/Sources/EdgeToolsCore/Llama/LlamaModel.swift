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

    /// Loads the vocabulary and metadata without the weights.
    public var vocabularyOnly: Bool

    public init(
      gpuLayerCount: Int = .max,
      useMemoryMapping: Bool = true,
      vocabularyOnly: Bool = false
    ) {
      self.gpuLayerCount = gpuLayerCount
      self.useMemoryMapping = useMemoryMapping
      self.vocabularyOnly = vocabularyOnly
    }
  }

  // MARK: - System Info

  /// A description of the backend features the linked llama.cpp build was compiled with.
  public func llamaSystemInfo() -> String {
    String(cString: llama_print_system_info())
  }

  // MARK: - LlamaModel

  public final class LlamaModel: @unchecked Sendable {
    private let raw: OpaquePointer
    private let vocab: OpaquePointer

    public init(path: String, parameters: LlamaModelParameters = LlamaModelParameters()) throws {
      _ = llamaBackendInitialized
      var modelParameters = llama_model_default_params()
      modelParameters.n_gpu_layers =
        parameters.gpuLayerCount == .max ? -1 : Int32(clamping: parameters.gpuLayerCount)
      modelParameters.use_mmap = parameters.useMemoryMapping
      modelParameters.vocab_only = parameters.vocabularyOnly
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

    public func withUnsafeModelPointer<R, E: Error>(
      _ body: (OpaquePointer) throws(E) -> R
    ) throws(E) -> R {
      try body(self.raw)
    }

    public func withUnsafeVocabPointer<R, E: Error>(
      _ body: (OpaquePointer) throws(E) -> R
    ) throws(E) -> R {
      try body(self.vocab)
    }
  }

  // MARK: - LlamaModelMetadata

  public struct LlamaModelMetadata: Sendable {
    private let model: LlamaModel

    public init(contentsOfGGUF path: String) throws {
      self.init(
        model: try LlamaModel(path: path, parameters: LlamaModelParameters(vocabularyOnly: true))
      )
    }

    public init(model: LlamaModel) {
      self.model = model
    }

    public var architecture: String? {
      self["general.architecture"]
    }

    public var name: String? {
      self["general.name"]
    }

    public var chatTemplate: String? {
      self.model.withUnsafeModelPointer { model in
        llama_model_chat_template(model, nil).map { String(cString: $0) }
      }
    }

    public subscript(key: String) -> String? {
      self.model.withUnsafeModelPointer { model in
        let length = llama_model_meta_val_str(model, key, nil, 0)
        guard length > 0 else { return nil }
        return String(unsafeUninitializedCapacity: Int(length) + 1) { buffer in
          buffer.withMemoryRebound(to: CChar.self) { characters in
            Int(llama_model_meta_val_str(model, key, characters.baseAddress, characters.count))
          }
        }
      }
    }
  }

  // MARK: - Helpers

  private let llamaBackendInitialized: Bool = {
    llama_backend_init()
    return true
  }()
#endif
