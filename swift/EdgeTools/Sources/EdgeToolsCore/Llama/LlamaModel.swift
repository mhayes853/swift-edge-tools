#if Llama && canImport(CLlama)
  import CLlama

  #if FoundationEssentials
    import _EdgeToolsFoundation
  #endif

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
      public static let multimodalProjectorLoadFailed =
        Self(rawValue: "multimodal-projector-load-failed")
      public static let multimodalProcessingFailed = Self(rawValue: "multimodal-processing-failed")
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

  /// An owned llama.cpp model.
  ///
  /// The model is immutable after loading, and llama.cpp permits the model operations used by
  /// EdgeTools to run concurrently. Non-copyability ensures the handle is destroyed exactly once.
  public struct LlamaModel: ~Copyable, @unchecked Sendable {
    public let handle: OpaquePointer

    public init(path: String, parameters: LlamaModelParameters = LlamaModelParameters()) throws {
      _ = llamaBackendInitialized
      var modelParameters = llama_model_default_params()
      modelParameters.n_gpu_layers =
        parameters.gpuLayerCount == .max ? -1 : Int32(clamping: parameters.gpuLayerCount)
      modelParameters.use_mmap = parameters.useMemoryMapping
      modelParameters.vocab_only = parameters.vocabularyOnly
      guard let handle = llama_model_load_from_file(path, modelParameters) else {
        throw LlamaRuntimeError(
          code: .modelLoadFailed,
          message: "The model at \(path) could not be loaded."
        )
      }
      guard llama_model_get_vocab(handle) != nil else {
        llama_model_free(handle)
        throw LlamaRuntimeError(
          code: .vocabularyUnavailable,
          message: "The model at \(path) does not carry a vocabulary."
        )
      }
      self.handle = handle
    }

    /// Takes ownership of a llama.cpp model handle.
    public init(handle: consuming OpaquePointer) {
      self.handle = consume handle
    }

    deinit {
      llama_model_free(self.handle)
    }

    public borrowing func chatTemplate(named name: String? = nil) -> String? {
      let template =
        if let name {
          name.withCString { llama_model_chat_template(self.handle, $0) }
        } else {
          llama_model_chat_template(self.handle, nil)
        }
      return template.map { String(cString: $0) }
    }

    public borrowing func metadataValue(forKey key: String) -> String? {
      llamaMeasuredCString(
        measure: { llama_model_meta_val_str(self.handle, key, nil, 0) },
        fill: { llama_model_meta_val_str(self.handle, key, $0, $1) }
      )
    }
  }

  #if FoundationEssentials
    extension LlamaModel {
      public init(
        url: URL,
        parameters: LlamaModelParameters = LlamaModelParameters()
      ) throws {
        try self.init(path: url.path(), parameters: parameters)
      }
    }
  #endif

  // MARK: - LlamaModelBox

  package final class LlamaModelBox: Sendable {
    package let model: LlamaModel

    package init(model: consuming LlamaModel) {
      self.model = consume model
    }
  }

  // MARK: - LlamaModelMetadata

  public struct LlamaModelMetadata: ~Copyable, Sendable {
    private let model: LlamaModel

    public init(contentsOfGGUF path: String) throws {
      self.init(
        model: try LlamaModel(path: path, parameters: LlamaModelParameters(vocabularyOnly: true))
      )
    }

    public init(model: consuming LlamaModel) {
      self.model = consume model
    }

    public var architecture: String? {
      self["general.architecture"]
    }

    public var name: String? {
      self["general.name"]
    }

    public var chatTemplate: String? {
      self.model.chatTemplate()
    }

    public subscript(key: String) -> String? {
      self.model.metadataValue(forKey: key)
    }
  }

  #if FoundationEssentials
    extension LlamaModelMetadata {
      public init(contentsOfGGUF url: URL) throws {
        try self.init(contentsOfGGUF: url.path())
      }
    }
  #endif

  // MARK: - Helpers

  package func llamaMeasuredCString(
    measure: () -> Int32,
    fill: (UnsafeMutablePointer<CChar>?, Int) -> Int32
  ) -> String? {
    let measured = measure()
    let count = measured >= 0 ? measured : -measured
    var storage = [CChar](repeating: 0, count: Int(count) + 1)
    let written = storage.withUnsafeMutableBufferPointer { fill($0.baseAddress, $0.count) }
    guard written >= 0 else { return nil }
    return storage.withUnsafeBufferPointer { buffer in
      String(decoding: buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
  }

  private let llamaBackendInitialized: Bool = {
    llama_backend_init()
    return true
  }()
#endif
