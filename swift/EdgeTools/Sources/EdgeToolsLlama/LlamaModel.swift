#if canImport(CLlama)
  import CLlama

  #if FoundationEssentials
    import _EdgeToolsFoundation
  #endif

  // MARK: - LlamaModelParameters

  public struct LlamaModelParameters: Hashable, Sendable {
    public var gpuLayerCount: Int
    public var useMemoryMapping: Bool
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

  public func llamaSystemInfo() -> String {
    String(cString: llama_print_system_info())
  }

  // MARK: - LlamaModel

  /// An owned llama.cpp model.
  ///
  /// Initialize `LlamaBackend` before creating a model directly and keep it initialized for the
  /// model's lifetime. Higher-level engine APIs manage this lifecycle automatically.
  public struct LlamaModel: ~Copyable, @unchecked Sendable {
    public let handle: OpaquePointer

    public init(path: String, parameters: LlamaModelParameters = LlamaModelParameters()) throws {
      var modelParameters = llama_model_default_params()
      modelParameters.n_gpu_layers =
        parameters.gpuLayerCount == .max ? -1 : Int32(clamping: parameters.gpuLayerCount)
      modelParameters.use_mmap = parameters.useMemoryMapping
      modelParameters.vocab_only = parameters.vocabularyOnly
      guard let handle = llama_model_load_from_file(path, modelParameters) else {
        let message = "The model at \(path) could not be loaded."
        throw LlamaRuntimeError(code: .modelLoadFailed, message: message)
      }
      self.handle = handle
    }

    public init(handle: consuming OpaquePointer) {
      self.handle = consume handle
    }

    deinit { llama_model_free(self.handle) }

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
      public init(url: URL, parameters: LlamaModelParameters = LlamaModelParameters()) throws {
        try self.init(path: url.path(), parameters: parameters)
      }
    }
  #endif

  // MARK: - LlamaModelMetadata

  /// Read-only metadata from a llama.cpp model.
  ///
  /// Initialize `LlamaBackend` before loading metadata from a GGUF path.
  public struct LlamaModelMetadata: ~Copyable, Sendable {
    private let model: LlamaModel

    public init(contentsOfGGUF path: String) throws {
      let model = try LlamaModel(path: path, parameters: LlamaModelParameters(vocabularyOnly: true))
      self.init(model: model)
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

#endif
