#if canImport(CLlama)
  import CLlama

  /// An owned process-wide llama.cpp backend lifecycle.
  public struct LlamaBackend: ~Copyable {
    /// Initializes llama.cpp for the lifetime of this value.
    public init() {
      llama_backend_init()
    }

    deinit {
      llama_backend_free()
    }
  }
#endif
