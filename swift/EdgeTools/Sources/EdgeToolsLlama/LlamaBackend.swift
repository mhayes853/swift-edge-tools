#if canImport(CLlama)
  import CLlama

  /// The process-wide llama.cpp backend lifecycle.
  public enum LlamaBackend {
    /// Initializes llama.cpp once at the start of the process.
    public static func initialize() {
      llama_backend_init()
    }

    /// Frees llama.cpp once at the end of the process.
    public static func free() {
      llama_backend_free()
    }
  }
#endif
