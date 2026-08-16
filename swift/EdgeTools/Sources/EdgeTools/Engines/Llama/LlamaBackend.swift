#if Llama && canImport(CLlama)
  import CLlama

  // MARK: - LlamaBackend

  /// Process-wide llama.cpp backend lifecycle for the vendored build.
  public enum LlamaBackend {
    /// Initializes the llama.cpp backend once for the process.
    public static func initialize() {
      llama_backend_init()
    }

    /// Frees the llama.cpp backend.
    public static func shutdown() {
      llama_backend_free()
    }

    /// A description of the compiled backend features.
    public static var systemInfo: String {
      String(cString: llama_print_system_info())
    }

    /// True when the vendored build carries the cactus handoff probe API.
    public static var supportsHybridProbe: Bool {
      true
    }
  }
#endif
