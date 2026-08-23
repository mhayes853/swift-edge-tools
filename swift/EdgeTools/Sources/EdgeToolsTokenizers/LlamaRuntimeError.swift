#if Llama && canImport(EdgeToolsLlama)
  import EdgeToolsLlama

  extension LlamaRuntimeError.Code {
    public static let tokenizationFailed = Self(rawValue: "tokenization-failed")
    public static let vocabularyUnavailable = Self(rawValue: "vocabulary-unavailable")
  }
#endif
