#if Llama && canImport(EdgeToolsLlama)
  import EdgeToolsLlama

  extension LlamaRuntimeError.Code {
    static let tokenizationFailed = Self(rawValue: "tokenization-failed")
    static let vocabularyUnavailable = Self(rawValue: "vocabulary-unavailable")
  }
#endif
