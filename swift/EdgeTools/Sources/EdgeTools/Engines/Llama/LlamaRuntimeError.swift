#if Llama && canImport(EdgeToolsLlama)
  import EdgeToolsLlama

  extension LlamaRuntimeError.Code {
    static let contextCreationFailed = Self(rawValue: "context-creation-failed")
    static let contextLengthExceeded = Self(rawValue: "context-length-exceeded")
    static let decodeFailed = Self(rawValue: "decode-failed")
    static let multimodalProjectorLoadFailed =
      Self(rawValue: "multimodal-projector-load-failed")
    static let multimodalProcessingFailed = Self(rawValue: "multimodal-processing-failed")
  }
#endif
